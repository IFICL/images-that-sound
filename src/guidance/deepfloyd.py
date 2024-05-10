from diffusers import IFPipeline, DDPMScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import seed_everything


class DeepfloydGuidance(nn.Module):
    def __init__(
        self,
        repo_id='DeepFloyd/IF-I-XL-v1.0', 
        fp16=True, 
        t_range=[0.02, 0.98],
        t_consistent=False,
        **kwargs
    ):
        super().__init__()

        self.repo_id = repo_id
        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = IFPipeline.from_pretrained(repo_id, torch_dtype=self.precision_t)

        self.unet = pipe.unet
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        self.pipe = pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.t_consistent = t_consistent

        self.register_buffer('alphas', self.scheduler.alphas_cumprod) # for convenience

    @torch.no_grad()
    def get_text_embeds(self, prompt, device):
        # prompt: [str]
        prompt = self.pipe._text_preprocessing(prompt, clean_caption=False)
        inputs = self.tokenizer(prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(device))[0]
        embeddings = embeddings.to(dtype=self.text_encoder.dtype, device=device)
        return embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, t=None, grad_scale=1):
        pred_rgb = pred_rgb.to(self.unet.dtype)
        # [0, 1] to [-1, 1] and make sure shape is [64, 64]
        images = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if t is None:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            if self.t_consistent: 
                t = torch.randint(self.min_step, self.max_step + 1, (1,), dtype=torch.long, device=images.device)
                t = t.repeat(images.shape[0])
            else:
                t = torch.randint(self.min_step, self.max_step + 1, (images.shape[0],), dtype=torch.long, device=images.device)
        else:
            t = t.to(dtype=torch.long, device=images.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)

            # pred noise
            model_input = torch.cat([images_noisy] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(model_input, tt, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (images - grad).detach()
        loss = 0.5 * F.mse_loss(images.float(), targets, reduction='sum') / images.shape[0]

        return loss


    @torch.no_grad()
    def produce_imgs(self, text_embeddings, height=64, width=64, num_inference_steps=50, guidance_scale=7.5):

        images = torch.randn((1, 3, height, width), device=text_embeddings.device, dtype=text_embeddings.dtype)
        images = images * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the image if we are doing classifier-free guidance to avoid doing two forward passes.
            model_input = torch.cat([images] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)

            # predict the noise residual
            noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            images = self.scheduler.step(noise_pred, t, images).prev_sample

        images = (images + 1) / 2

        return images

    def prompt_to_img(self, prompts, negative_prompts='', height=64, width=64, num_inference_steps=50, guidance_scale=7.5, device=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts, device) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts, device)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img
        imgs = self.produce_imgs(text_embeds, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':
    import argparse
    from PIL import Image
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='an oil paint of modern city, street view')
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--repo_id', type=str, default='DeepFloyd/IF-I-XL-v1.0', help="stable diffusion version")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--H', type=int, default=64)
    parser.add_argument('--W', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = DeepfloydGuidance(repo_id=opt.repo_id, fp16=opt.fp16).to(device)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps, device=device)

    # visualize image
    save_path = 'logs/test'
    os.makedirs(save_path, exist_ok=True)
    image = Image.fromarray(imgs[0], mode='RGB')
    image.save(os.path.join(save_path, f'{opt.prompt}.png'))