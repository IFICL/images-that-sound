from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available

# from .perpneg_utils import weighted_perpendicular_aggregator

from lightning import seed_everything

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.auffusion_converter import Generator, denormalize_spectrogram



class AuffusionGuidance(nn.Module):
    def __init__(
        self, 
        repo_id='auffusion/auffusion-full-no-adapter', 
        fp16=True,
        t_range=[0.02, 0.98],
        **kwargs
    ):
        super().__init__()

        self.repo_id = repo_id

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        self.vae, self.tokenizer, self.text_encoder, self.unet = self.create_model_from_pipe(repo_id, self.precision_t)
        self.scheduler = DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler", torch_dtype=self.precision_t)
        self.vocoder = Generator.from_pretrained(repo_id, subfolder="vocoder").to(dtype=self.precision_t)

        self.register_buffer('alphas_cumprod', self.scheduler.alphas_cumprod)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

    def create_model_from_pipe(self, repo_id, dtype):
        pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=dtype)
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
        return vae, tokenizer, text_encoder, unet

    @torch.no_grad()
    def get_text_embeds(self, prompt, device):
        # prompt: [str]
        # import pdb; pdb.set_trace()
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        prompt_embeds = self.text_encoder(inputs.input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        return prompt_embeds

    def train_step(self, text_embeddings, pred_spec, guidance_scale=100, as_latent=False, t=None, grad_scale=1, save_guidance_path:Path=None):
        # import pdb; pdb.set_trace()
        pred_spec = pred_spec.to(self.vae.dtype)

        if as_latent:
            latents = pred_spec
        else:    
            if pred_spec.shape[1] != 3:
                pred_spec = pred_spec.repeat(1, 3, 1, 1)

            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_spec)

        if t is None:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=latents.device)
        else:
            t = t.to(dtype=torch.long, device=latents.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas_cumprod[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents.device)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1 
                b = len(noise_pred)
                a_t = alphas[index].reshape(b,1,1,1).to(latents.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(latents.device)                
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                save_image(viz_images, save_guidance_path)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss


    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, generator=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.config.in_channels, height // 8, width // 8), generator=generator, dtype=self.unet.dtype).to(text_embeddings.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):
        latents = latents.to(self.vae.dtype)
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def spec_to_audio(self, spec):
        spec = spec.to(dtype=self.precision_t)
        denorm_spec = denormalize_spectrogram(spec)
        audio = self.vocoder.inference(denorm_spec)
        return audio

    def prompt_to_audio(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, device=None, generator=None):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts, device) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts, device)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        spec = imgs[0]
        denorm_spec = denormalize_spectrogram(spec)
        audio = self.vocoder.inference(denorm_spec)
        # # Img to Numpy
        # imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        # imgs = (imgs * 255).round().astype('uint8')

        return audio

    def prompt_to_spec(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, device=None, generator=None):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts, device) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts, device)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

if __name__ == '__main__':
    import numpy as np
    import argparse
    from PIL import Image
    import os
    import soundfile as sf

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='A kitten mewing for attention')
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--repo_id', type=str, default='auffusion/auffusion-full-no-adapter', help="stable diffusion version")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--H', type=int, default=256)
    parser.add_argument('--W', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--out_dir', type=str, default='logs/test')

    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = AuffusionGuidance(repo_id=opt.repo_id, fp16=opt.fp16)
    sd = sd.to(device)

    audio = sd.prompt_to_audio(opt.prompt, opt.negative, opt.H, opt.W, opt.steps, device=device)
    # import pdb; pdb.set_trace()
    # visualize audio
    save_folder = opt.out_dir
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f'{opt.prompt}.wav')
    sf.write(save_path, np.ravel(audio), samplerate=16000)