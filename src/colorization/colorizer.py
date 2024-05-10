import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm 
import os

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from diffusers import DiffusionPipeline


import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.colorization.views import ColorLView, ColorABView
from src.colorization.samplers import sample_stage_1, sample_stage_2



class FactorizedColorization(nn.Module):
    '''Colorization diffusion model by Factorized Diffusion
    '''
    def __init__(
        self, 
        inverse_color=False,
        **kwargs
    ):
        super().__init__()

        # Make DeepFloyd IF stage I
        self.stage_1 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-M-v1.0",
            variant="fp16",
            torch_dtype=torch.float16
        )
        self.stage_1.enable_model_cpu_offload()

        # Make DeepFloyd IF stage II
        self.stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-M-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        self.stage_2.enable_model_cpu_offload()
        
        # if inverse the gray scale
        self.inverse_color = inverse_color

        # get views
        self.views = [ColorLView(), ColorABView()]

        self.num_inference_steps = kwargs.get("num_inference_steps", 30)
        self.guidance_scale = kwargs.get("guidance_scale", 10.0)
        self.start_diffusion_step = kwargs.get("start_diffusion_step", 0)
        self.noise_level = kwargs.get("noise_level", 50)

    
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # Get prompt embeddings (need two, because code is designed for 
        # two components: L and ab)
        prompts = [prompt] * 2
        prompt_embeds = [self.stage_1.encode_prompt(p) for p in prompts]
        prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
        prompt_embeds = torch.cat(prompt_embeds)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds)  # These are just null embeds
        return prompt_embeds, negative_prompt_embeds
    
    def forward(
        self,
        gray_im, 
        prompt, 
        num_inference_steps=None, 
        guidance_scale=None, 
        start_diffusion_step=None, 
        noise_level=None, 
        generator=None
    ):  
        # 1. overwrite the hyparams if provided
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        start_diffusion_step = self.start_diffusion_step if start_diffusion_step is None else start_diffusion_step
        noise_level = self.noise_level if noise_level is None else noise_level

        # 2. prepare the text embeddings
        prompt_embeds, negative_prompt_embeds = self.get_text_embeds(prompt)

        # import pdb; pdb.set_trace()

        # 3. prepare grayscale image
        _, height, width = gray_im.shape
        if self.inverse_color:
            gray_im = 1.0 - gray_im
        
        gray_im = gray_im * 2.0 - 1 # normalize the pixel value

        # 4. Sample 64x64 image
        image = sample_stage_1(
            self.stage_1, 
            prompt_embeds,
            negative_prompt_embeds,
            self.views,
            height=height // 4,
            width=width // 4,
            fixed_im=gray_im,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            start_diffusion_step=start_diffusion_step
        )

        # 5. Sample 256x256 image, by upsampling 64x64 image
        image = sample_stage_2(
            self.stage_2,
            image,
            prompt_embeds,
            negative_prompt_embeds, 
            self.views,
            height=height,
            width=width,
            fixed_im=gray_im,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            generator=generator
        )

        # 6. return the final image
        image = image / 2 + 0.5 
        return image


# python colorizer.py --name colorize.castle.full --gray_im_path ./imgs/castle.full.png --prompt "a colorful photo of a white castle with bell towers" --num_samples 4 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7

# python colorizer.py --name colorize.racing.full --gray_im_path ./imgs/racing.full.png --prompt "a colorful photo of a auto racing game" --num_samples 4 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7

# python colorizer.py --name colorize.tiger.full --gray_im_path ./imgs/tiger.full.png --prompt "a colorful photo of a tigers" --num_samples 4 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7

# python colorizer.py --name colorize.dog.full --gray_im_path ./imgs/dog.full.png --prompt "a colorful photo of puppies on green grass" --num_samples 4 --guidance_scale 10.0 --num_inference_steps 30 --start_diffusion_step 7

# python colorizer.py --name colorize.spec.full --gray_im_path ./imgs/spec.full.png --prompt "a colorful photo of kittens" --num_samples 8 --guidance_scale 10.0 --num_inference_steps 30 --start_diffusion_step 0

if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gray_im_path", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str, help='Prompts to use for colorization')
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--noise_level", type=int, default=50, help='Noise level for stage 2')
    parser.add_argument("--start_diffusion_step", type=int, default=7, help='What step to start the diffusion process')


    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # create diffusion colorization instance 
    colorizer = FactorizedColorization(
        inverse_color=False,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        start_diffusion_step=args.start_diffusion_step,
        noise_level=args.noise_level,
    ).to(device)

    # Load gray image
    gray_im = Image.open(args.gray_im_path)
    gray_im = TF.to_tensor(gray_im).to(device)

    save_dir = os.path.join('results', args.name)
    os.makedirs(save_dir, exist_ok=True)

    # Sample illusions
    for i in tqdm(range(args.num_samples), desc="Sampling images"):
        generator = torch.manual_seed(args.seed + i)
        image = colorizer(gray_im, args.prompt, generator=generator)
        save_image(image, f'{save_dir}/{i:04}.png', padding=0)
