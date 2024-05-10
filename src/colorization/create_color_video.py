import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import glob 
import os
from omegaconf import OmegaConf, DictConfig, open_dict

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.colorization.colorizer import FactorizedColorization
from src.utils.animation_with_text import create_animation_with_text


# python src/colorization/create_color_video.py --sample_dir logs/soundify-denoise/colorization/bell_example_29 --prompt "a colorful photo of a castle with bell towers" --num_samples 4 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7

# python src/colorization/create_color_video.py --sample_dir logs/soundify-denoise/colorization/bell_example_005 --prompt "a colorful photo of a castle with bell towers" --num_samples 12 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7


# python src/colorization/create_color_video.py --sample_dir logs/soundify-denoise/colorization/tiger_example_02 --prompt "a colorful photo of a tigers" --num_samples 4 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7

# python src/colorization/create_color_video.py --sample_dir logs/soundify-denoise/colorization/tiger_example_06 --prompt "a colorful photo of a tigers" --num_samples 8 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7


# python src/colorization/create_color_video.py --sample_dir logs/soundify-denoise/colorization/race_example_002 --prompt "a colorful photo of a auto racing game" --num_samples 12 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7

# python src/colorization/create_color_video.py --sample_dir logs/soundify-denoise/colorization/bird_example_40 --prompt "a blooming garden with many birds" --num_samples 4 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7

# python src/colorization/create_color_video.py --sample_dir logs/soundify-denoise/colorization/kitten_example_08 --prompt "a colorful photo of kittens" --num_samples 4 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7

# python src/colorization/create_color_video.py --sample_dir logs/soundify-denoise/colorization/kitten_example_08_v2 --prompt "a colorful photo of kittens with blue eyes and pink noses" --num_samples 16 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7

# python src/colorization/create_color_video.py --sample_dir logs/soundify-denoise/colorization/dog_example_06 --prompt "a colorful photo of dogs" --num_samples 4 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7

# python src/colorization/create_color_video.py --sample_dir logs/soundify-denoise/colorization/train_example_02 --prompt "a colorful photo of a long train" --num_samples 4 --guidance_scale 10 --num_inference_steps 30 --start_diffusion_step 7

if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str, help='Prompts to use for colorization')
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--depth", type=int, default=0)
    # parser.add_argument('--no_colormap', default=False, action='store_true')
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


    sample_dirs = glob.glob(f"{args.sample_dir}" + "/*" * args.depth)
    sample_dirs.sort()

    # read sample dir 
    for sample_dir in sample_dirs:
        gray_im_path = f'{sample_dir}/img.png'
        spec = f'{sample_dir}/spec_colormap.png'
        if not os.path.exists(spec): 
            spec = f'{sample_dir}/spec.png'
        
        audio = f'{sample_dir}/audio.wav'
        config_path = f'{sample_dir}/config.yaml'
        cfg = OmegaConf.load(config_path)
        image_prompt = args.prompt
        audio_prompt = cfg.trainer.audio_prompt

        with open_dict(cfg):
            cfg.trainer.colorization_prompt = args.prompt
        OmegaConf.save(cfg, config_path)

        # Load gray image
        gray_im = Image.open(gray_im_path)
        gray_im = TF.to_tensor(gray_im).to(device)

        img_save_dir = os.path.join(sample_dir, 'colorized_imgs')
        os.makedirs(img_save_dir, exist_ok=True)

        video_save_dir = os.path.join(sample_dir, 'colorized_videos')
        os.makedirs(video_save_dir, exist_ok=True)

        # Sample illusions
        for i in tqdm(range(args.num_samples), desc="Sampling images"):
            generator = torch.manual_seed(args.seed + i)
            image = colorizer(gray_im, args.prompt, generator=generator)
            img_save_path = f'{img_save_dir}/{i:04}.png'
            save_image(image, img_save_path, padding=0)

            video_save_path = f'{video_save_dir}/{i:04}.mp4'
            create_animation_with_text(img_save_path, spec, audio, video_save_path, image_prompt, audio_prompt)

