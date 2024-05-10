import os
from typing import Optional
from tqdm import tqdm
import warnings
import soundfile as sf
import numpy as np
import shutil
import matplotlib.pyplot as plt

import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn as nn
from transformers import logging
from lightning import seed_everything

from torchvision.utils import save_image


import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.rich_utils import print_config_tree
from src.utils.animation_with_text import create_animation_with_text, create_single_image_animation_with_text
from src.utils.pylogger import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)


def save_model(output_dir, step, net):
    save_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'checkpoint_latest.pth.tar')
    torch.save(
        {
            'step': step,
            'state_dict': net.state_dict(),
        }, 
        path
    )

def save_audio(audio, save_path):
    sf.write(save_path, audio, samplerate=16000)


def encode_prompt(prompt, diffusion_guidance, device, time_repeat=1):
    '''Encode text prompts into embeddings 
    '''
    prompts = [prompt] * time_repeat
    null_prompts = [''] * time_repeat

    # Prompts -> text embeds
    cond_embeds = diffusion_guidance.get_text_embeds(prompts, device) # [B, 77, 768]
    uncond_embeds = diffusion_guidance.get_text_embeds(null_prompts, device) # [B, 77, 768]
    text_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0) # [2 * B, 77, 768]
    return text_embeds



@hydra.main(version_base="1.3", config_path="../configs/main_sds", config_name="main.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for training
    """

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")
        logging.set_verbosity_error()
    
    if cfg.extras.get("print_config"):
        print_config_tree(cfg, resolve=True)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    log.info(f"Instantiating Image Learner model <{cfg.image_learner._target_}>")
    image_learner = hydra.utils.instantiate(cfg.image_learner).to(device)

    log.info(f"Instantiating Image Diffusion model <{cfg.image_diffusion_guidance._target_}>")
    image_diffusion_guidance = hydra.utils.instantiate(cfg.image_diffusion_guidance).to(device)

    log.info(f"Instantiating Audio Diffusion guidance model <{cfg.audio_diffusion_guidance._target_}>")
    audio_diffusion_guidance = hydra.utils.instantiate(cfg.audio_diffusion_guidance).to(device)

    # create optimizer 
    log.info(f"Instantiating optimizer <{cfg.optimizer._target_}>")
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    optimizer = optimizer(params=image_learner.parameters())

    # create transformation
    log.info(f"Instantiating image transformation <{cfg.image_transformation._target_}>")
    image_transformation = hydra.utils.instantiate(cfg.image_transformation).to(device)

    log.info(f"Instantiating audio transformation <{cfg.audio_transformation._target_}>")
    audio_transformation = hydra.utils.instantiate(cfg.audio_transformation).to(device)

    log.info(f"Starting training!")
    trainer(cfg, image_learner, image_diffusion_guidance, audio_diffusion_guidance, optimizer, image_transformation, audio_transformation, device)


def trainer(cfg, image_learner, image_diffusion, audio_diffusion, optimizer, image_transformation, audio_transformation, device):
    image_guidance_scale, audio_guidance_scale = cfg.trainer.image_guidance_scale, cfg.trainer.audio_guidance_scale
    # image_weight, audio_weight = cfg.trainer.image_weight, cfg.trainer.audio_weight
    image_start_step = cfg.trainer.get("image_start_step", 0)
    audio_start_step = cfg.trainer.get("audio_start_step", 0)

    accumulate_grad_batches = cfg.trainer.get("accumulate_grad_batches", 1)
    use_colormap = cfg.trainer.get("use_colormap", False)
    crop_image = cfg.trainer.get("crop_image", False)
    
    image_text_embeds = encode_prompt(cfg.trainer.image_prompt, image_diffusion, device, time_repeat=cfg.trainer.batch_size)
    audio_text_embeds = encode_prompt(cfg.trainer.audio_prompt, audio_diffusion, device, time_repeat=1)

    image_learner.train()
    for step in tqdm(range(cfg.trainer.num_iteration), desc="Training"):
        # import pdb; pdb.set_trace()
        image = image_learner() # [C, H, W]
        images = image.unsqueeze(0) # (1, C, H, W)
        
        # perform image guidance
        rgb_images = image_transformation(images) # (B, C, h, w)

        # perform audio guidance
        spec_images = audio_transformation(images) # (1, 1, H, W)

        if step >= image_start_step:
            image_weight = cfg.trainer.image_weight
            image_loss = image_diffusion.train_step(image_text_embeds, rgb_images, guidance_scale=image_guidance_scale, grad_scale=1)
        else:
            image_weight = 0.0
            image_loss = torch.tensor(0.0).to(device)
        
        if step >= audio_start_step:
            audio_weight = cfg.trainer.audio_weight
            audio_loss = audio_diffusion.train_step(audio_text_embeds, spec_images, guidance_scale=audio_guidance_scale, grad_scale=1)
        else:
            audio_weight = 0.0
            audio_loss = torch.tensor(0.0).to(device)
        
        loss = image_weight * image_loss + audio_weight * audio_loss
        loss = loss / accumulate_grad_batches
        loss.backward()

        # apply gradient accumulation
        if (step + 1) % accumulate_grad_batches == 0 or (step + 1) == cfg.trainer.num_iteration:
            optimizer.step()
            optimizer.zero_grad()

        tqdm.write(f"Iteration: {step+1}/{cfg.trainer.num_iteration}, loss: {loss.item():.4f} | visual loss: {image_loss.item():.4f} | audio loss: {audio_loss.item():.4f}")

        if (step + 1) % cfg.trainer.save_step == 0:
            save_model(cfg.output_dir, step, image_learner)
        
        if (step + 1) % cfg.trainer.visualize_step == 0:
            img_save_dir = os.path.join(cfg.output_dir, 'image_results')
            os.makedirs(img_save_dir, exist_ok=True)
            img_save_path = os.path.join(img_save_dir, f'img_{str(step+1).zfill(6)}.png')
            save_image(image, img_save_path)

            spec_save_dir = os.path.join(cfg.output_dir, 'spec_results')
            os.makedirs(spec_save_dir, exist_ok=True)
            spec_save_path = os.path.join(spec_save_dir, f'spec_{str(step+1).zfill(6)}.png')
            save_image(spec_images.squeeze(0), spec_save_path)

            audio_save_dir = os.path.join(cfg.output_dir, 'audio_results')
            os.makedirs(audio_save_dir, exist_ok=True)
            audio_save_path = os.path.join(audio_save_dir, f'audio_{str(step+1).zfill(6)}.wav')

            audio = audio_diffusion.spec_to_audio(spec_images.squeeze(0))
            audio = np.ravel(audio)
            save_audio(audio, audio_save_path)

    # obtain final results
    img = image_learner() # [C, H, W]
    spec = audio_transformation(img.unsqueeze(0)).squeeze(0) # (1, H, W)
    audio = audio_diffusion.spec_to_audio(spec)
    audio = np.ravel(audio)

    if crop_image:
        pixel = 32
        audio_length = int(pixel / image_learner.width * audio.shape[0])
        img = img[..., :-pixel]
        spec = spec[..., :-pixel] 
        audio = audio[:-audio_length]   

    # save the final results
    sample_dir = os.path.join(cfg.output_dir, 'results', f'final')
    os.makedirs(sample_dir, exist_ok=True)

    # save config
    cfg_path = os.path.join(cfg.output_dir, '.hydra', 'config.yaml')
    cfg_save_path = os.path.join(sample_dir, 'config.yaml')
    shutil.copyfile(cfg_path, cfg_save_path)

    # save image
    img_save_path = os.path.join(sample_dir, f'img.png')
    save_image(img, img_save_path)

    # save audio
    audio_save_path = os.path.join(sample_dir, f'audio.wav')
    save_audio(audio, audio_save_path)

    # save spec
    spec_save_path = os.path.join(sample_dir, f'spec.png')
    save_image(spec.mean(dim=0, keepdim=True), spec_save_path)

    if use_colormap:
        spec_save_path = os.path.join(sample_dir, f'spec_colormap.png')
        spec_colormap = spec.mean(dim=0).detach().cpu().numpy()
        plt.imsave(spec_save_path, spec_colormap, cmap='gray')

    # log.info("Generating video ...")
    # save video
    video_output_path = os.path.join(cfg.output_dir, 'video.mp4')

    if image_learner.num_channels == 1:
        create_single_image_animation_with_text(spec_save_path, audio_save_path, video_output_path, cfg.trainer.image_prompt, cfg.trainer.audio_prompt)
    else:
        create_animation_with_text(img_save_path, spec_save_path, audio_save_path, video_output_path, cfg.trainer.image_prompt, cfg.trainer.audio_prompt)
    # log.info("Generated video.")


if __name__ == "__main__":
    main()
