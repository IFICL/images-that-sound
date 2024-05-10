import os
from typing import Optional
from tqdm import tqdm
import warnings
import soundfile as sf
import numpy as np
import shutil
import glob
import copy
import matplotlib.pyplot as plt

import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
import torch
import torch.nn as nn
from transformers import logging
from lightning import seed_everything

from torchvision.utils import save_image


import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.rich_utils import print_config_tree
from src.utils.animation_with_text import create_animation_with_text, create_single_image_animation_with_text
from src.utils.re_ranking import select_top_k_ranking, select_top_k_clip_ranking
from src.utils.pylogger import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)



def save_audio(audio, save_path):
    sf.write(save_path, audio, samplerate=16000)


def encode_prompt(prompt, diffusion_guidance, device, negative_prompt='', time_repeat=1):
    '''Encode text prompts into embeddings 
    '''
    prompts = [prompt] * time_repeat
    negative_prompts = [negative_prompt] * time_repeat

    # Prompts -> text embeds
    cond_embeds = diffusion_guidance.get_text_embeds(prompts, device) # [B, 77, 768]
    uncond_embeds = diffusion_guidance.get_text_embeds(negative_prompts, device) # [B, 77, 768]
    text_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0) # [2 * B, 77, 768]
    return text_embeds

def estimate_noise(diffusion, latents, t, text_embeddings, guidance_scale): 
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    # predict the noise residual
    noise_pred = diffusion.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

    # perform guidance
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    return noise_pred



@hydra.main(version_base="1.3", config_path="../configs/main_imprint", config_name="main.yaml")
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

    log.info(f"Instantiating Image Diffusion model <{cfg.image_diffusion_guidance._target_}>")
    image_diffusion_guidance = hydra.utils.instantiate(cfg.image_diffusion_guidance).to(device)

    log.info(f"Instantiating Audio Diffusion guidance model <{cfg.audio_diffusion_guidance._target_}>")
    audio_diffusion_guidance = hydra.utils.instantiate(cfg.audio_diffusion_guidance).to(device)

    # create transformation
    log.info(f"Instantiating latent transformation <{cfg.latent_transformation._target_}>")
    latent_transformation = hydra.utils.instantiate(cfg.latent_transformation).to(device)

    # create audio evaluator
    if cfg.audio_evaluator: 
        log.info(f"Instantiating audio evaluator <{cfg.audio_evaluator._target_}>")
        audio_evaluator = hydra.utils.instantiate(cfg.audio_evaluator).to(device)
    else:
        audio_evaluator = None

    if cfg.visual_evaluator:
        log.info(f"Instantiating visual evaluator <{cfg.visual_evaluator._target_}>")
        visual_evaluator = hydra.utils.instantiate(cfg.visual_evaluator).to(device)
    else:
        visual_evaluator = None

    clip_scores = []
    clap_scores = []
    log.info(f"Starting sampling!")
    for idx in tqdm(range(cfg.trainer.num_samples), desc='Sampling'):
        clip_score, clap_score = create_sample(cfg, image_diffusion_guidance, audio_diffusion_guidance, latent_transformation, visual_evaluator, audio_evaluator, idx, device)
        clip_scores.append(clip_score)
        clap_scores.append(clap_score)

    # re-ranking by metrics
    enable_rank = cfg.trainer.get("enable_rank", False)
    if enable_rank:
        log.info(f"Starting re-ranking and selection!")
        select_top_k_ranking(cfg, clip_scores, clap_scores)
    
    enable_clip_rank = cfg.trainer.get("enable_clip_rank", False)
    if enable_clip_rank:
        log.info(f"Starting re-ranking and selection by CLIP score!")
        select_top_k_clip_ranking(cfg, clip_scores)

    log.info(f"Finished!")


@torch.no_grad()
def create_sample(cfg, image_diffusion, audio_diffusion, latent_transformation, visual_evaluator, audio_evaluator, idx, device):
    image_guidance_scale, audio_guidance_scale = cfg.trainer.image_guidance_scale, cfg.trainer.audio_guidance_scale
    height, width = cfg.trainer.img_height, cfg.trainer.img_width
    inverse_image = cfg.trainer.get("inverse_image", False)
    use_colormap = cfg.trainer.get("use_colormap", False)
    crop_image = cfg.trainer.get("crop_image", False)

    generator = torch.manual_seed(cfg.seed + idx)

    # obtain the image and spec for each modality's diffusion process
    image = image_diffusion.prompt_to_img(cfg.trainer.image_prompt, negative_prompts=cfg.trainer.image_neg_prompt, height=height, width=width, num_inference_steps=50, guidance_scale=image_guidance_scale, device=device, generator=generator)
    image = image.mean(dim=1) # make grayscale image

    spec = audio_diffusion.prompt_to_spec(cfg.trainer.audio_prompt, negative_prompts=cfg.trainer.audio_neg_prompt, height=height, width=width, num_inference_steps=100, guidance_scale=audio_guidance_scale, device=device, generator=generator)
    spec = spec.mean(dim=1) # make a single channel 

    # perform the naive baseline
    mag_ratio = cfg.trainer.get("mag_ratio", 0.5)
    if inverse_image:
        image = 1.0 - image
    image_mask = 1 - mag_ratio * image
    spec_new = spec * image_mask
    img = image
    # import pdb; pdb.set_trace()
    audio = audio_diffusion.spec_to_audio(spec_new)
    audio = np.ravel(audio)

    if crop_image:
        pixel = 32
        audio_length = int(pixel / width * audio.shape[0])
        img = img[..., :-pixel]
        spec = spec[..., :-pixel] 
        spec_new = spec_new[..., :-pixel] 
        audio = audio[:-audio_length]  

    # evaluate with CLIP
    if visual_evaluator is not None:
        clip_score = visual_evaluator(img.repeat(3, 1, 1).unsqueeze(0), cfg.trainer.image_prompt)
    else:
        clip_score = None

    # evaluate with CLAP
    if audio_evaluator is not None:
        clap_score = audio_evaluator(cfg.trainer.audio_prompt, audio)
    else:
        clap_score = None

    sample_dir = os.path.join(cfg.output_dir, 'results', f'example_{str(idx+1).zfill(3)}')
    os.makedirs(sample_dir, exist_ok=True)

    # import pdb; pdb.set_trace()
    # save config with example-specific information 
    cfg_save_path = os.path.join(sample_dir, 'config.yaml')
    current_cfg = copy.deepcopy(cfg)
    current_cfg.seed = cfg.seed + idx
    with open_dict(current_cfg):
        current_cfg.clip_score = clip_score
        current_cfg.clap_score = clap_score
    OmegaConf.save(current_cfg, cfg_save_path)
    
    # save image
    img_save_path = os.path.join(sample_dir, f'img.png')
    save_image(img, img_save_path)

    # save audio
    audio_save_path = os.path.join(sample_dir, f'audio.wav')
    save_audio(audio, audio_save_path)

    # save spec
    spec_save_path = os.path.join(sample_dir, f'spec_ori.png')
    save_image(spec, spec_save_path)

    spec_save_path = os.path.join(sample_dir, f'spec.png')
    save_image(spec_new, spec_save_path)

    # save spec with colormap (renormalize the spectrogram range)
    if use_colormap:
        spec_save_path = os.path.join(sample_dir, f'spec_colormap.png')
        spec_colormap = spec_new.mean(dim=0).cpu().numpy()
        plt.imsave(spec_save_path, spec_colormap, cmap='gray')

    
    # save video 
    video_output_path = os.path.join(sample_dir, f'video.mp4')
    if img.shape[-2:] == spec.shape[-2:]:
        create_single_image_animation_with_text(spec_save_path, audio_save_path, video_output_path, cfg.trainer.image_prompt, cfg.trainer.audio_prompt)
    else:
        create_animation_with_text(img_save_path, spec_save_path, audio_save_path, video_output_path, cfg.trainer.image_prompt, cfg.trainer.audio_prompt)
    return clip_score, clap_score


if __name__ == "__main__":
    main()
