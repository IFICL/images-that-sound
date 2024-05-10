import argparse
from PIL import Image
from tqdm import tqdm
import os
import glob
import soundfile as sf
from omegaconf import OmegaConf, DictConfig

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from lightning import seed_everything

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.evaluator.clap import CLAPEvaluator
from src.evaluator.clip import CLIPEvaluator
from src.utils.consistency_check import griffin_lim


def bootstrap_confidence_intervals(data, num_bootstraps=10000):
    # Bootstrap resampling
    bootstrap_samples = np.random.choice(data, size=(num_bootstraps, data.shape[0]), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)

    # Compute confidence interval
    confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])

    # Calculate point estimate (mean)
    sample_mean = np.mean(data)

    # Calculate margin of error
    margin_of_error = (confidence_interval[1] - confidence_interval[0]) / 2

    return sample_mean, margin_of_error


def eval(args):
    seed_everything(2024, workers=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tqdm.write("Preparing samples to be evaluated...")
    eval_samples = glob.glob(f'{args.eval_path}/*/results/*')
    eval_samples.sort()

    if args.max_sample != -1:
        eval_samples = eval_samples[:args.max_sample]

    tqdm.write("Instantiating audio evaluator...")
    audio_evaluator = CLAPEvaluator().to(device)

    tqdm.write("Instantiating visual evaluator...")
    visual_evaluator = CLIPEvaluator().to(device)

    clip_scores = []
    clap_scores = []

    for sample in tqdm(eval_samples, desc="Evaluating"):
        clip_score, clap_score = evaluate_single_sample(sample, audio_evaluator, visual_evaluator, device, use_griffin_lim=args.use_griffin_lim)

        clip_scores.append(clip_score)
        clap_scores.append(clap_score)
    
    clip_scores = np.array(clip_scores)
    clap_scores = np.array(clap_scores)

    # Choose a multiplier (e.g., for 95% confidence interval, multiplier is approximately 1.96)
    confidence_multiplier = 1.96
    n = clip_scores.shape[0]

    avg_clip_score = clip_scores.mean()
    # _, clip_error = bootstrap_confidence_intervals(clip_scores)
    clip_error = confidence_multiplier * np.std(clip_scores) / np.sqrt(n)
    tqdm.write(f"Averaged CLIP score: {avg_clip_score * 100} | margin of error: {clip_error * 100}")

    avg_clap_score = clap_scores.mean()
    # _, clap_error = bootstrap_confidence_intervals(clap_scores)
    clap_error = confidence_multiplier * np.std(clap_scores) / np.sqrt(n)
    tqdm.write(f"Averaged CLAP score: {avg_clap_score * 100} | margin of error: {clap_error * 100}")


def evaluate_single_sample(sample_dir, audio_evaluator, visual_evaluator, device, use_griffin_lim=False):
    # import pdb; pdb.set_trace()
    # read sample dir 
    gray_im_path = f'{sample_dir}/spec.png'
    audio_path = f'{sample_dir}/audio.wav'
    config_path = f'{sample_dir}/config.yaml'
    cfg = OmegaConf.load(config_path)
    image_prompt = cfg.trainer.image_prompt
    audio_prompt = cfg.trainer.audio_prompt

    # Load gray image and evaluate
    gray_im = Image.open(gray_im_path)
    gray_im = TF.to_tensor(gray_im).to(device)
    spec = gray_im.detach().cpu()
    gray_im = gray_im.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    gray_im = gray_im.unsqueeze(0)

    clip_score = visual_evaluator(gray_im, image_prompt)

    # load audio waveform and evaluate
    audio, sr = sf.read(audio_path)

    if use_griffin_lim:
        # import pdb; pdb.set_trace()
        audio = griffin_lim(spec, audio)

    clap_score = audio_evaluator(audio_prompt, audio, sampling_rate=sr)
    
    return clip_score, clap_score

# python src/evaluator/eval.py --eval_path "logs/Evaluation/auffusion"
# python src/evaluator/eval.py --eval_path "logs/Evaluation/AV-IF-SDS-V2"
# python src/evaluator/eval.py --eval_path "logs/Evaluation/AV-Denoise-cfg7.5"
# python src/evaluator/eval.py --eval_path "logs/Evaluation/AV-Denoise-notime"


if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", required=True, type=str)
    parser.add_argument('--use_griffin_lim', default=False, action='store_true')
    parser.add_argument("--max_sample", type=int, default=-1)

    args = parser.parse_args()
    
    eval(args)