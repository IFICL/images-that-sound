import numpy as np 
import os
import glob
import shutil

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.pylogger import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)


def select_top_k_ranking(cfg, clip_scores, clap_scores):
    top_ranks = cfg.trainer.get("top_ranks", 0.2)
    clap_scores = np.array(clap_scores)
    clip_scores = np.array(clip_scores)
    top_ranks = int(clip_scores.shape[0] * top_ranks)

    # Get top-k indices for each array
    top_k_clap_indices = np.argsort(clap_scores)[::-1][:top_ranks]
    top_k_clip_indices = np.argsort(clip_scores)[::-1][:top_ranks]

    # Find the joint indices of top-k indices
    joint_indices = set(top_k_clap_indices).intersection(set(top_k_clip_indices))
    joint_indices = list(joint_indices)

    ori_dir = os.path.join(cfg.output_dir, 'results')
    examples = glob.glob(f'{ori_dir}/*')
    examples.sort()
    selected_dir = os.path.join(cfg.output_dir, 'results_selected')
    os.makedirs(selected_dir, exist_ok=True)
    
    log.info(f"Selected {len(joint_indices)} examples.")
    for ind in joint_indices: 
        example_dir_path = examples[ind]
        dir_name = example_dir_path.split('/')[-1]
        save_dir_path = os.path.join(selected_dir, dir_name)
        shutil.copytree(example_dir_path, save_dir_path)
    return 

def select_top_k_clip_ranking(cfg, clip_scores):
    # import pdb; pdb.set_trace()
    top_ranks = cfg.trainer.get("top_ranks", 0.1)
    clip_scores = np.array(clip_scores)
    top_ranks = int(clip_scores.shape[0] * top_ranks)

    # Get top-k indices
    top_k_clip_indices = np.argsort(clip_scores)[::-1][:top_ranks]

    ori_dir = os.path.join(cfg.output_dir, 'results')
    examples = glob.glob(f'{ori_dir}/*')
    examples.sort()
    selected_dir = os.path.join(cfg.output_dir, 'results_selected')
    os.makedirs(selected_dir, exist_ok=True)
    
    log.info(f"Selected {len(top_k_clip_indices)} examples.")
    for ind in top_k_clip_indices: 
        example_dir_path = examples[ind]
        dir_name = example_dir_path.split('/')[-1]
        save_dir_path = os.path.join(selected_dir, dir_name)
        shutil.copytree(example_dir_path, save_dir_path)
    return 
