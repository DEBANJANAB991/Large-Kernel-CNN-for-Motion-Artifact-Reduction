#!/usr/bin/env python3
"""
Inference script for artifact reduction models.

Runs trained models on artifact sinograms and saves predicted clean sinograms.

Input:
    ARTIFACT_SINOGRAM_2D_TEST/*.npy

Output:
    PREDICTED_SINOGRAM_2D_TEST/<model_name>/*.npy
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from tqdm import tqdm

# ensure repo root imports work
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# import model builder
from models.model_wrapper import build_model

# config
from config import (
    ARTIFACT_SINOGRAM_2D_TEST_v2,
    PREDICTED_SINOGRAM_2D_TEST_v2,
    CKPT_DIR
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

def run_inference(model_name):

    print(f"\nRunning inference for model: {model_name}")

    # build model
    model = build_model(model_name).to(DEVICE)

    # checkpoint path
    if model_name == "replknet":
        ckpt_path = Path(CKPT_DIR) / "replknet" / "best_model.pth"
    else:
        ckpt_path = Path(CKPT_DIR) / model_name / "best_model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    model.eval()

    print(f"Loaded checkpoint from: {ckpt_path}")

    # input directory
    in_dir = Path(ARTIFACT_SINOGRAM_2D_TEST_v2)

    # output directory (per model)
    out_dir = Path(PREDICTED_SINOGRAM_2D_TEST_v2) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.npy"))

    print(f"Found {len(files)} sinograms to process")
    print(f"Saving predictions to: {out_dir}")

    # inference
    with torch.no_grad():

        for f in tqdm(files, desc=f"{model_name} inference"):

            sino = np.load(f).astype(np.float32)

            # normalize per sinogram
            s_min = sino.min()
            s_max = sino.max()

            sino_norm = (sino - s_min) / (s_max - s_min + 1e-12)

            # convert to tensor (1,1,H,W)
            x = torch.from_numpy(sino_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # prediction
            pred = model(x)

            pred = pred.squeeze().cpu().numpy()

            # denormalize
            pred = pred * (s_max - s_min) + s_min

            # save
            np.save(out_dir / f.name, pred)

    print(f"\nFinished inference for {model_name}")
    print(f"Predictions saved to: {out_dir}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(description="Run inference for trained models")

    parser.add_argument(
        "--model",
        choices=["mr_lkv", "unet", "replknet", "swinir", "restormer"],
        default="mr_lkv",
        help="Model to run inference with"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run inference for all models"
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    if args.all:

        models = ["mr_lkv", "unet", "replknet", "swinir", "restormer"]

        for m in models:
            run_inference(m)

    else:

        run_inference(args.model)


if __name__ == "__main__":
    main()