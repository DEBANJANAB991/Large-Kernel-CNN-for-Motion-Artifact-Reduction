#!/usr/bin/env python3
"""
Module: Image-Domain Inference and Evaluation for CT Motion Artifact Reduction

This script performs inference and quantitative evaluation of trained deep learning
models for motion artifact reduction in CT images.

Key Features:
- Supports multiple architectures (MR-LKV, UNet, RepLKNet, SwinIR, Restormer)
- Ensures fair evaluation using unseen test data
- Maintains consistency with training normalization strategy
- Handles model-specific input constraints via padding
- Provides efficiency benchmarks
"""
import sys
from pathlib import Path
import argparse
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ptflops import get_model_complexity_info

# ============================================================
# PATH SETUP
# ============================================================
SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
TABLE_DIR   = RESULTS_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

from model_wrapper import build_model
from config.config import RECONSTRUCTED_CT_VOLUME, CKPT_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def run_inference(model_name):

    print(f"\nRunning IMAGE DOMAIN inference for: {model_name}")

    # -------------------------
    # Load model
    # -------------------------
    model = build_model(model_name).to(DEVICE)
    ckpt_path = Path(CKPT_DIR) / model_name / "best_model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {k: v for k, v in state_dict.items() if "attn_mask" not in k}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # -------------------------
    # Paths
    # -------------------------
    ROOT = Path(RECONSTRUCTED_CT_VOLUME)

    INPUT_DIR = ROOT / "artifact"
    GT_DIR    = ROOT / "clean"
    PRED_DIR  = ROOT / model_name

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.npy"))
    print(f"Found {len(files)} volumes")

    if not files:
        print("No input files found")
        return

    # -------------------------
    # FLOPs + Params
    # -------------------------
    sample = np.load(files[0])
    _, H, W = sample.shape
    
    macs, _ = get_model_complexity_info(
        model,
        (1, H, W),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )

    flops_gmac = macs / 1e9
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"Params: {params_m:.2f} M")
    print(f"MACs   : {flops_gmac:.2f} G")
 
        

    # -------------------------
    # Inference
    # -------------------------
    results = []
    times = []

    with torch.no_grad():

        for f in tqdm(files, desc="Inference"):

            gt_path = GT_DIR / f.name
            if not gt_path.exists():
                print(f"Missing GT: {f.name}")
                continue

            artifact_vol = np.load(f).astype(np.float32)
            gt_vol       = np.load(gt_path).astype(np.float32)

            Nz = artifact_vol.shape[0]

            pred_volume = []

            psnr_list, ssim_list, mae_list, rmse_list = [], [], [], []

            for z in range(Nz):

                artifact = artifact_vol[z]
                gt       = gt_vol[z]

                # Normalize
                s_min = artifact.min()
                s_max = artifact.max()
                denom = s_max - s_min + 1e-12
                # Normalize using artifact statistics to maintain consistency with training distribution
                artifact_norm = (artifact - s_min) / denom
                gt_norm       = np.clip((gt - s_min) / denom, 0, 1)

                x = torch.from_numpy(artifact_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

                # Padding input to match model constraints
                _, _, h, w = x.shape
                multiple = 16
                pad_h = (multiple - h % multiple) % multiple
                pad_w = (multiple - w % multiple) % multiple

                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

                # Timing
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()

                pred = model(x)

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                end = time.time()

                times.append(end - start)

                pred = torch.clamp(pred, 0, 1)
                pred = pred.squeeze().cpu().numpy()

                pred = pred[:h, :w]

                # Metrics
                psnr_val = peak_signal_noise_ratio(gt_norm, pred, data_range=1.0)
                ssim_val = structural_similarity(gt_norm, pred, data_range=1.0)
                mae_val  = mean_absolute_error(gt_norm.flatten(), pred.flatten())
                rmse_val = np.sqrt(mean_squared_error(gt_norm.flatten(), pred.flatten()))

                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                mae_list.append(mae_val)
                rmse_list.append(rmse_val)

                # Denormalize
                pred = pred * denom + s_min

                pred_volume.append(pred)

            # Save volume
            pred_volume = np.stack(pred_volume, axis=0)
            np.save(PRED_DIR / f.name, pred_volume)

            # Store metrics per volume
            results.append([
                f.name,
                np.mean(psnr_list),
                np.mean(ssim_list),
                np.mean(mae_list),
                np.mean(rmse_list)
            ])

    # -------------------------
    # Summary
    # -------------------------
    #avg_time_ms = (sum(times) / len(times)) * 1000
    avg_time_ms = (sum(times) / max(1, len(times))) * 1000

    df = pd.DataFrame(results, columns=["file", "PSNR", "SSIM", "MAE", "RMSE"])
    df["Inference_time_ms"] = avg_time_ms
    df["Params_M"] = params_m
    df["MACs_G"] = flops_gmac

    csv_path = TABLE_DIR / f"image_domain_metrics_{model_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Avg inference time: {avg_time_ms:.2f} ms")

    print(f"\nSaved → {csv_path}")
    print(f"Predictions → {PRED_DIR}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["mr_lkv", "unet", "replknet", "swinir", "restormer"],
        default="mr_lkv"
    )
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        for m in ["mr_lkv", "unet", "replknet", "swinir", "restormer"]:
            run_inference(m)
    else:
        run_inference(args.model)


if __name__ == "__main__":
    main()