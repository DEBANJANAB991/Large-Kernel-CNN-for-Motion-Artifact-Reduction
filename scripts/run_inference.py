#!/usr/bin/env python3

import sys
from pathlib import Path
import argparse
import time
import json
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

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
TABLE_DIR   = RESULTS_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

from models.model_wrapper import build_model

from config import (
    ART_SINOGRAM_2D,
    CLEAN_SINOGRAM_2D_TEST,
    PREDICTED_SINOGRAM_2D_TEST_v2,
    CKPT_DIR
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def run_inference(model_name):

    print(f"\nRunning inference for model: {model_name}")

    # -------------------------
    # Load model
    # -------------------------
    model     = build_model(model_name).to(DEVICE)
    ckpt_path = Path(CKPT_DIR) / model_name / "best_model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
   # model.load_state_dict(
    #    checkpoint["model"] if "model" in checkpoint else checkpoint
    #)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint


    state_dict = {k: v for k, v in state_dict.items() if "attn_mask" not in k}

    model.load_state_dict(state_dict, strict=False)
    torch.cuda.empty_cache()
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # -------------------------
    # Data paths
    # -------------------------
    in_dir   = Path(ART_SINOGRAM_2D)
    gt_dir   = Path(CLEAN_SINOGRAM_2D_TEST)
    pred_dir = Path(PREDICTED_SINOGRAM_2D_TEST_v2) / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.npy"))
    print(f"Found {len(files)} test samples")

    if not files:
        print(f"No files found in {in_dir}")
        return

    # -------------------------
    # Determine input size
    # -------------------------
    sample = np.load(files[0]).astype(np.float32)
    H, W   = sample.shape
    C      = 1
    print(f"Input size: ({H}, {W})")
    if model_name == "restormer":
        print("Computing FLOPs for Restormer at safe resolution (128x128)")
        flops_input = (1, 128, 128)
    elif model_name == "swinir":
        flops_input = (1, 128, 128)
        print("Computing FLOPs for SwinIR at 128x128 (training resolution)")
    else:
        flops_input = (C, H, W)
    # -------------------------
    # FLOPs + Params
    # -------------------------
    macs, _ = get_model_complexity_info(
        model,
        flops_input,   #(C, H, W),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    flops_gmac = macs / 1e9
    params_m   = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"Params : {params_m:.2f} M")
    if model_name == "swinir" or model_name == "restormer":
        H, W = 540, 800
        scale = (H * W) / (128 * 128)
        flops_gmac = flops_gmac* scale
        print(f"MACs   : {flops_gmac:.2f} G")
    else:
        print(f"MACs   : {flops_gmac:.2f} G")

    # -------------------------
    # Warm-up
    # -------------------------
    # Pad dummy to multiple of 16 for warm-up
    pad_h_dummy = (16 - H % 16) % 16
    pad_w_dummy = (16 - W % 16) % 16
    dummy = torch.randn(1, 1, H + pad_h_dummy, W + pad_w_dummy).to(DEVICE)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # -------------------------
    # Inference loop
    # -------------------------
    results = []
    times   = []

    with torch.no_grad():

        for f in tqdm(files, desc=f"{model_name} inference"):

            # Check GT exists
            gt_path = gt_dir / f.name
            if not gt_path.exists():
                print(f"Skipping {f.name}: missing GT file")
                continue

            # Load
            artifact = np.load(f).astype(np.float32)
            gt       = np.load(gt_path).astype(np.float32)

            # Normalize using artifact statistics
            s_min = artifact.min()
            s_max = artifact.max()
            denom = s_max - s_min + 1e-12
            gt_norm =gt
            artifact_norm = (artifact - s_min) / denom
            gt_norm = np.clip((gt - s_min) / denom, 0, 1)

            # Save normalization metadata for reconstruction
            meta = {"s_min": float(s_min), "s_max": float(s_max)}
            with open(pred_dir / f"{f.stem}.json", "w") as fp:
                json.dump(meta, fp)

            # To tensor
            x = torch.from_numpy(artifact_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

            
            # Model-specific padding
            # -------------------------
            _, _, h, w = x.shape

            if model_name == "restormer":
                multiple = 8
            #elif model_name == "swinir":
            #    multiple = 8   # also safer for SwinIR
            else:
                multiple = 16  # UNet / MR-LKV / RepLKNet

            pad_h = (multiple - h % multiple) % multiple
            pad_w = (multiple - w % multiple) % multiple

            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            # -------------------------
            # Timing
            # -------------------------
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            pred = model(x)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)

            # Post-process
            pred = torch.clamp(pred, 0, 1)
            pred = pred.squeeze().cpu().numpy()

            # Crop back to original size for ALL models
            pred = pred[:H, :W]

            # Shape check
            if gt_norm.shape != pred.shape:
                print(f"Shape mismatch — GT: {gt_norm.shape}, Pred: {pred.shape}")
                continue
            
            
            data_range = gt.max() - gt.min()

            # safety (avoid zero range)
            #if data_range < 1e-6:
            #    data_range = 1.0
            # -------------------------
            # Metrics on 2D sinogram slices
            # -------------------------
            psnr_val = peak_signal_noise_ratio(gt_norm, pred, data_range=1.0)
            ssim_val = structural_similarity(gt_norm, pred, data_range=1.0)
            mae_val  = mean_absolute_error(gt_norm.flatten(), pred.flatten())
            rmse_val = np.sqrt(mean_squared_error(gt_norm.flatten(), pred.flatten()))

            results.append([f.name, psnr_val, ssim_val, mae_val, rmse_val])

            # Save denormalized prediction for 3D reconstruction
            pred_denorm = pred * denom +s_min
            np.save(pred_dir / f.name, pred_denorm)

    # -------------------------
    # Summary
    # -------------------------
    if not results:
        print(f"No results for {model_name}")
        return

    avg_time_ms = (sum(times) / len(times)) * 1000
    print(f"\nAvg inference time: {avg_time_ms:.2f} ms")

    df = pd.DataFrame(results, columns=["file", "PSNR", "SSIM", "MAE", "RMSE"])
    df["Inference_time_ms"] = avg_time_ms
    df["Params_M"]          = params_m
    df["MACs_G"]            = flops_gmac

    csv_path = TABLE_DIR / f"metrics_{model_name}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nMean PSNR : {df['PSNR'].mean():.2f} dB")
    print(f"Mean SSIM : {df['SSIM'].mean():.4f}")
    print(f"Mean MAE  : {df['MAE'].mean():.6f}")
    print(f"Mean RMSE : {df['RMSE'].mean():.6f}")
    print(f"Metrics saved → {csv_path}")
    print(f"Predictions  → {pred_dir}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["mr_lkv", "unet", "replknet", "swinir", "restormer", "mr_lkv_l1"],
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