#!/usr/bin/env python3
"""
Evaluation: PSNR, SSIM, MAE, RMSE, VIF, LPIPS
Computed on reconstructed CT volumes for projection domain.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")


from config.config import RECONSTRUCTED_CT_VOLUME_PROJECTION_DOMAIN as RECONSTRUCTED_CT_VOLUME


# ============================================================
# PATHS
# ============================================================

CT_ROOT     = Path(RECONSTRUCTED_CT_VOLUME)
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- LPIPS availability ---------------- #
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# ---------------- VIF availability ---------------- #
try:
    from sewar.full_ref import vifp
    VIF_AVAILABLE = True
except ImportError:
    VIF_AVAILABLE = False

# ============================================================
# FIND VOLUME
# ============================================================

def find_volume(folder: str, patient_id: str):
    base_id = patient_id.split(" ")[0]
    vol_dir = CT_ROOT / folder
    for name in [
        f"{patient_id}.npy",
        f"{base_id}.npy",
        f"{patient_id} {patient_id}.npy",
        f"{base_id} {base_id}.npy",
    ]:
        p = vol_dir / name
        if p.exists():
            return p
    return None


# ============================================================
# NORMALIZE
# ============================================================

def normalize_pair(clean_vol: np.ndarray, pred_vol: np.ndarray):
    #Normalizes both using clean statistics to maintain consistency with training distribution
    v_min = clean_vol.min()
    v_max = clean_vol.max()
    denom = v_max - v_min + 1e-8
    clean_norm = (clean_vol - v_min) / denom
    pred_norm  = np.clip((pred_vol - v_min) / denom, 0, 1)
    #pred_norm  = (pred_vol  - v_min) / denom
    return clean_norm, pred_norm


# ============================================================
# LPIPS SETUP 
# ============================================================

def get_lpips_model():
    if not LPIPS_AVAILABLE:
        return None
    loss_fn = lpips.LPIPS(net='alex', verbose=False)
    loss_fn.eval()
    return loss_fn


def compute_lpips(lpips_model, img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes LPIPS between two 2D grayscale slices.
    Input: numpy arrays in [0,1]
    LPIPS expects (B, 3, H, W) in [-1, 1]
    """
    if lpips_model is None:
        return float('nan')

    # Convert to 3-channel tensor in [-1, 1]
    def to_tensor(img):
        img_t = torch.from_numpy(img).float()
        img_t = img_t.unsqueeze(0).unsqueeze(0)  
        img_t = img_t.repeat(1, 3, 1, 1)          
        img_t = img_t * 2.0 - 1.0                 
        return img_t

    with torch.no_grad():
        t1 = to_tensor(img1)
        t2 = to_tensor(img2)
        score = lpips_model(t1, t2)

    return float(score.item())


# ============================================================
# VIF 
# ============================================================

def compute_vif(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes VIF between two 2D grayscale slices.
    Input: numpy arrays in [0,1]
    sewar expects uint8 images
    """
    if not VIF_AVAILABLE:
        return float('nan')

    try:
        img1_u8 = (img1 * 255).astype(np.uint8)
        img2_u8 = (img2 * 255).astype(np.uint8)
        score   = vifp(img1_u8, img2_u8)
        return float(score)
    except Exception:
        return float('nan')


# ============================================================
# COMPUTE ALL METRICS PER VOLUME
# ============================================================

def compute_all_metrics(
    clean_vol:   np.ndarray,
    pred_vol:    np.ndarray,
    lpips_model,
    skip_empty_threshold: float = 1e-6
) -> dict:
    """
    Computes PSNR, SSIM, MAE, RMSE, VIF, LPIPS per volume.
    PSNR, SSIM, VIF, LPIPS averaged per axial slice.
    MAE, RMSE on full flattened volume.
    """
    clean_norm, pred_norm = normalize_pair(clean_vol, pred_vol)

    psnr_list  = []
    ssim_list  = []
    vif_list   = []
    lpips_list = []

    for z in range(clean_norm.shape[0]):
        c = clean_norm[z]
        p = pred_norm[z]

        #  Exclude slices with negligible signal
        if c.max() < skip_empty_threshold:
            continue

        # PSNR
        psnr_list.append(
            peak_signal_noise_ratio(c, p, data_range=1.0)
        )

        # SSIM
        ssim_list.append(
            structural_similarity(
                c, p,
                data_range=1.0,
                win_size=7,
                gaussian_weights=True
            )
        )

        # VIF
        vif_list.append(compute_vif(c, p))

        # LPIPS
        if z % 5 == 0:
            lpips_list.append(compute_lpips(lpips_model, c, p))

    mae  = mean_absolute_error(
        clean_norm.flatten(), pred_norm.flatten()
    )
    rmse = np.sqrt(mean_squared_error(
        clean_norm.flatten(), pred_norm.flatten()
    ))

    return {
        "PSNR":  float(np.nanmean(psnr_list))  if psnr_list  else float('nan'),
        "SSIM":  float(np.nanmean(ssim_list))  if ssim_list  else float('nan'),
        "MAE":   float(mae),
        "RMSE":  float(rmse),
        "VIF":   float(np.nanmean(vif_list))   if vif_list   else float('nan'),
        "LPIPS": float(np.nanmean(lpips_list)) if lpips_list else float('nan'),
    }


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model_name:   str,
    clean_folder: str = "clean_test",
    lpips_model   = None
):
    pred_dir  = CT_ROOT / model_name
    clean_dir = CT_ROOT / clean_folder

    if not pred_dir.exists():
        print(f"Predicted folder not found: {pred_dir}")
        return None

    pred_files = sorted(pred_dir.glob("*.npy"))
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Found     : {len(pred_files)} predicted volumes")
    print(f"{'='*60}")

    results = []

    for pred_path in tqdm(pred_files, desc=model_name):

        patient_id = pred_path.stem
        clean_path = find_volume(clean_folder, patient_id)

        if clean_path is None:
            print(f"  Skipping {patient_id}: clean not found")
            continue

        clean_vol = np.load(clean_path).astype(np.float32)
        pred_vol  = np.load(pred_path).astype(np.float32)

        if clean_vol.shape != pred_vol.shape:
            print(f"  Skipping {patient_id}: shape mismatch "
                  f"{clean_vol.shape} vs {pred_vol.shape}")
            continue

        metrics = compute_all_metrics(clean_vol, pred_vol, lpips_model)

        print(
            f"  {patient_id:30s} | "
            f"PSNR: {metrics['PSNR']:6.2f} | "
            f"SSIM: {metrics['SSIM']:.4f} | "
            f"VIF: {metrics['VIF']:.4f} | "
            f"LPIPS: {metrics['LPIPS']:.4f}"
        )

        results.append({"patient": patient_id, **metrics})

    if not results:
        print("No results.")
        return None

    df = pd.DataFrame(results)

    # Save CSV
    csv_path = RESULTS_DIR / f"full_metrics_{model_name}.csv"
    df.to_csv(csv_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {model_name} ({len(df)} patients)")
    print(f"{'='*60}")
    print(f"PSNR  : {df['PSNR'].mean():.2f} ± {df['PSNR'].std():.2f} dB")
    print(f"SSIM  : {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
    print(f"MAE   : {df['MAE'].mean():.5f} ± {df['MAE'].std():.5f}")
    print(f"RMSE  : {df['RMSE'].mean():.5f} ± {df['RMSE'].std():.5f}")
    print(f"VIF   : {df['VIF'].mean():.4f} ± {df['VIF'].std():.4f}")
    print(f"LPIPS : {df['LPIPS'].mean():.4f} ± {df['LPIPS'].std():.4f}")
    print(f"\nSaved → {csv_path}")

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="unet")
    parser.add_argument("--clean-folder", default="clean")
    parser.add_argument("--no-lpips",     action="store_true",
                        help="Skip LPIPS computation (faster)")
    parser.add_argument("--no-vif",       action="store_true",
                        help="Skip VIF computation (faster)")
    args = parser.parse_args()

    # Load LPIPS model once and reuse across all patients
    lpips_model = None
    if LPIPS_AVAILABLE and not args.no_lpips:
        print("Loading LPIPS model (AlexNet, CPU)...")
        lpips_model = get_lpips_model()
        print("LPIPS model loaded.")

    if args.no_vif:
        global VIF_AVAILABLE
        VIF_AVAILABLE = False

    if args.model == "all":
        models    = ["unet", "mr_lkv", "replknet", "swinir", "restormer"]
        summaries = []

        for m in models:
            df = evaluate_model(
                m,
                clean_folder=args.clean_folder,
                lpips_model=lpips_model
            )
            if df is not None and not df.empty:
                summaries.append({
                    "model": m,
                    "PSNR":  df["PSNR"].mean(),
                    "SSIM":  df["SSIM"].mean(),
                    "MAE":   df["MAE"].mean(),
                    "RMSE":  df["RMSE"].mean(),
                    "VIF":   df["VIF"].mean(),
                    "LPIPS": df["LPIPS"].mean(),
                })

        if summaries:
            print(f"\n{'='*60}")
            print("ALL MODELS — FINAL COMPARISON TABLE")
            print(f"{'='*60}")
            summary_df = pd.DataFrame(summaries).set_index("model")
            print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
            out = RESULTS_DIR / "full_metrics_all_models.csv"
            summary_df.to_csv(out)
            print(f"\nSaved → {out}")
    else:
        evaluate_model(
            args.model,
            clean_folder=args.clean_folder,
            lpips_model=lpips_model
        )


if __name__ == "__main__":
    main()