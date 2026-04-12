#!/usr/bin/env python3

"""
Module: DICOM to Sinogram Conversion

This script performs:
- Loading CT volumes from DICOM series (CQ500 dataset)
- Selecting appropriate CT acquisition series 
- Normalizing volumes
- Forward projection using DiffCT (cone-beam geometry)
- Saving sinograms (.npy) and corresponding metadata
"""
import os
import math
import numpy as np
import pydicom
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
import json

ROOT = Path(__file__).resolve()

# Go up until we find config folder
for parent in ROOT.parents:
    if (parent / "config").exists():
        sys.path.insert(0, str(parent))
        break

from ExternalRepo.diffct.diffct.differentiable import ConeProjectorFunction
from config.config import DATASET_PATH, CLEAN_SINOGRAM_ROOT


# -----------------------------------------------------------
# 1. LOAD DICOM SERIES 
# -----------------------------------------------------------
def load_dicom_series(folder):
    
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".dcm")]

    # Safe sorting based on InstanceNumber, with fallback
    def sort_key(p):
        try:
            return pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber
        except:
            return 0

    files_sorted = sorted(files, key=sort_key)

    slices = []
    first_ds = None

    for f in files_sorted:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)

        if first_ds is None:
            first_ds = ds

        slices.append(arr)

    volume = np.stack(slices, axis=0)  # (Z, Y, X)
    return volume, first_ds


# -----------------------------------------------------------
# 3. VISUALIZE SINOGRAM
# -----------------------------------------------------------
def save_sino_preview(sino, out_png):

    num_views, U, V = sino.shape
    mid_u = U // 2
    mid_v = V // 2

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(sino[:, :, mid_v].T, cmap='gray', aspect='auto')
    axs[0].set_title("Central detector-row")

    axs[1].imshow(sino[:, mid_u, :].T, cmap='gray', aspect='auto')
    axs[1].set_title("Central detector-column")

    axs[2].imshow(sino[num_views//2], cmap='gray')
    axs[2].set_title("One projection")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()



# -----------------------------------------------------------
# SELECT THE CORRECT CT SERIES (only CT PLAIN THIN or CT PLAIN)
# -----------------------------------------------------------
def select_ct_series(patient_dir):

    VALID_NAMES = {
        "ct plain",
        "ct plain thin",
        "ct thin plain"
    }

    candidates = []

    for root, _, files in os.walk(patient_dir):

        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if len(dcm_files) < 20:
            continue

        folder = os.path.basename(root).lower()

      
        folder = folder.replace("_", " ").replace("-", " ")
        folder = " ".join(folder.split())  

        if folder in VALID_NAMES:
            candidates.append((root, len(dcm_files)))

    if len(candidates) == 0:
        return None

    # Pick the one with most slices
    best_series = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]

    return best_series

# -----------------------------------------------------------
# 5. MAIN SINOGRAM GENERATION
# -----------------------------------------------------------
def main():

    ROOT = Path(DATASET_PATH)
    OUT_DIR = Path(CLEAN_SINOGRAM_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Geometry (defined based on metadata from CQ500 and typical clinical CT settings)
    det_u = 800
    det_v = 800
    du = dv = 1.0
    sid = 530
    sdd = 1095
    num_views = 540
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    angles = torch.linspace(0, 2*math.pi, num_views, device=device)

    # Find patient folders
    patients = sorted([
        p for p in ROOT.rglob("*")
        if p.is_dir() and p.name.startswith("CQ500CT")
    ])[:100]

    print(f"Found {len(patients)} patients.")

    for p in tqdm(patients, desc="Processing"):

        
        series = select_ct_series(str(p))

        if series is None:
            print(f"[WARNING] Skipping {p.name} — no valid CT series found")
            continue

        print(f"Using series: {series}")

        # --------------------------
        # Load DICOM
        # --------------------------
        vol_np, first_ds = load_dicom_series(series)

        # Normalize volume to [0,1] for stable projection
        vol_np = vol_np.astype(np.float32)
        vol_np = (vol_np - vol_np.min()) / (vol_np.max() - vol_np.min())

      
        Nz, Ny, Nx = vol_np.shape

        # Extract voxel spacing safely
        px, py = map(float, first_ds.PixelSpacing)
        th = float(getattr(first_ds, "SliceThickness", min(px, py)))
        voxel_spacing = min(px, py, th)

        meta = {
            "Nz": Nz,
            "Ny": Ny,
            "Nx": Nx,
            "voxel_spacing": float(voxel_spacing)
        }

        vol_torch = torch.tensor(vol_np, dtype=torch.float32,
                                 device=device).contiguous()

        # --------------------------
        # Forward projection using ConeProjectorFunction (diffct)
        # --------------------------
        sino = ConeProjectorFunction.apply(
            vol_torch, angles,
            det_u, det_v, du, dv,
            sdd, sid, voxel_spacing
        )

        sino_np = sino.detach().cpu().numpy()

        # Save sinogram
        np.save(OUT_DIR / f"{p.name}.npy", sino_np)
        #Save metadata json alongside the sinogram for future reference
        with open(OUT_DIR / f"{p.name}.json", "w") as f:
            json.dump(meta, f, indent=4)

        # Save visualization
        save_sino_preview(sino_np, OUT_DIR / f"{p.name}.png")

    print("\nDONE — all clean sinograms generated & saved.")


# -----------------------------------------------------------
if __name__ == "__main__":
    main()
