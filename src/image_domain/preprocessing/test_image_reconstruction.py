#!/usr/bin/env python3
"""
Module: Cone-Beam CT Reconstruction using DiffCT

This script reconstructs 3D CT volumes from clean and motion-corrupted sinograms for testing. 

Pipeline:
1. Load 3D sinograms
2. Apply cone-beam geometric weighting
3. Apply ramp + Hann filtering in frequency domain
4. Perform backprojection using DiffCT
5. Normalize reconstructed volumes


"""
import math
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys
from ExternalRepo.diffct.diffct.differentiable import ConeBackprojectorFunction

# ================================
# IMPORT CONFIG PATHS
# ================================
from config.config import (
    TEST_CLEAN_SINOGRAM,
    ARTIFACT_SINOGRAM_3D_TEST,
    RECONSTRUCTED_CT_VOLUME
)

# ================================
# GEOMETRY (FIXED)
# ================================
NUM_VIEWS = 540
DET_U = 800
DET_V = 800
DU = 1.0
DV = 1.0
SID = 530.0
SDD = 1095.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# PATHS
# ================================
ROOT_RECON = Path(RECONSTRUCTED_CT_VOLUME)
CLEAN_SINO_DIR = Path(TEST_CLEAN_SINOGRAM)
ART_SINO_DIR   = Path(ARTIFACT_SINOGRAM_3D_TEST)

OUT_CLEAN = ROOT_RECON / "clean"
OUT_ART   = ROOT_RECON / "artifact"

OUT_CLEAN.mkdir(exist_ok=True, parents=True)
OUT_ART.mkdir(exist_ok=True, parents=True)

# ================================
# LOAD METADATA
# ================================
def load_metadata(patient_id):
    """
    Loads metadata from clean sinogram folder.
    Handles naming like:
    CQ500CT10 CQ500CT10.json
    """
    candidates = [
        CLEAN_SINO_DIR / f"{patient_id}.json",
        CLEAN_SINO_DIR / f"{patient_id} {patient_id}.json"
    ]

    for path in candidates:
        if path.exists():
            with open(path) as f:
                meta = json.load(f)

            return (
                int(meta["Nz"]),
                int(meta["Ny"]),
                int(meta["Nx"]),
                float(meta["voxel_spacing"])
            )

    # fallback (safe)
    print(f"Metadata not found for {patient_id}, using defaults")
    return 260, 512, 512, 0.4688


# ================================
# FDK FILTER (HANN + RAMP)
# ================================
def ramp_filter_hann(sino):
    freqs = torch.fft.fftfreq(DET_U, device=sino.device)
    omega = 2 * math.pi * freqs

    ramp = torch.abs(omega).view(1, DET_U, 1)

    hann = 0.5 * (1 + torch.cos(math.pi * freqs / (freqs.abs().max() + 1e-12)))
    hann = hann.view(1, DET_U, 1)

    filt = ramp * hann

    sino_fft = torch.fft.fft(sino, dim=1)
    return torch.real(torch.fft.ifft(sino_fft * filt, dim=1))


# ================================
# FDK RECONSTRUCTION
# ================================
def fdk_reconstruct(sino_np, Nz, Ny, Nx, voxel_size):

    sino = torch.tensor(sino_np, dtype=torch.float32, device=DEVICE).contiguous()

    angles = torch.linspace(0, 2 * math.pi, NUM_VIEWS, device=DEVICE).contiguous()

    # -------- Weighting --------
    u = (torch.arange(DET_U, device=DEVICE) - (DET_U - 1)/2) * DU
    v = (torch.arange(DET_V, device=DEVICE) - (DET_V - 1)/2) * DV

    u = u.view(1, DET_U, 1)
    v = v.view(1, 1, DET_V)

    weights = SDD / torch.sqrt(SDD**2 + u**2 + v**2)
    sino = (sino * weights).contiguous()

    # -------- Filtering --------
    sino = ramp_filter_hann(sino).contiguous()

    # -------- Backprojection --------
    reco = ConeBackprojectorFunction.apply(
        sino, angles,
        Nz, Ny, Nx,
        DU, DV,
        SDD, SID,
        voxel_size
    )

    # -------- Scaling --------
    reco = reco * (math.pi / NUM_VIEWS)

    return reco.detach().cpu().numpy().astype(np.float32)


# ================================
# NORMALIZATION
# ================================
def normalize_volume(vol):
    vmin = np.percentile(vol, 1)
    vmax = np.percentile(vol, 99)

    vol = np.clip(vol, vmin, vmax)
    vol = (vol - vmin) / (vmax - vmin + 1e-8)

    return vol


# ================================
# PROCESS FUNCTION
# ================================
def process_folder(sino_dir, out_dir):

    files = sorted(sino_dir.glob("*.npy"))

    print(f"\nProcessing {len(files)} files from {sino_dir}")

    for f in tqdm(files):

        sino = np.load(f)

        if sino.ndim != 3:
            print(f"Skipping {f.name}")
            continue

        patient_id = f.stem

        #LOAD METADATA
        Nz, Ny, Nx, voxel_size = load_metadata(patient_id)

        reco = fdk_reconstruct(sino, Nz, Ny, Nx, voxel_size)

        #SAME normalization for both domains
        reco = normalize_volume(reco)

        np.save(out_dir / f.name, reco)


# ================================
# MAIN
# ================================
if __name__ == "__main__":

    print("Reconstructing CLEAN sinograms...")
    process_folder(CLEAN_SINO_DIR, OUT_CLEAN)

    print("Reconstructing ARTIFACT sinograms...")
    process_folder(ART_SINO_DIR, OUT_ART)

    print("\nDONE — metadata-based reconstruction complete")