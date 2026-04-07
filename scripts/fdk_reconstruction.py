#!/usr/bin/env python3
"""
FDK Reconstruction using ConeBackprojectorFunction.
Uses Hann-windowed ramp filter to reduce ring artifacts.
Per-patient metadata loaded for correct Nz and voxel_spacing.
Overview shows only non-empty slices.
"""

import math
import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from diffct.differentiable import ConeBackprojectorFunction
from config import (
    MERGED_SINOGRAM_3D_TEST_v2,
    RECONSTRUCTED_CT_VOLUME,
    TEST_CLEAN_SINOGRAM,
    ARTIFACT_SINOGRAM_3D_TEST
)

# ============================================================
# GEOMETRY — must match dicom_to_sinogram.py exactly
# ============================================================

NUM_VIEWS = 540
DET_U     = 800
DET_V     = 800
DU        = 1.0
DV        = 1.0
SID       = 530.0
SDD       = 1095.0
MAX_FILES = 30
USE_GPU   = True

# Fallback defaults if no metadata found
DEFAULT_NZ         = 260
DEFAULT_NY         = 512
DEFAULT_NX         = 512
DEFAULT_VOXEL_SIZE = 0.4688

ROOT_SINO_DIR = Path(MERGED_SINOGRAM_3D_TEST_v2)
CT_OUT_ROOT   = Path(RECONSTRUCTED_CT_VOLUME)
SCRIPT_DIR    = Path(__file__).resolve().parent
PNG_OUT_ROOT  = SCRIPT_DIR / "reconstructed_directory"

CT_OUT_ROOT.mkdir(parents=True,  exist_ok=True)
PNG_OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD PER-PATIENT METADATA
# ============================================================

def load_patient_meta(patient_id: str) -> dict:
    """
    Loads per-patient metadata saved by generate_metadata.py.
    Handles all naming formats.
    Falls back to defaults if not found.
    """
    clean_dir = Path(TEST_CLEAN_SINOGRAM)

    # Extract base patient ID — handles "CQ500CT113 CQ500CT113" format
    base_id = patient_id.split(" ")[0]

    # Try all possible naming formats
    candidates = [
        f"{patient_id}.json",
        f"{patient_id} {patient_id}.json",
        f"{base_id}.json",
        f"{base_id} {base_id}.json",
    ]

    for name in candidates:
        json_path = clean_dir / name
        if json_path.exists():
            with open(json_path) as fp:
                meta = json.load(fp)
            print(f"  Loaded: {name} — Nz={meta['Nz']}, voxel={meta['voxel_spacing']:.4f}")
            return meta

    print(f"  Warning: no metadata for {patient_id} — using defaults")
    return {
        "Nz":            DEFAULT_NZ,
        "Ny":            DEFAULT_NY,
        "Nx":            DEFAULT_NX,
        "voxel_spacing": DEFAULT_VOXEL_SIZE
    }


# ============================================================
# FIND SINOGRAM — handles space in filename
# ============================================================

def find_sinogram(sino_dir: Path, patient_id: str):
    exact = sino_dir / f"{patient_id}.npy"
    if exact.exists():
        return exact

    spaced = sino_dir / f"{patient_id} {patient_id}.npy"
    if spaced.exists():
        return spaced

    matches = list(sino_dir.glob(f"{patient_id}*.npy"))
    if matches:
        return matches[0]

    return None


# ============================================================
# HANN-WINDOWED RAMP FILTER — reduces ring artifacts
# ============================================================

def ramp_filter_hann(sino: torch.Tensor) -> torch.Tensor:
    """
    Hann-windowed ramp filter.
    Reduces ring/streak artifacts compared to pure ramp.
    sino: (num_views, det_u, det_v)
    """
    device = sino.device
    _, det_u, _ = sino.shape

    freqs = torch.fft.fftfreq(det_u, device=device)
    omega = 2.0 * math.pi * freqs
    ramp  = torch.abs(omega).reshape(1, det_u, 1)

    hann = 0.5 * (
        1.0 + torch.cos(
            math.pi * freqs / (freqs.abs().max() + 1e-12)
        )
    ).reshape(1, det_u, 1)

    filt     = ramp * hann
    sino_fft = torch.fft.fft(sino, dim=1)
    return torch.real(torch.fft.ifft(sino_fft * filt, dim=1))


# ============================================================
# RECONSTRUCT
# ============================================================

def reconstruct_one(
    sino_np:    np.ndarray,
    device:     torch.device,
    Nz:         int,
    Ny:         int,
    Nx:         int,
    voxel_size: float
) -> np.ndarray:
    """
    sino_np: (540, 800, 800)
    returns: (Nz, Ny, Nx)
    """
    angles = torch.linspace(
        0.0, 2.0 * math.pi, NUM_VIEWS, device=device
    ).contiguous()

    sino = torch.tensor(
        sino_np.astype(np.float32), device=device
    ).contiguous()

    # Step 1 — cosine weighting
    u = (torch.arange(DET_U, device=device) - (DET_U - 1) / 2.0) * DU
    v = (torch.arange(DET_V, device=device) - (DET_V - 1) / 2.0) * DV
    u = u.view(1, DET_U, 1)
    v = v.view(1, 1, DET_V)
    weights = SDD / torch.sqrt(SDD**2 + u**2 + v**2)
    sino_w  = sino * weights

    # Step 2 — Hann-windowed ramp filter
    sino_filt = ramp_filter_hann(sino_w).contiguous()

    # Step 3 — backprojection
    reco = ConeBackprojectorFunction.apply(
        sino_filt,
        angles,
        Nz,
        Ny,
        Nx,
        float(DU),
        float(DV),
        float(SDD),
        float(SID),
        float(voxel_size)
    )

    # Step 4 — relu + scale
    reco = torch.relu(reco) * (math.pi / NUM_VIEWS)
    return reco.detach().cpu().numpy().astype(np.float32)


# ============================================================
# NORMALIZE SLICE
# ============================================================

def normalize_slice(s: np.ndarray) -> np.ndarray:
    """
    Tighter percentile normalization for soft tissue contrast.
    Skips empty slices gracefully.
    """
    if s.max() < 1e-6:
        return np.zeros_like(s, dtype=np.float32)

    nonzero = s[s > 0]
    if len(nonzero) > 0:
        vmin = np.percentile(nonzero, 5)
        vmax = np.percentile(nonzero, 99)
    else:
        vmin, vmax = s.min(), s.max()

    s = np.clip(s, vmin, vmax)
    return (s - vmin) / (vmax - vmin + 1e-8)


# ============================================================
# SAVE SLICES
# ============================================================

def save_slices(volume: np.ndarray, png_dir: Path, patient_id: str):
    """
    Saves every axial slice as PNG.
    Skips empty slices.
    """
    png_dir.mkdir(parents=True, exist_ok=True)

    for i in range(volume.shape[0]):
        s = volume[i].astype(np.float32)

        if s.max() < 1e-6:
            continue

        s = normalize_slice(s)
        s = (s * 255).astype(np.uint8)
        Image.fromarray(s).save(png_dir / f"slice_{i:03d}.png")


# ============================================================
# SAVE OVERVIEW — shows only non-empty slices
# ============================================================

def save_overview(volume: np.ndarray, png_dir: Path, patient_id: str):
    """
    Saves 3-panel overview.
    Picks top/mid/bottom only from slices with actual content.
    """
    png_dir.mkdir(parents=True, exist_ok=True)

    nz        = volume.shape[0]
    slice_vars = [np.var(volume[i]) for i in range(nz)]
    max_var    = max(slice_vars) if max(slice_vars) > 0 else 1.0

    # Only consider slices with at least 5% of max variance
    content_indices = [
        i for i, v in enumerate(slice_vars)
        if v > 0.05 * max_var
    ]

    if not content_indices:
        content_indices = list(range(nz))

    n = len(content_indices)
    top    = content_indices[n // 4]
    mid    = content_indices[n // 2]
    bottom = content_indices[(3 * n) // 4]

    slices = {"top": top, "mid": mid, "bottom": bottom}

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (name, idx) in zip(axs, slices.items()):
        s = normalize_slice(volume[idx].astype(np.float32))
        ax.imshow(s, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Slice {idx} ({name})")
        ax.axis("off")

    plt.suptitle(patient_id, fontsize=14)
    plt.tight_layout()
    plt.savefig(
        png_dir / f"{patient_id}_overview.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close()


# ============================================================
# PROCESS ONE FOLDER
# ============================================================

def process_folder(
    sino_dir:    Path,
    ct_out_dir:  Path,
    png_out_dir: Path,
    label:       str,
    device:      torch.device
):
    if not sino_dir.exists():
        print(f"Folder not found: {sino_dir}")
        return

    ct_out_dir.mkdir(parents=True,  exist_ok=True)
    png_out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(sino_dir.glob("*.npy"))[:MAX_FILES]
    print(f"\nProcessing {label} — {len(files)} sinograms")
    print(f"  Volumes → {ct_out_dir}")
    print(f"  PNGs    → {png_out_dir}")

    for p in tqdm(files, desc=label):
        try:
            sino_np = np.load(p).astype(np.float32)

            if sino_np.ndim != 3:
                print(f"Skipping {p.name}: not 3D")
                continue

            # Load per-patient metadata
            meta       = load_patient_meta(p.stem)
            Nz         = int(meta["Nz"])
            Ny         = int(meta["Ny"])
            Nx         = int(meta["Nx"])
            voxel_size = float(meta["voxel_spacing"])

            # Clamp Nz to sensible range for head CT
            Nz = max(128, min(Nz, 400))

            # Reconstruct
            reco_np = reconstruct_one(
                sino_np, device,
                Nz=Nz, Ny=Ny, Nx=Nx,
                voxel_size=voxel_size
            )

            # Save volume
            np.save(ct_out_dir / f"{p.stem}.npy", reco_np)

            # Save PNGs
            patient_png = png_out_dir / p.stem
            save_slices(reco_np,   patient_png / "slices", p.stem)
            save_overview(reco_np, patient_png,             p.stem)

        except Exception as e:
            print(f"Failed: {p.name} — {e}")
            import traceback
            traceback.print_exc()

    print(f"Done — {label}")


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    )
    print(f"Using device: {device}")

    if len(sys.argv) < 2:
        print("Usage: python fdk_reconstruction.py <model_name | all | clean | artifact>")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg == "clean":
        process_folder(
            sino_dir    = Path(TEST_CLEAN_SINOGRAM),
            ct_out_dir  = CT_OUT_ROOT  / "clean",
            png_out_dir = PNG_OUT_ROOT / "clean",
            label       = "clean",
            device      = device
        )

    elif arg == "artifact":
        process_folder(
            sino_dir    = Path(ARTIFACT_SINOGRAM_3D_TEST),
            ct_out_dir  = CT_OUT_ROOT  / "artifact",
            png_out_dir = PNG_OUT_ROOT / "artifact",
            label       = "artifact",
            device      = device
        )

    elif arg == "all":
        process_folder(
            sino_dir    = Path(TEST_CLEAN_SINOGRAM),
            ct_out_dir  = CT_OUT_ROOT  / "clean",
            png_out_dir = PNG_OUT_ROOT / "clean",
            label       = "clean",
            device      = device
        )
        process_folder(
            sino_dir    = Path(ARTIFACT_SINOGRAM_3D_TEST),
            ct_out_dir  = CT_OUT_ROOT  / "artifact",
            png_out_dir = PNG_OUT_ROOT / "artifact",
            label       = "artifact",
            device      = device
        )
        model_dirs = sorted([
            d.name for d in ROOT_SINO_DIR.iterdir() if d.is_dir()
        ])
        print(f"Reconstructing predicted models: {model_dirs}")
        for model_name in model_dirs:
            process_folder(
                sino_dir    = ROOT_SINO_DIR / model_name,
                ct_out_dir  = CT_OUT_ROOT   / model_name,
                png_out_dir = PNG_OUT_ROOT  / model_name,
                label       = model_name,
                device      = device
            )

    else:
        process_folder(
            sino_dir    = ROOT_SINO_DIR / arg,
            ct_out_dir  = CT_OUT_ROOT   / arg,
            png_out_dir = PNG_OUT_ROOT  / arg,
            label       = arg,
            device      = device
        )

    print("\nDONE — all reconstructions and PNGs saved")


if __name__ == "__main__":
    main()