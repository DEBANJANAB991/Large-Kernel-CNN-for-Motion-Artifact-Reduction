#!/usr/bin/env python3
"""
Generates MPR comparison figures (Clean | Artifact | MR-LKV) WITH ROI zoom.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RECONSTRUCTED_CT_VOLUME

CT_ROOT = Path(RECONSTRUCTED_CT_VOLUME)
OUT_DIR = Path(__file__).resolve().parent / "results" / "mpr_mrlkv_roi"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def find_volume(folder: str, patient_id: str):
    base_id = patient_id.split(" ")[0]
    for name in [
        f"{patient_id}.npy",
        f"{base_id}.npy",
        f"{patient_id} {patient_id}.npy",
        f"{base_id} {base_id}.npy",
    ]:
        p = CT_ROOT / folder / name
        if p.exists():
            return np.load(p).astype(np.float32)
    return None


def normalize(vol: np.ndarray, ref: np.ndarray = None) -> np.ndarray:
    if ref is None:
        ref = vol
    v_min = ref.min()
    v_max = ref.max()
    return np.clip((vol - v_min) / (v_max - v_min + 1e-8), 0, 1)


def pick_best_axial(volume: np.ndarray) -> int:
    nz = volume.shape[0]
    max_var = 0
    best_idx = nz // 2
    for i in range(nz):
        v = np.var(volume[i])
        if v > max_var:
            max_var = v
            best_idx = i
    return best_idx


def compute_artifact_severity(clean, artifact):
    c = normalize(clean)
    a = normalize(artifact, ref=clean)
    return float(np.abs(c - a).mean())


def compute_improvement(clean, artifact, pred):
    c = normalize(clean)
    a = normalize(artifact, ref=clean)
    p = normalize(pred, ref=clean)
    return float(np.abs(c - a).mean() - np.abs(c - p).mean())


# ============================================================
# ROI FUNCTIONS
# ============================================================

def get_roi_coords_3views(clean, artifact, ax_idx, cor_idx, sag_idx):

    clean_n = normalize(clean)
    art_n   = normalize(artifact, ref=clean)

    error = np.abs(clean_n - art_n)

    ax_err = error[ax_idx]
    ay, ax = np.unravel_index(np.argmax(ax_err), ax_err.shape)

    cor_err = error[:, cor_idx, :]
    cy, cx = np.unravel_index(np.argmax(cor_err), cor_err.shape)

    sag_err = error[:, :, sag_idx]
    sy, sx = np.unravel_index(np.argmax(sag_err), sag_err.shape)

    return (ay, ax), (cy, cx), (sy, sx)


def crop_roi(img, y, x, size=40):
    h, w = img.shape
    y1 = max(0, y - size // 2)
    y2 = min(h, y + size // 2)
    x1 = max(0, x - size // 2)
    x2 = min(w, x + size // 2)
    return img[y1:y2, x1:x2]


def add_roi_overlay(ax, img, y, x, size=40):
    import matplotlib.patches as patches

    rect = patches.Rectangle(
        (x - size//2, y - size//2),
        size, size,
        linewidth=1.5,
        edgecolor='orange',
        facecolor='none'
    )
    ax.add_patch(rect)

    roi = crop_roi(img, y, x, size)

    inset = ax.inset_axes([0.65, 0.05, 0.3, 0.3])
    inset.imshow(roi, cmap="gray", vmin=0, vmax=1)
    inset.set_xticks([])
    inset.set_yticks([])


# ============================================================
# MAIN FIGURE
# ============================================================

def generate_mpr_figure(patient_id: str, clean_folder: str = "clean"):

    columns = [
        ("Ground truth", clean_folder),
        ("Corrupted", "artifact"),
        ("MR-LKV", "mr_lkv"),
    ]

    volumes = {}

    for title, folder in columns:
        volumes[folder] = find_volume(folder, patient_id)

    clean_vol = volumes.get(clean_folder)
    artifact_vol = volumes.get("artifact")
    mrlkv_vol = volumes.get("mr_lkv")

    if clean_vol is None:
        return None

    severity = compute_artifact_severity(clean_vol, artifact_vol) if artifact_vol is not None else 0
    improvement = compute_improvement(clean_vol, artifact_vol, mrlkv_vol) if mrlkv_vol is not None else 0

    clean_n = normalize(clean_vol)
    Nz, Ny, Nx = clean_n.shape

    ax_idx  = pick_best_axial(clean_n)
    cor_idx = Ny // 2
    sag_idx = Nx // 2

    roi_coords = None
    if artifact_vol is not None:
        roi_coords = get_roi_coords_3views(clean_vol, artifact_vol, ax_idx, cor_idx, sag_idx)

    fig, axs = plt.subplots(3, len(columns), figsize=(4 * len(columns), 10))

    row_labels = ["Axial", "Coronal", "Sagittal"]

    for col_idx, (title, folder) in enumerate(columns):

        vol = volumes.get(folder)
        vol_n = normalize(vol, ref=clean_vol) if vol is not None else None

        axs[0, col_idx].set_title(title, fontsize=10)

        for row_idx in range(3):
            ax = axs[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=10)

            if vol_n is None:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                continue

            if row_idx == 0:
                s = vol_n[ax_idx]
                (y, x) = roi_coords[0] if roi_coords else (None, None)
            elif row_idx == 1:
                s = vol_n[:, cor_idx, :]
                (y, x) = roi_coords[1] if roi_coords else (None, None)
            else:
                s = vol_n[:, :, sag_idx]
                (y, x) = roi_coords[2] if roi_coords else (None, None)

            ax.imshow(s, cmap="gray", vmin=0, vmax=1)

            if roi_coords:
                add_roi_overlay(ax, s, y, x)

    plt.suptitle(
        f"{patient_id.split()[0]} | Severity: {severity:.4f} | Improvement: {improvement:+.4f}",
        fontsize=12
    )

    plt.tight_layout()

    out_path = OUT_DIR / f"{patient_id.split()[0]}_mrlkv.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "patient_id": patient_id,
        "severity": severity,
        "improvement": improvement
    }


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-folder", default="clean")
    args = parser.parse_args()

    clean_dir = CT_ROOT / args.clean_folder
    patients = sorted([p.stem for p in clean_dir.glob("*.npy")])

    print(f"Found {len(patients)} patients\n")

    for patient_id in tqdm(patients):
        try:
            generate_mpr_figure(patient_id, args.clean_folder)
        except Exception as e:
            print(f"Error: {patient_id} → {e}")


if __name__ == "__main__":
    main()