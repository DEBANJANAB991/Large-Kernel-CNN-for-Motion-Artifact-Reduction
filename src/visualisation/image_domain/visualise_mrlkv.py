#!/usr/bin/env python3
"""
MPR comparison with:
✔ 3 models (Clean, Artifact, MR-LKV)
✔ ROI detection
✔ Best slice selection
✔ Fair normalization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RECONSTRUCTED_CT_VOLUME

CT_ROOT = Path(RECONSTRUCTED_CT_VOLUME)

OUT_DIR = Path(__file__).resolve().parent / "mpr_3models_roi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODELS 
# ============================================================
COLUMNS = [
    ("Clean", "clean"),
    ("Artifact", "artifact"),
    ("MR-LKV", "mr_lkv"),
]

# ============================================================
# HELPERS
# ============================================================
def load_volume(folder, patient_id):
    p = CT_ROOT / folder / f"{patient_id}.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    return None


def normalize_volume(vol, ref):
    vmin, vmax = ref.min(), ref.max()
    return np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)


def percentile_window(vol, ref=None):
    if ref is None:
        ref = vol

    vmin = np.percentile(ref, 1)
    vmax = np.percentile(ref, 99)

    return np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)


def pick_best_axial(volume):
    vars = [np.var(volume[i]) for i in range(volume.shape[0])]
    return int(np.argmax(vars))


# ============================================================
# ROI
# ============================================================
def get_roi(clean, artifact, ax_idx, cor_idx, sag_idx):
    # Identify regions with maximum discrepancy between clean and corrupted volumes
    c = normalize_volume(clean, clean)
    a = normalize_volume(artifact, clean)

    err = np.abs(c - a)

    # Axial
    ax_err = err[ax_idx]
    ay, ax = np.unravel_index(np.argmax(ax_err), ax_err.shape)

    # Coronal
    cor_err = err[:, cor_idx, :]
    cy, cx = np.unravel_index(np.argmax(cor_err), cor_err.shape)

    # Sagittal
    sag_err = err[:, :, sag_idx]
    sy, sx = np.unravel_index(np.argmax(sag_err), sag_err.shape)

    return (ay, ax), (cy, cx), (sy, sx)


def crop(img, y, x, size=40):
    h, w = img.shape
    y1, y2 = max(0, y - size//2), min(h, y + size//2)
    x1, x2 = max(0, x - size//2), min(w, x + size//2)
    return img[y1:y2, x1:x2]


def add_roi(ax, img, y, x, size=40):
    import matplotlib.patches as patches

    rect = patches.Rectangle(
        (x - size//2, y - size//2),
        size, size,
        edgecolor='orange',
        facecolor='none',
        linewidth=1.5
    )
    ax.add_patch(rect)

    roi = crop(img, y, x, size)

    inset = ax.inset_axes([0.65, 0.05, 0.3, 0.3])
    inset.imshow(roi, cmap="gray", vmin=0, vmax=1)
    inset.set_xticks([])
    inset.set_yticks([])


# ============================================================
# MAIN FUNCTION
# ============================================================
def generate(patient_id):

    volumes = {}
    clean_vol = None

    # Load all
    for name, folder in COLUMNS:
        vol = load_volume(folder, patient_id)
        volumes[folder] = vol

        if folder == "clean" and vol is not None:
            clean_vol = vol

    if clean_vol is None:
        return

    artifact_vol = volumes["artifact"]

    clean_n = normalize_volume(clean_vol, clean_vol)

    Nz, Ny, Nx = clean_vol.shape

    ax_idx  = pick_best_axial(clean_n)
    cor_idx = Ny // 2
    sag_idx = Nx // 2

    # ROI
    roi_ax, roi_cor, roi_sag = get_roi(
        clean_vol, artifact_vol,
        ax_idx, cor_idx, sag_idx
    )

    # ========================================================
    # PLOT
    # ========================================================
    fig, axs = plt.subplots(3, len(COLUMNS), figsize=(4 * len(COLUMNS), 10))

    row_labels = ["Axial", "Coronal", "Sagittal"]

    for col_idx, (title, folder) in enumerate(COLUMNS):

        vol = volumes[folder]

        if vol is None:
            continue

        vol_n = percentile_window(vol)

        axs[0, col_idx].set_title(title, fontsize=10)

        for row_idx in range(3):

            ax = axs[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=10)

            if row_idx == 0:
                img = vol_n[ax_idx]
                y, x = roi_ax

            elif row_idx == 1:
                img = vol_n[:, cor_idx, :]
                y, x = roi_cor

            else:
                img = vol_n[:, :, sag_idx]
                y, x = roi_sag

            ax.imshow(img, cmap="gray", vmin=0, vmax=1)

            add_roi(ax, img, y, x)

    plt.suptitle(f"{patient_id}", fontsize=12)

    plt.tight_layout()

    save_path = OUT_DIR / f"{patient_id}_mpr.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# MAIN LOOP
# ============================================================
def main():

    clean_dir = CT_ROOT / "clean"
    patients = sorted([p.stem for p in clean_dir.glob("*.npy")])

    print(f"Found {len(patients)} patients\n")

    for p in tqdm(patients):
        try:
            generate(p)
        except Exception as e:
            print(f"Error: {p} → {e}")


if __name__ == "__main__":
    main()