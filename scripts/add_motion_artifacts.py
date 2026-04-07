#!/usr/bin/env python3

import numpy as np
import random
from scipy.ndimage import rotate, map_coordinates
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to Python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from config import CLEAN_SINOGRAM_ROOT, ARTIFACT_SINOGRAM_ROOT


# ============================================================
# RANDOM MOTION EVENTS (REALISTIC)
# ============================================================

def sample_motion_events(num_views):

    # number of motion events (1–3)
    num_events = np.random.randint(1, 3)

    # avoid edges
    base_positions = np.random.randint(100, num_views - 100, size=num_events)

    # rotation
    rot_events = base_positions
    rot_degs   = np.random.uniform(-5, 5, size=num_events)

    # translation happens slightly after rotation
    trans_events = base_positions + np.random.randint(5, 20, size=num_events)
    trans_px     = np.random.uniform(-15, 15, size=num_events)

    return rot_events, rot_degs, trans_events, trans_px


# ============================================================
# APPLY REALISTIC MOTION (ROTATION + TRANSLATION)
# ============================================================

def apply_motion_to_3d_sinogram(sino_3d):

    sino_out = sino_3d.copy()
    num_views, det_u, det_v = sino_3d.shape

    # sample random motion events
    rot_events, rot_degs, trans_events, trans_px = sample_motion_events(num_views)

    # -------------------------
    # ROTATION
    # -------------------------
    for ev, deg in zip(rot_events, rot_degs):
        if ev >= num_views:
            continue

        for v in range(ev, num_views):
            sino_out[v] = rotate(
                sino_out[v],
                angle=deg,
                reshape=False,
                order=3,          # cubic interpolation
                mode="constant",  # more visible artifacts
                cval=0
            )

    # -------------------------
    # TRANSLATION (U-axis)
    # -------------------------
    grid_u, grid_v = np.meshgrid(
        np.arange(det_u),
        np.arange(det_v),
        indexing="ij"
    )

    for ev, shift in zip(trans_events, trans_px):
        if ev >= num_views:
            continue

        for v in range(ev, num_views):
            coords = np.vstack([
                (grid_u - shift).ravel(),
                grid_v.ravel()
            ])

            warped = map_coordinates(
                sino_out[v],
                coords,
                order=3,
                mode='constant',
                cval=0
            )

            sino_out[v] = warped.reshape(det_u, det_v)

    return sino_out.astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    CLEAN_DIR    = Path(CLEAN_SINOGRAM_ROOT)
    ARTIFACT_DIR = Path(ARTIFACT_SINOGRAM_ROOT)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(CLEAN_DIR.glob("*.npy"))
    print(f"Found {len(clean_files)} clean 3D sinograms.")

    processed = 0
    skipped   = 0

    for clean_path in tqdm(clean_files, desc="Generating artifact sinograms"):

        out_path = ARTIFACT_DIR / clean_path.name

        if out_path.exists():
            skipped += 1
            continue

        sino_clean = np.load(clean_path)

        if sino_clean.ndim != 3:
            print(f"Skipping {clean_path.name}: not 3D")
            skipped += 1
            continue

        # APPLY REALISTIC MOTION
        sino_artifact = apply_motion_to_3d_sinogram(sino_clean)

        np.save(out_path, sino_artifact)
        processed += 1

    print(f"\nDONE — processed: {processed}, skipped: {skipped}")


if __name__ == "__main__":
    main()