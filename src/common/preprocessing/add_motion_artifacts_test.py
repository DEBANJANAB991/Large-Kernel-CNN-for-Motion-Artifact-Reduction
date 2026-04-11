#!/usr/bin/env python3
"""
Module: Motion Artifact Simulation in Projection Domain (3D Sinograms) for Testing

Key features:
- View-dependent motion events
- Combination of rotational and translational disturbances
- Persistent motion across projection angles

"""
import numpy as np
import random
from scipy.ndimage import rotate, map_coordinates
from pathlib import Path
from tqdm import tqdm
import sys



from config.config import TEST_CLEAN_SINOGRAM, ARTIFACT_SINOGRAM_3D_TEST

def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# RANDOM MOTION EVENTS (REALISTIC)
# ============================================================

def generate_motion_schedule(num_views):
    """
    Generates random motion events for a sinogram.

    Motion model:
    - A small number of motion events (1–2) are sampled
    - Each event defines:
        - Rotation angle (degrees)
        - Translation shift (pixels)
        - Starting projection index

    Returns:
        rot_events, rot_degs, trans_events, trans_px
    """
    # number of motion events (1–2)
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

def simulate_motion_corruption(sino_3d):
    """
    Applies motion corruption to a 3D sinogram.

    Steps:
    1. Sample motion schedule
    2. Apply rotation (simulates angular head movement)
    3. Apply translation (simulates lateral displacement)

    Motion persists across views after initiation,
    mimicking real CT acquisition behavior.
    """
    
    sino_out = sino_3d.copy()
    num_views, det_u, det_v = sino_3d.shape

    # sample random motion events
    rot_events, rot_degs, trans_events, trans_px = generate_motion_schedule(num_views)

    # -------------------------
    # ROTATION
    # -------------------------
    for event_idx, rotation_angle in zip(rot_events, rot_degs):
        if event_idx >= num_views:
            continue

        for v in range(event_idx, num_views):
            sino_out[v] = rotate(
                sino_out[v],
                angle=rotation_angle,
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

    for event_idx, translation_shift in zip(trans_events, trans_px):
        if event_idx >= num_views:
            continue

        for v in range(event_idx, num_views):
            coords = np.vstack([
                (grid_u - translation_shift).ravel(),
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
    set_random_seed(42)
    CLEAN_DIR    = Path(TEST_CLEAN_SINOGRAM)
    ARTIFACT_DIR = Path(ARTIFACT_SINOGRAM_3D_TEST)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(CLEAN_DIR.glob("*.npy"))
    print(f"Found {len(clean_files)} clean 3D sinograms.")

    num_processed = 0
    num_skipped   = 0

    for clean_path in tqdm(clean_files, desc="Generating artifact sinograms"):

        out_path = ARTIFACT_DIR / clean_path.name

        if out_path.exists():
            num_skipped += 1
            continue

        sino_clean = np.load(clean_path)

        if sino_clean.ndim != 3:
            print(f"[WARNING] Skipping {clean_path.name}: not 3D")
            num_skipped += 1
            continue

        # APPLY REALISTIC MOTION
        sino_artifact = simulate_motion_corruption(sino_clean)

        np.save(out_path, sino_artifact)
        num_processed += 1

    print(f"\nDONE — processed: {num_processed}, skipped: {num_skipped}")


if __name__ == "__main__":
    main()