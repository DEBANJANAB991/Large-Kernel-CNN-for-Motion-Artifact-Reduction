#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys


from config.config import (
    CLEAN_SINOGRAM_ROOT,
    ARTIFACT_SINOGRAM_ROOT,
    CLEAN_SINOGRAM_2D,
    ARTIFACT_ROOT_2D
)

# ============================================================
# SETTINGS
# ============================================================

USE_FILTERING = True       # removes empty slices
MEAN_THRESHOLD = 0.01   
STRIDE = 1                 

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    clean_dir = Path(CLEAN_SINOGRAM_ROOT)
    artifact_dir = Path(ARTIFACT_SINOGRAM_ROOT)

    out_clean = Path(CLEAN_SINOGRAM_2D)
    out_artifact = Path(ARTIFACT_ROOT_2D)

    out_clean.mkdir(parents=True, exist_ok=True)
    out_artifact.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(clean_dir.glob("*.npy"))

    print(f"Found {len(clean_files)} 3D sinograms.")

    total_slices = 0
    saved_slices = 0

    for clean_path in tqdm(clean_files, desc="Converting to 2D"):

        artifact_path = artifact_dir / clean_path.name

        if not artifact_path.exists():
            print(f"Skipping {clean_path.name}: missing artifact file")
            continue

        # Load
        clean_3d = np.load(clean_path)
        artifact_3d = np.load(artifact_path)

        if clean_3d.shape != artifact_3d.shape:
            print(f"Shape mismatch: {clean_path.name}")
            continue

        V, U, Vdet = clean_3d.shape

        # Loop over detector_v
        for k in range(0, Vdet, STRIDE):

            clean_slice = clean_3d[:, :, k]
            artifact_slice = artifact_3d[:, :, k]

            total_slices += 1

            # ---------------------------------------------------
            # skip empty slices
            # ---------------------------------------------------
            if USE_FILTERING:
                if np.mean(clean_slice) < MEAN_THRESHOLD:
                    continue


            # ---------------------------------------------------
            # Save with matching names
            # ---------------------------------------------------
            base_name = clean_path.stem
            slice_name = f"{base_name}_slice_{k:04d}.npy"

            np.save(out_clean / slice_name, clean_slice)
            np.save(out_artifact / slice_name, artifact_slice)

            saved_slices += 1

    print("\nDONE ✅")
    print(f"Total slices processed: {total_slices}")
    print(f"Slices saved: {saved_slices}")


# ============================================================
if __name__ == "__main__":
    main()