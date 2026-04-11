#!/usr/bin/env python3
"""
Test-time sinogram_to_2D conversion.
Converts ALL slices without filtering — required for full 3D reconstruction.
No normalization applied — inference script handles normalization.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

from config.config import (
    TEST_CLEAN_SINOGRAM,
    ARTIFACT_SINOGRAM_3D_TEST,
    CLEAN_SINOGRAM_2D_TEST,
    ART_SINOGRAM_2D
)

# ============================================================
# SETTINGS
# ============================================================

STRIDE = 1  # keep all slices — do not skip any for testing

# ============================================================
# MAIN
# ============================================================

def main():

    clean_dir    = Path(TEST_CLEAN_SINOGRAM)
    artifact_dir = Path(ARTIFACT_SINOGRAM_3D_TEST)
    out_clean    = Path(CLEAN_SINOGRAM_2D_TEST)
    out_artifact = Path(ART_SINOGRAM_2D)

    out_clean.mkdir(parents=True, exist_ok=True)
    out_artifact.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(clean_dir.glob("*.npy"))
    print(f"Found {len(clean_files)} test 3D sinogram pairs.")

    total_slices   = 0
    saved_slices   = 0
    skipped_patients = 0

    for clean_path in tqdm(clean_files, desc="Converting test sinograms to 2D"):

        artifact_path = artifact_dir / clean_path.name

        # Check artifact exists
        if not artifact_path.exists():
            print(f"Skipping {clean_path.name}: missing artifact file")
            skipped_patients += 1
            continue

        # Load
        clean_3d    = np.load(clean_path).astype(np.float32)
        artifact_3d = np.load(artifact_path).astype(np.float32)

        # Validate
        if clean_3d.ndim != 3:
            print(f"Skipping {clean_path.name}: not 3D")
            skipped_patients += 1
            continue

        if clean_3d.shape != artifact_3d.shape:
            print(f"Skipping {clean_path.name}: shape mismatch "
                  f"{clean_3d.shape} vs {artifact_3d.shape}")
            skipped_patients += 1
            continue

        V, U, Vdet = clean_3d.shape  # (540, 800, 800)
        base_name = clean_path.stem

        # Slice along detector_v axis → true 2D sinograms (540, 800)
        # NO filtering — ALL slices saved for full reconstruction
        for k in range(0, Vdet, STRIDE):

            clean_slice    = clean_3d[:, :, k]     # (540, 800)
            artifact_slice = artifact_3d[:, :, k]  # (540, 800)

            total_slices += 1

            # NO normalization — inference script handles this
            # NO filtering  — all slices needed for reconstruction

            slice_name = f"{base_name}_slice_{k:04d}.npy"

            np.save(out_clean    / slice_name, clean_slice)
            np.save(out_artifact / slice_name, artifact_slice)

            saved_slices += 1

    print(f"\nDONE")
    print(f"Total slices processed : {total_slices}")
    print(f"Slices saved           : {saved_slices}")
    print(f"Patients skipped       : {skipped_patients}")
    print(f"\nClean 2D    → {out_clean}")
    print(f"Artifact 2D → {out_artifact}")


if __name__ == "__main__":
    main()