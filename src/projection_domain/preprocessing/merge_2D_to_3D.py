#!/usr/bin/env python3
"""
Assembles 2D predicted sinogram slices back into 3D sinograms.

Filename format: "CQ500CT19 CQ500CT19_slice_0064.npy"
Slice axis: axis=2 (detector_v) — each slice shape (540, 800)
Total slices per patient: 800 (k = 0000 to 0799)
Output 3D shape: (540, 800, 800)
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import json
import re

ROOT = Path(__file__).resolve()

# Go up until we find config folder
for parent in ROOT.parents:
    if (parent / "config").exists():
        sys.path.insert(0, str(parent))
        break
from config import (
    PREDICTED_SINOGRAM_2D_TEST_v2,
    MERGED_SINOGRAM_3D_TEST_v2
)

ROOT_IN_DIR  = Path(PREDICTED_SINOGRAM_2D_TEST_v2)
ROOT_OUT_DIR = Path(MERGED_SINOGRAM_3D_TEST_v2)
ROOT_OUT_DIR.mkdir(parents=True, exist_ok=True)


SLICE_PATTERN = re.compile(r"^(CQ500CT\d+)\s+\1_slice_(\d{4})\.npy$")

# Expected dimensions 
NUM_VIEWS = 540   
DET_U     = 800  
DET_V     = 800   


# ============================================================
# PROCESS ONE MODEL
# ============================================================

def process_model(model_name):

    model_in_dir = ROOT_IN_DIR / model_name

    if not model_in_dir.exists():
        print(f"Model folder not found: {model_in_dir}")
        return

    print(f"\nProcessing model: {model_name}")

    out_dir = ROOT_OUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Group slices by patient
    # -------------------------
    groups = defaultdict(list)

    for f in sorted(model_in_dir.glob("*.npy")):
        m = SLICE_PATTERN.match(f.name)
        if m:
            patient_id = m.group(1)      # "CQ500CT19"
            k          = int(m.group(2)) # 64
            groups[patient_id].append((k, f))
        else:
            print(f"Skipped (name mismatch): {f.name}")

    print(f"Found {len(groups)} patients for model: {model_name}")

    if not groups:
        print(f"No valid files found. Check filename pattern matches: "
              f"'CQ500CT19 CQ500CT19_slice_0064.npy'")
        return

    # -------------------------
    # Assemble 3D sinogram per patient
    # -------------------------
    for patient_id, slices in groups.items():

        slices = sorted(slices, key=lambda x: x[0])  # sort by k

        # Validate slice count
        if len(slices) != DET_V:
            print(f"Warning: {patient_id} has {len(slices)} slices, "
                  f"expected {DET_V} — missing slices will be zero-filled")

        # Initialise empty 3D sinogram
        sino_3d = np.zeros((NUM_VIEWS, DET_U, DET_V), dtype=np.float32)
        meta    = None

        for k, f in slices:

            # Validate k is in range
            if k >= DET_V:
                print(f"Warning: slice index {k} out of range for {f.name}")
                continue

            slice_data = np.load(f).astype(np.float32)  # expected (540, 800)

            # Validate slice shape
            if slice_data.shape != (NUM_VIEWS, DET_U):
                print(f"Unexpected shape {slice_data.shape} for {f.name} "
                      f"— expected ({NUM_VIEWS}, {DET_U}), skipping")
                continue

            # Place back at correct detector_v position
            sino_3d[:, :, k] = slice_data

            # Load normalization metadata — saved by run_inference.py
            json_path = f.with_suffix(".json")
            if meta is None and json_path.exists():
                with open(json_path) as fp:
                    meta = json.load(fp)

        # Save assembled 3D sinogram
        out_path = out_dir / f"{patient_id}.npy"
        np.save(out_path, sino_3d)
        print(f"{model_name} | {patient_id}: shape {sino_3d.shape} saved")

        # Save metadata
        if meta is not None:
            with open(out_dir / f"{patient_id}.json", "w") as fp:
                json.dump(meta, fp)
        else:
            print(f"Warning: no metadata (.json) found for {patient_id}")

    print(f"\nDONE — {model_name} assembled 3D sinograms → {out_dir}")


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:
        print("Usage: python 2D_to_3D.py <model_name | all>")
        sys.exit(1)

    arg = sys.argv[1]

    if arg.lower() == "all":
        model_dirs = sorted([
            d.name for d in ROOT_IN_DIR.iterdir() if d.is_dir()
        ])
        print(f"Running for ALL models: {model_dirs}")
        for model_name in model_dirs:
            process_model(model_name)
    else:
        process_model(arg)

    print("\nDONE — all models assembled")


if __name__ == "__main__":
    main()