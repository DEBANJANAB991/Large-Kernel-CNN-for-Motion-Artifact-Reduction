# config.py 

from pathlib import Path

# ------------------------------------------------------------------
# Mandatory paths 
# ------------------------------------------------------------------
DATASET_PATH   = Path("/home/vault/iwi5/iwi5293h/CT-Motion-Artifact-Reduction_bkp/data/dicom_raw")
CLEAN_SINOGRAM_ROOT=Path("/home/woody/iwi5/iwi5293h/clean_sinograms") #clean sinogram in 3D generated from dicom files
TEST_CLEAN_SINOGRAM = Path("/home/woody/iwi5/iwi5293h/clean_sinograms_test") #test clean sinogram in 3D generated from dicom files
CLEAN_SINOGRAM_2D=Path("/home/woody/iwi5/iwi5293h/clean_sinograms_2d") #clean sinogram in 2D generated from CLEAN_SINOGRAM_ROOT
CLEAN_SINOGRAM_2D_TEST=Path("/home/woody/iwi5/iwi5293h/clean_sinograms_2d_test") #test clean sinogram in 2D generated from TEST_CLEAN_SINOGRAM
ARTIFACT_ROOT_2D=Path("/home/woody/iwi5/iwi5293h/art_sinograms_2d_v2")#artifact in 2d generated from CLEAN_SINOGRAM_2D by adding motion artifacts
ARTIFACT_SINOGRAM_3D_TEST=Path("/home/woody/iwi5/iwi5293h/art_sinograms_3D_test")
PREDICTED_SINOGRAM_2D_TEST_v2=Path("/home/woody/iwi5/iwi5293h/predicted_sinograms_2d_test_v2") # test predicted sinogram in 2D generated after testing using MR_LKV on ARTIFACT_SINOGRAM_2D_TEST_v2
MERGED_SINOGRAM_3D_TEST_v2=Path("/home/woody/iwi5/iwi5293h/merged_sinograms_3d_test_v2") # test merged sinogram in 3D generated after merging predicted 2D sinograms from PREDICTED_SINOGRAM_2D_TEST_v2
MERGED_ARTIFACT_3D_TEST_v2=Path("/home/woody/iwi5/iwi5293h/merged_artifact_3d_test_v2") # test merged artifact sinogram in 3D generated after merging artifact 2D sinograms from ARTIFACT_SINOGRAM_2D_TEST_v2
RECONSTRUCTED_CT_VOLUME  =Path("/home/woody/iwi5/iwi5293h/reconstructed_ct_volumes") #reconstructed ct volume generated
ART_SINOGRAM_2D=Path("/home/woody/iwi5/iwi5293h/art_sinograms_2d") # new artifact in 2D generated from CLEAN_SINOGRAM_2D_TEST by adding motion artifacts using a different random seed
ARTIFACT_SINOGRAM_ROOT=Path("/home/woody/iwi5/iwi5293h/artifact_sinograms") #artifact in 3D generated from CLEAN_SINOGRAM_ROOT by adding motion artifacts
RECON_CLEAN_ROOT = Path("/home/woody/iwi5/iwi5293h/recon_clean") #reconstructed ct volume generated from clean sinograms
RECON_ARTIFACT_ROOT = Path("/home/woody/iwi5/iwi5293h/recon_artifact") #reconstructed ct volume generated from artifact sinograms

# Where to save model checkpoints (scratch)

CKPT_DIR=Path("/home/vault/iwi5/iwi5293h/models")


RECON_ROOT     = DATASET_PATH.parent / "recon"    

# ------------------------------------------------------------------
# Fan-beam geometry (pixel units, square detector)
# ------------------------------------------------------------------
N_VIEWS         = 360
N_DET           = 512
STEP_SIZE       = 0.5
MAX_SAMPLES=20000
MAX_SAMPLES_TEST=100
SID = 560.0           # Source → Isocenter (mm)
SDD = 1100.0          # Source → Detector (mm)
DET_SPACING = 1.2     # Detector spacing (mm)
DET_COUNT = 736       # Number of detector elements
VOXEL_SPACING = 1.0   # MUST remain 1.0 for diffct
# Training hyperparameters
# ------------------------------------------------------------------
# Total number of epochs
EPOCHS = 100
MU_WATER = 0.019 

# Batch size per GPU
BATCH_SIZE=8


# Learning rate for optimizer
LR = 1e-4

# How often (in epochs) to save a checkpoint
SAVE_INTERVAL = 10