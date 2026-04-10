#!/bin/bash -l
#
# SLURM Job Script for Optuna Hyperparameter Optimization of FixMatch Model
#
# ===============================
# SLURM Directives
# ===============================  
#SBATCH --gres=gpu:rtx3080:1           # Request 1 NVIDIA a100 GPU
#SBATCH --partition=rtx3080       # Specify the GPU partition a100
#SBATCH --time=24:00:00                 # Maximum runtime of 24 hours
#SBATCH --export=NONE                   # Do not export current environment variables
#SBATCH --job-name=MRLKV # Job name
#SBATCH --output=results/logs2/MRLKV.out      # Standard output log file (%j expands to job ID)
#SBATCH --error=results/logs2/MRLKV.err       # Standard error log file (%j expands to job ID)

# ===============================
# Environment Configuration#  
# ===============================

 
# Set HTTP and HTTPS proxies if required (uncomment and modify if needed)
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Unset SLURM_EXPORT_ENV to prevent SLURM from exporting environment variables
unset SLURM_EXPORT_ENV
 
# Load the necessary modules
module load python/3.12-conda        # Load Python Anaconda module
 
# Activate the Conda environment
conda activate /home/woody/iwi5/iwi5293h/software/private/conda/envs/thesis-gpu 
 
# ===============================
# Navigate to Script Directory
# ===============================
#cd /home/hpc/iwi5/iwi5293h/Debanjana_Master_Thesis/scripts 
cd /home/hpc/iwi5/iwi5293h/Debanjana_Master_Thesis/src



 
# ===============================
# Execute the Python Training Script
# ===============================

#Preprocessing steps
#python3 common/preprocessing/add_motion_artifacts.py
#python3 common/preprocessing/dicom_to_sinogram.py
#python3 common/preprocessing/dicom_to_sinogram_test.py
#python3 common/preprocessing/add_motion_artifacts_test.py

#Image-domain specific commands
#python3 image_domain/preprocessing/train_image_reconstruction.py
#python3 image_domain/preprocessing/test_image_reconstruction.py
#python3 image_domain/training/train.py --model unet
#python3 image_domain/training/train.py --model restormer --batch-size 1 --patch-size 128
#python3 image_domain/training/train.py --model swinir --batch-size 2 --patch-size 128
#python3 image_domain/training/train.py --model replk --batch-size 1 --patch-size 128
#python3 image_domain/training/train.py --model mr_lkv --norm batch
#python3 image_domain/inference/run_inference.py --model unet
#python3 image_domain/inference/run_inference.py --model mr_lkv
#python3 image_domain/inference/run_inference.py --model replknet
#python3 image_domain/inference/run_inference.py --model swinir
#python3 image_domain/inference/run_inference.py --model restormer

#projection-domain specific commands
#python3 projection_domain/training/train.py --model unet --max-views-per-patient 100 --unet-base 32
#python3 projection_domain/training/train.py --model mr_lkv --norm batch --max-views-per-patient 100 
#python3 projection_domain/training/train.py --patch 128 --model swinir --batch-size 2 --max-views-per-patient 100 
#python3 projection_domain/training/train.py --model restormer  --batch-size 1 --max-views-per-patient 100  
#python3 projection_domain/training/train.py --model replk --max-views-per-patient 100
#python3 projection_domain/preprocessing/sinogram_to_2D.py
#python3 projection_domain/preprocessing/sinogram_to_2d_test.py
#python3 projection_domain/inference/run_inference.py --model unet
#python3 projection_domain/inference/run_inference.py --model mr_lkv
#python3 projection_domain/inference/run_inference.py --model replknet
#python3 projection_domain/inference/run_inference.py --model swinir
#python3 projection_domain/inference/run_inference.py --model restormer
#python3 projection_domain/preprocessing/merge_2D_to_3D.py
#python3 projection_domain/reconstruction/fdk_reconstruction.py restormer
#python3 projection_domain/reconstruction/fdk_reconstruction.py mr_lkv
#python3 projection_domain/reconstruction/fdk_reconstruction.py replknet
#python3 projection_domain/reconstruction/fdk_reconstruction.py swinir
#python3 projection_domain/reconstruction/fdk_reconstruction.py unet



#common evaluation command
#python3 common/evaluation/final_evaluation.py --model unet --clean-folder clean
#python3 common/evaluation/final_evaluation.py --model mr_lkv --clean-folder clean
#python3 common/evaluation/final_evaluation.py --model replknet --clean-folder clean
#python3 common/evaluation/final_evaluation.py --model swinir --clean-folder clean
#python3 common/evaluation/final_evaluation.py --model restormer --clean-folder clean


#visualisation commands
#python3 visualisation/image_domain/visualise_ct.py
#python3 visualisation/image_domain/visualise_ct_mrlkv.py
#python3 visualisation/image_domain/metrics.py
#python3 visualisation/projection_domain/visualise_ct.py --clean-folder clean
#python3 visualisation/projection_domain/visualise_ct_mrlkv.py
#python3 visualisation/projection_domain/metrics.py





