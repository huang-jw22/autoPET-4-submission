# [autoPET IV Challenge (Task I) Winning Solution] from Team BIRTH

🏆 Our method won **the 1st place** in the [MICCAI 2025 autoPET IV Challenge](https://autopet-iv.grand-challenge.org/autopet-iv/)! 🏆

This repository contains the code and models used for our winning solution to the **autoPET IV Challenge** Task I. For detailed method description, please refer to our preprint paper: &nbsp; [![arXiv](https://img.shields.io/badge/arXiv-2509.02402-b31b1b.svg)](https://arxiv.org/abs/2509.02402)

## Overview
In autoPET IV challenge (Task I), we primarily investigated the incorporation of human guidance for lesion segmentation in PET/CT. Inspired by top-performing teams in preceding challenges, our method adopts an integrated pipeline, including Tracer Classification, Lesion Segmentation, Organ Supervision and Post-processing. To efficiently leverage iterative interactions and progressively enhance segmentation accuracy, we explored stochastic click sampling strategies during training.

### Classifier:
- **Tracer Classifier**: Weights for tracer classification on the autoPET dataset was made publicly available by H. Kalisch et al. at https://github.com/hakal104/autoPETIII/ .

### Segmentation Models:
- **FDG Models**: 
For FDG lesion segmentation, we trained two nnU-Net models with identical architecture and methods but different training strategies:
  1. Full Guided model: All 10 FG/BG clicks were concatenated as input channels to enhance the model's performance with maximal human guidance.
  2. Stochastic Click Sampling: A random number of clicks (between 0 and 10) were sampled to simulate varying levels of user interaction in training data, so as to create a more balanced model optimized for progressive human interaction.

    In final submission, we chose different models based on the number of clicks provided in the test phase. For cases with limited user guidance (0-5 clicks), the Stochastic Click Sampling model is employed for prediction; for cases with dense human guidance (6-10 clicks), the Full Guided model would be selected.

- **PSMA Model**: 
The PSMA model was trained specifically on the PSMA dataset with the stochastic click sampling approach. A pre-trained weights by M. Rokuss et al. (https://zenodo.org/records/13753413) was used for initialization.

For detailed method description, please refer to our paper.

## Model Checkpoints

All model weights are available under https://drive.google.com/drive/folders/1kdblExa1I6QFtoZHxEVwVRittJb1hdpp?usp=drive_link.

They include the following files/folders:

- **Tracer Classifier**: Model weights for the tracer classifier are available in `tracer_classifier.pt`.
- **nnUNet Models**: There are three nnUNet models available in autopet4-nnUNet_results.zip:
  - `Dataset723_OrganSupervisedAutoPETIII`: Organ-Supervised Model trained on unified dataset with full guidance (10 clicks input).
  - `Dataset820_AutoPET_PSMA_MultiClicks`: Model trained on PSMA data with stochastic click sampling in training data.
  - `Dataset826_AutoPET_All_MultiClicks_Organs`: Organ-Supervised Model trained on unified dataset with stochastic click sampling in training data.

