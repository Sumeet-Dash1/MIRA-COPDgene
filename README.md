# MIRA-COPDgene

This repository contains code, data, and documentation for a project focused on **Deformable Image Registration** using lung CT images from the COPDGene study. The project evaluates spatial accuracy of deformable image registration and explores various segmentation and registration pipelines.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Contributors](#contributors)

---

## Overview
This project implements and evaluates different methods for lung segmentation and deformable registration using lung CT images. The work explores techniques such as threshold-based segmentation, deep learning-based segmentation, and B-spline-based transformations for registration accuracy improvement.

The repository also includes scripts for processing, analyzing, and comparing the results of registration pipelines and transformation techniques.

---

## Features
- **Segmentation Pipelines**:
  - Threshold-based segmentation
  - Deep learning-based segmentation using the LungMask model
  - 3D Slicer plugin-based segmentation
- **Registration Techniques**:
  - Rigid and non-rigid (B-spline) transformations using Elastix
  - Evaluation of sliding motion and lung mask effects
- **Performance Metrics**:
  - Target Registration Error (TRE) calculation
  - Comparisons across multiple COPD cases

---

## Installation
### Prerequisites
- Python 3.8+
- Libraries:
  - `SimpleITK`
  - `NumPy`
  - `Matplotlib`
  - `Elastix`
  - `scipy`
  - `nibabel`
  - `opencv-python`
- ITK-SNAP or 3D Slicer (optional for visualization)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Sumeet-Dash1/MIRA-COPDgene.git
   cd MIRA-COPDgene

## Directory Structure

## Directory Structure

```plaintext
MIRA-COPDgene/
├── data/                      # Raw input data
├── data_test/                 # Test dataset
├── elastix/                   # Elastix executable file
├── mask/                      # Masks for segmentation
├── mask_test/                 # Test masks
├── parameter/                 # Parameter files for registration
├── results/                   # Final registration results
├── scripts/                   # Scripts for automation or batch processing
├── submit/                    # Submission-related files or configurations
├── utils/                     # Utility scripts
│   ├── datasets.py            # Dataset management scripts
│   ├── metrics.py             # Scripts for calculating metrics
│   ├── preprocessing.py       # Preprocessing pipeline
│   ├── registration.py        # Registration pipeline
│   ├── segment.py             # Segmentation-related functions
│   ├── utils.py               # General utility functions
│   ├── vis.py                 # Visualization scripts
├── Preprocessing.ipynb        # Jupyter notebook for preprocessing
├── Registration.ipynb         # Jupyter notebook for registration
├── Test_Registration.ipynb    # Jupyter notebook for testing registration
```

## Contributors
- [Sumeet Dash](https://github.com/Sumeet-Dash1)
- [Huy Tran Quang](https://github.com/huytrnq) 
 
