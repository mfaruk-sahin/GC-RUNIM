# GC-RUNIM: Genetic-Cauchy and Runge-Kutta-driven Adaptive Intensive Mutation

This repository contains the official Python implementation of the **GC-RUNIM** algorithm, as proposed in the manuscript: 
*"Joint Optimization of Band Fusion and Multi-Level Thresholding via a Hybrid Metaheuristic Framework: Genetic-Cauchy and Runge-Kutta-driven Adaptive Intensive Mutation (GC-RUNIM)"*.

## 1. Algorithmic Overview
GC-RUNIM is an advanced hybrid metaheuristic framework engineered to simultaneously optimize band fusion weights and multi-level thresholding in high-dimensional multispectral satellite imagery. The architecture integrates a Genetic Algorithm (GA)-based global exploration phase with an elitist Cauchy mutation strategy. For the exploitation phase, it utilizes the Runge-Kutta (RUN) mechanism augmented by a novel **Adaptive Intensive Mutation (AIM)** operator to perform hyper-focused micro-refinement.

## 2. Reproducibility: Satellite Imagery Preprocessing
To ensure strict experimental reproducibility, the Sentinel-2 multispectral dataset is processed through a deterministic pipeline before evaluation. The `Sentinel2DataLoader` class performs the following automated steps:
* **Band Extraction:** Extraction of 8 critical spectral bands (B2, B3, B4, B5, B8, B8A, B11, B12) from the raw dataset.
* **Radiometric Normalization:** Min-max scaling is applied to project the raw reflectance values into a normalized floating-point tensor space of [0, 1]. This is a mathematical prerequisite to guarantee gradient stability during the multiplicative joint fitness evaluation.
* **Spatial Alignment (Resolution Matching):** The 20m spatial resolution bands (B5, B11, B12) are upscaled to 10m to match the reference visible/NIR bands (B2, B3, B4, B8) using bilinear interpolation with anti-aliasing enabled.
* **Band Stacking:** The aligned bands are stacked into a 3D multi-channel hyperspatial matrix (H, W, C).

## 3. Parameter Initialization
The structural stability of the GC-RUNIM framework relies on precise hyperparameter calibration. The default parameters, utilized in the experimental section of the paper, are configured in the `FinalHybridOptimizer` class as follows:
* **Population Size (N):** `50` (Configured to ensure sufficient multi-dimensional sampling capacity).
* **Maximum Iterations (Epochs):** `100`.
* **Exploration-Exploitation Transition Ratio ($\rho$):** `0.5` (Allocates exactly 50% of the epochs to the GA phase and 50% to the RUN phase).
* **Stagnation Limit:** `8` (Triggers early stopping in the GA phase if the fitness improvement falls below 1e-6).
* **Directed Perturbation Amplitude ($\alpha$):** `0.008` (Utilized during the RK-style local refinement to maximize inter-class contrast).

## 4. Dependencies
The framework relies on standard scientific computing and optimization libraries:
```bash
pip install numpy scipy scikit-image matplotlib seaborn pandas mealpy
