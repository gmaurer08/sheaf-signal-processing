# Sparse Sheaf Signal Processing

This repository contains the code for the sheaf signal processing project for the course *Signal Processing for Machine Learning* (2025/2026) of the Data Science Master's Degree at Sapienza.

A power point presentation of the project is available here: [presentation](Sparse_Sheaf_Signal_Processing.pdf)

This project explores sparse sheaf signal processing over tangent vector bundles on point clouds in $\mathbb{R}^3$ via wavelet dictionaries. The goal is to compare the performances on signal compression/denoising tasks using wavelet dictionaries built from the connection Laplacian (VDM) and the trivial graph Laplacian.

**Structure**
* Implementation of a Vector Diffusion Mapping class
* Implementation of a Wavelet class
* Signal compression and signal denoising experiments

**Datasets**
* Random point cloud in a $[0.5,0.5]^3$ cube in $\mathbb{R}^3$ (300 nodes)
* Fibonacci sphere in $\mathbb{R}^3$ (300 nodes)
* Swiss wind speed and directions dataset (151 nodes)
* ERA5 dataset of wind measurements around the Arctic Circle (100 nodes)

### Structure of the repository

```text
sheaf-signal-processing/
├── README.md
├── LICENSE
├── .gitignore
├── main.ipynb                  # main notebook
├── vdm.ipynb                   # notebook to test the vdm class
├── wavelet.ipynb               # notebook to test the wavelet class
├── interactive_results.html    # interactive plot with the signal compression and denoising results
├── python/
   ├── tsp.py                   # topological signal processing class
   ├── utils.py                 # useful functions
   ├── vdm.py                   # vector diffusion mapping class
   └── wavelet.py               # wavelet dictionary class
```
