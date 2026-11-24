# Quasar Microlensing ML

**Machine-learning approaches for inferring quasar accretion-disc properties from gravitational microlensing light curves.**

## Overview

This repository contains the computational framework developed for a Master's thesis in Astronomy focused on applying machine-learning techniques to gravitational microlensing observations of strongly lensed quasars.

The main objective is to investigate alternative representations of microlensing light curves and evaluate their ability to recover physical properties of quasar accretion discs, particularly their characteristic half-light radius.

The project explores two complementary approaches to light-curve parametrization:

1. **Statistical feature parametrization:** Extraction of manually engineered statistical and temporal descriptors that characterize the variability and morphology of microlensing light curves.
2. **Autoencoder latent representation:** Use of recurrent neural-network autoencoders to learn compact, low-dimensional representations directly from the light curves.

Both representations are intended to serve as inputs to supervised regression models for estimating accretion-disc properties.

The framework is designed to work with synthetic light curves that incorporate observational effects such as photometric noise, irregular temporal sampling, and seasonal gaps. This allows the robustness of different machine-learning approaches to be investigated under conditions resembling real observational datasets.

## Repository Structure

The repository currently contains three main Python scripts:

```text
quasar-microlensing-ML/
│
├── generate_light_curves.py
├── extract_statistical_features.py
├── rnn_autoencoder.py
│
├── README.md
```
### 1. Light-Curve Generation

**File:** `generate_light_curves.py`

This script generates synthetic gravitational microlensing light curves from numerical magnification maps.

The main functionalities include:

- Loading magnification maps and their associated simulation parameters.
- Convolving magnification maps with extended-source brightness profiles.
- Generating random source trajectories across magnification maps.
- Producing light curves for different source sizes.
- Converting magnification values into magnitude variations.
- Incorporating observational effects, including:
  - Gaussian photometric noise.
  - Random temporal subsampling.
  - Seasonal observational gaps.
- Converting trajectory lengths into physical time units using the adopted microlensing parameters.

The generated datasets provide the input for the subsequent parametrization and machine-learning stages.

### 2. Statistical Feature Extraction

**File:** `extract_statistical_features.py`

This script extracts manually engineered descriptors from the synthetic microlensing light curves.

The parametrization is inspired by the methodology proposed by Jiménez-Vicente & Mediavilla (2025), with additional descriptors intended to characterize the statistical and temporal properties of the light curves.

The extracted features include:

- Mean magnitude.
- Absolute linear slope.
- Standard deviation of the detrended light curve.
- Interquartile range.
- Mean absolute first-order temporal derivative.
- Mean absolute second-order temporal derivative.
- Maximum and minimum detrended magnitudes.
- Waviness.

Temporal derivatives are estimated using local polynomial fits, allowing the analysis to account for irregular sampling and avoid directly differentiating across large observational gaps.

The resulting feature vectors provide a compact, interpretable representation of each light curve and can subsequently be used as inputs to supervised regression models.

### 3. Autoencoder-Based Latent Representation

**File:** `rnn_autoencoder.py`

This script implements an autoencoder architecture based on recurrent neural networks (GRUs) to learn compact representations of microlensing light curves.

Unlike the statistical parametrization, this approach does not require manually defining the descriptors. Instead, the neural network learns a compressed representation by reconstructing the input light curves through a lower-dimensional latent space.

The main functionalities include:

- Loading previously generated synthetic light curves.
- Handling variable-length light curves and missing observations.
- Training a recurrent autoencoder to reconstruct the input time series.
- Extracting latent-space representations from the trained encoder.
- Saving the learned representations for subsequent analysis.

The latent representation is currently designed as a low-dimensional feature vector (e.g., eight dimensions) for each light curve.

These vectors can be used as inputs to supervised regression algorithms, allowing a direct comparison between learned representations and manually engineered statistical features.

The two parametrization strategies are intended to be evaluated under equivalent simulation conditions and regression frameworks. This will allow their predictive performance, robustness to observational effects, and computational requirements to be compared.

## Scientific Motivation

The innermost regions of quasars are too compact to be spatially resolved by current astronomical facilities. Gravitational microlensing provides an indirect method to investigate their structure because the magnification experienced by an extended source depends on its size and surface-brightness distribution.

The temporal variability produced by microlensing therefore contains information about the physical properties of the accretion disc.

However, extracting this information from observed light curves is challenging due to the complexity of magnification patterns, the diversity of lens configurations, and observational limitations such as noise, irregular sampling, and seasonal gaps.

This project investigates whether machine-learning models operating on compact light-curve representations can provide an efficient and robust alternative for estimating accretion-disc properties.

## Future Development

The current implementation provides the initial components of the proposed methodology. Future development will focus on:

- Extending the synthetic light-curve generator to explore a wider range of convergence and shear configurations.
- Investigating alternative accretion-disc brightness profiles.
- Refining and evaluating the statistical feature set.
- Exploring different autoencoder architectures and latent-space dimensionalities.
- Implementing and optimizing supervised regression models for accretion-disc size inference.
- Comparing both parametrization strategies across different regions of the convergence--shear parameter space.
- Evaluating model robustness to observational noise, irregular sampling, and seasonal gaps.
- Applying the developed methodology to real multiwavelength observations of strongly lensed quasars.

Potential observational datasets include OGLE, the Zwicky Transient Facility (ZTF), VISTA Variables in the Vía Láctea (VVV), and suitable Gaia observations of systems identified through the Gaia GraL catalogue.

An important long-term goal is to use multiwavelength microlensing measurements to constrain the wavelength-dependent accretion-disc size and investigate the underlying radial temperature profile.

## Requirements
The project is developed in Python and relies primarily on scientific computing and machine-learning libraries.
The current implementation uses or may require:
- NumPy
- SciPy
- Matplotlib
- Astropy
- scikit-learn
- PyTorch

The exact dependencies and versions will be documented as the implementation evolves.

## Status

**Development stage:** Initial implementation.

The repository currently contains the synthetic light-curve generation pipeline, statistical feature extraction, and recurrent autoencoder implementation. The integration and systematic comparison of supervised regression models, as well as the application to real observational datasets, remain part of the ongoing thesis development.

## Main References

- Kochanek, C. S. (2004). *Quantitative interpretation of quasar microlensing light curves*. The Astrophysical Journal.
- Vernardos, G., & Tsagkatakis, G. (2019). *Deep learning for quasar microlensing light curves*. Monthly Notices of the Royal Astronomical Society.
- Jiménez-Vicente, J., & Mediavilla, E. (2025). *Machine-learning analysis of quasar microlensing light curves*. Monthly Notices of the Royal Astronomical Society.

---

**Author:** Juan D. Galan V.  
**Project:** Master's Thesis  
**Topic:** Machine Learning and Gravitational Microlensing of Quasars
