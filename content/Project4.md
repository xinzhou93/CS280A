---
title: "Neural Radiance Field"
tags: [project, cs280a]
---

# Overview

# Part 0: Camera Calibration and 3D Scanning
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P01.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P4/P02.png" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

# Part 1: Fit a Neural Field to a 2D Image

## Model Architecture

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/mlp_img.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    image source: CS180 website: https://cal-cs180.github.io/fa25/hw/proj4/index.html
    </figcaption>
  </figure>
</div>

The neural field model learns a continuous representation of a 2D image by mapping pixel coordinates to RGB colors through a fully-connected MLP network.

Network Structure:
- Architecture: 4-layer fully-connected MLP
  - Input layer: 42D → 256D
  - Hidden layers: 256D → 256D (×2)
  - Output layer: 256D → 3D (RGB)
- Width: 256 neurons per hidden layer
- Activation functions: ReLU for hidden layers, Sigmoid for output (ensures RGB ∈ [0,1])

Positional Encoding:
- Frequency levels: L = 10
- Input dimension: $42D = 2D \text{ original coordinates} + 40D \text{ encoding } ( 2 * 2 * L + 2)$
- Formula: $$PE(x) = \{x, sin(2^0 \pi x), cos(2^0 \pi x), sin(2^1 \pi x), cos(2^1 \pi x), ..., sin(2^{L-1} \pi x), cos(2^{L-1} \pi x)\}$$
- High-frequency components enable the network to capture fine details that would otherwise be difficult to represent with a simple coordinate-based MLP

Training Configuration:

| Optimizer | Learning Rate | Loss Function | Batch Size | Total Iterations | Training Strategy |
|-----------|---------------|---------------|------------|------------------|-------------------|
| Adam | 1e-2 | MSE | 10,000 pixels | 2,000 | Random pixel sampling |

## Results

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P11.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P12.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

The training progression shows that the image reconstruction quality improves over iterations.

## Hyperparameter Comparisons

To understand the impact of network width and positional encoding frequency, I trained 4 models with different configurations in a 2×2 grid:

Hyperparameters:

| Configuration | L (Positional Encoding) | Hidden Dimension |
| ------------- | ----------------------- | ---------------- |
| Config 1      | 5                       | 128              |
| Config 2      | 5                       | 256              |
| Config 3      | 10                      | 128              |
| Config 4      | 10                      | 256              |

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P13.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/P15.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

**Hyperparameter Comparison Results:**

| Configuration | L (Positional Encoding) | Hidden Dimension | Final PSNR (dB) |
|---------------|-------------------------|------------------|-----------------|
| Config 1 | 5 | 128 | 34.2 |
| Config 2 | 5 | 256 | 34.9 |
| Config 3 | 10 | 128 | 33.6 |
| **Config 4 (Best)** | **10** | **256** | **36.8** |

**Key Observations:**
- **Higher L (more frequency components)** → Better capture of fine details and textures
- **Larger hidden_dim (more network capacity)** → Improved reconstruction quality and higher PSNR
- **Best configuration**: L=10, hidden_dim=256 achieves the highest PSNR of 36.8 dB

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P14.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Training progression showing reconstruction quality improving over iterations
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P4/P16.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Final reconstruction results and metrics
    </figcaption>
  </figure>
</div>


# Part 2: Fit a Neural Radiance Field from Multi-view Images

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P21.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/val_iter_0000.png" alt="Image 1" style="width: 50; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 0
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/val_iter_0400.png" alt="Image 1" style="width: 50; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 400
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/val_iter_0800.png" alt="Image 1" style="width: 50; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 800
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/val_iter_1000.png" alt="Image 1" style="width: 50; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 1000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/val_iter_2000.png" alt="Image 1" style="width: 50; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 2000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/val_iter_3000.png" alt="Image 1" style="width: 50; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 3000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/val_iter_4000.png" alt="Image 1" style="width: 50; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 4000
    </figcaption>
  </figure>
 <figure style="margin: 0;">
    <img src="/P4/val_iter_4500.png" alt="Image 1" style="width: 50; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 4500
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/val_iter_4999.png" alt="Image 1" style="width: 50; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 4999
    </figcaption>
  </figure>
</div>


<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P2_loss.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P2_PSNR.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P22.gif" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>



# Part 2.6: Training with Your Own Data

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_0000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 0
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_1000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 1000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_2000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 2000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_3000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 3000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_4000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 4000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_5000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 5000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_6000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 6000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_7000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 7000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_8000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 8000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_9000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 9000
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_9500.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 9500
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P26.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

## Novel Views

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/frame_000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P4/frame_003.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/frame_005.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/frame_007.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/frame_009.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P4/frame_011.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P4/frame_014.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P27.gif" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>


# Bells and Whistles

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/depth_frame_001.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/depth_frame_010.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/depth_frame_020.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/depth_frame_030.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/depth_frame_040.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P4/depth_frame_050.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>


<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/BW2.gif" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>