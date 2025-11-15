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
    Hyperparameters comparision of my portrait
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/P15.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Hyperparameters comparision of the fox image
    </figcaption>
  </figure>
</div>

**Key Observations:**
- Higher L can better capture fine details and textures
- Larger network width with more neurons can improve reconstruction quality and higher PSNR
- The best configuration is $L=10, width=256$, which achieves the highest PSNR for both images.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P14.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    PSNR for my portrait
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P4/P16.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    PSNR for the fox image
    </figcaption>
  </figure>
</div>


# Part 2: Fit a Neural Radiance Field from Multi-view Images

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/mlp_nerf1.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    image source: CS180 website: https://cal-cs180.github.io/fa25/hw/proj4/index.html
    </figcaption>
  </figure>
</div>

## Implementation Overview

NeRF represents a 3D scene as a continuous volumetric field that maps 5D coordinates (3D position + 2D viewing direction) to volume density and view-dependent color.

### Part 2.1-2.2: Ray Generation and Sampling

**Creating Rays from Cameras:**
- For each pixel (i, j) in the image, I construct a ray originating from the camera center
- Ray direction is computed using the camera intrinsics (focal length) and pixel coordinates
- The camera-to-world matrix (c2w) transforms rays from camera space to world space
- Each ray is defined by origin **o** and direction **d**: $r(t) = o + td$

**Sampling Points Along Rays:**
- Sample N=64 points uniformly between near and far bounds along each ray
- Use stratified sampling: divide the ray into N bins and randomly sample within each bin
- This perturbation prevents aliasing and improves rendering quality
- Sampled points: $t_i \sim U[t_{near} + \frac{i}{N}(t_{far}-t_{near}), t_{near} + \frac{i+1}{N}(t_{far}-t_{near})]$

### Part 2.3: Positional Encoding

High-frequency positional encoding enables the MLP to represent fine details:
- Position encoding: $\gamma(x) = [x, \sin(2^0\pi x), \cos(2^0\pi x), ..., \sin(2^{L-1}\pi x), \cos(2^{L-1}\pi x)]$
- L=10 for 3D positions → 63D features (3 + 3×2×10)
- L=4 for viewing directions → 27D features (3 + 3×2×4)
- Without encoding, the network struggles to capture high-frequency geometric details

### Part 2.4: NeRF Model Architecture

8-layer MLP with skip connections:
- **Layers 1-4**: Process encoded 3D position (63D → 256D)
- **Layer 5**: Skip connection concatenates original position encoding with layer 4 output
- **Layers 6-7**: Further process combined features
- **Layer 8**: Split into two heads:
  - Density head: Outputs volume density σ (1D, view-independent)
  - Feature head: Outputs 256D feature vector
- **Direction processing**: Concatenate feature vector with encoded direction (27D)
- **Final layer**: Outputs view-dependent RGB color (3D)

### Part 2.5: Volume Rendering

Classical volume rendering with alpha compositing:
- Query NeRF at each sampled point along the ray to get $(rgb_i, \sigma_i)$
- Compute alpha values: $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$ where $\delta_i$ is distance between samples
- Compute transmittance: $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$ (light transmission up to point i)
- Final pixel color: $C(r) = \sum_{i=1}^{N} T_i \alpha_i c_i$
- This accumulates colors from front to back, properly handling occlusion and transparency

### Part 2.6: Training Process

**Dataset:** Lego bulldozer with 100 training views (200×200 resolution)

**Training Configuration:**

| Loss Function | Optimizer | Learning Rate | Batch Size | Total Iterations | Sampling Strategy |
|---------------|-----------|---------------|------------|------------------|-------------------|
| MSE | Adam | 5e-4 | 4096 rays | 5000 | Random rays from all images |

**Training Procedure:**
1. Randomly sample 4096 rays from all training images
2. For each ray, sample 64 points and query NeRF
3. Render pixel colors using volume rendering
4. Compute MSE loss between rendered and ground truth RGB
5. Backpropagate and update network weights
6. Monitor PSNR on validation view to track reconstruction quality

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P21.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Visualization of rays and samples with cameras (up to 100 rays)
    </figcaption>
  </figure>
</div>

## Training progression visualization with predicted images across iterations
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

## Spherical rendering
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P22.gif" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Spherical rendering video of the Lego using provided test cameras
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