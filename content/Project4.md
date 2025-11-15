---
title: "Neural Radiance Field"
tags: [project, cs280a]
---

# Overview

This project explores Neural Radiance Fields (NeRF), a powerful technique for synthesizing novel views of 3D scenes from 2D images. The project is divided into several parts, progressively building understanding from simple 2D image fitting to complex 3D scene reconstruction.

**Part 1** begins with fitting a neural field to represent a 2D image, demonstrating how MLPs with positional encoding can learn continuous functions mapping pixel coordinates to RGB colors. I implemented a 4-layer network with L=10 positional encoding and explored how different hyperparameters (frequency levels and network width) affect reconstruction quality.

**Part 2** extends these concepts to 3D by implementing a full NeRF pipeline for the Lego bulldozer dataset. This involved implementing ray generation from cameras, stratified sampling along rays, an 8-layer NeRF architecture with skip connections, and volume rendering with alpha compositing. The model successfully reconstructs the 3D scene and enables rendering from novel camera viewpoints.

**Part 2.6** applies NeRF to my own real-world dataset: a rubber duck photographed with an iPhone. This required camera pose estimation using COLMAP, careful dataset filtering to ensure uniform camera distances, and extensive hyperparameter tuning (particularly near/far bounds). Despite challenges from real-world complexity, the model achieved ~20 dB PSNR and produces recognizable novel views.

**Bells and Whistles** implements depth map rendering for the Lego scene, visualizing the 3D geometric structure learned by NeRF through grayscale depth maps that show distance from camera to surfaces.

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

**Creating Rays from Cameras:**
- For each pixel `(i, j)` in the image, I construct a ray originating from the camera center.
- Ray direction is computed using the camera intrinsics (focal length) and pixel coordinates.
- The camera-to-world matrix (`c2w`) transforms rays from camera space to world space.
- Each ray is defined by origin `o` and direction `d`: $r(t) = o + td$

**Sampling Points Along Rays:**
- Sample `N=64` points uniformly between near and far bounds along each ray.
- Use stratified sampling: divide the ray into N bins and randomly sample within each bin to avoid fixed set of 3D points. This perturbation prevents overfitting and improves rendering quality

**Positional Encoding:**
High-frequency positional encoding enables the MLP to represent fine details:
- L=10 for 3D positions → 63D features ($3 + 3×2×10$)
- L=4 for viewing directions → 27D features ($3 + 3×2×4$)

**NeRF Model Architecture**
8-layer MLP with skip connections:
- **Layers 1-4**: Process encoded 3D position (63D → 256D)
- **Layer 5**: Skip connection concatenates original position encoding with layer 4 output
- **Layers 6-7**: Further process combined features
- **Layer 8**: Split into two heads:
  - Density head: Outputs volume density (1D, view-independent)
  - Feature head: Outputs 256D feature vector
- **Direction processing**: Concatenate feature vector with encoded direction (27D)
- **Final layer**: Outputs view-dependent RGB color (3D)

**Volume Rendering**
- Query NeRF at each sampled point along the ray to get $(rgb_i, \sigma_i)$
- Compute alpha values: $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$ where $\delta_i$ is distance between samples
- Compute transmittance: $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$ 
- Final pixel color: $C(r) = \sum_{i=1}^{N} T_i \alpha_i c_i$
- This accumulates colors from front to back, properly handling occlusion and transparency


**Training Configuration:**

| Loss Function | Optimizer | Learning Rate | Batch Size | Total Iterations | Sampling Strategy |
|---------------|-----------|---------------|------------|------------------|-------------------|
| MSE | Adam | 5e-4 | 4096 rays | 5000 | Random rays from all images |


<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P21.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Visualization of rays and samples with cameras (up to 100 rays)
    </figcaption>
  </figure>
</div>

## Training progression visualization with predicted images across iterations

During each training iteration, I randomly sample 4096 rays from all training images to ensure diverse coverage of the scene. For each ray, 64 points are sampled along its length and queried through the NeRF model to obtain color and density values. These values are then accumulated using volume rendering to produce the final pixel colors. The MSE loss is computed between the rendered colors and ground truth RGB values from the training images. This loss is backpropagated through the network to update the model weights via the Adam optimizer. Throughout training, I monitor the PSNR on a validation view to track reconstruction quality and ensure the model is learning effectively.
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

## Dataset Creation and Preprocessing

I captured my own dataset using an iPhone camera, photographing a toy duck on a table with an ArUco marker for scale reference. The capture process involved 360° coverage by walking around the object, capturing approximately 90 images from all angles. 

**Dataset Filtering Challenge:**

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/w.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Dataset with varying distances
    </figcaption>
  </figure>
</div>

After poses estimation, I discovered a critical issue from the Viser interface: cameras at varying distances from the object caused training instability. The outer loop of captures (farther from object) had significantly different distances than the inner loop. Thus I filtered the dataset to keep only cameras with uniform distance, resulting in $74$ training images shown in the section of Part 0, with mean distance of $0.192$ m ($19.2$cm) and standard deviation of only $0.048$m. This filtering dramatically improved training stability and reconstruction quality.

## Hyperparameter Tuning

**Near/Far Bounds Experimentation:**

Finding the correct near/far bounds was the most challenging aspect:

1. **Initial attempt** (near=$0.07$, far=$0.31$): Based on computed camera distances, but resulted in poor PSNR (~$14-17$ dB). The bounds were too tight and clipped important parts of the scene.

2. **Instructor's suggestion** (near=$0.02$, far=$0.5$): Still produced suboptimal results for my scene.

3. **Final solution** (near=$0.01$, far=$1.0$): Counterintuitively, using very wide bounds worked best, achieving ~$20$ dB PSNR. This allows the network to learn the actual scene boundaries rather than being constrained by potentially inaccurate estimates.

**Training Configuration:**

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Iterations | 10,000 | 2× longer than Lego due to real-world complexity |
| Hidden dimension | 256 | Standard capacity, balanced quality vs. speed |
| Batch size | 4096 rays | Same as Lego for fair comparison |
| Learning rate | 5e-4 | Standard Adam learning rate |
| Near/far bounds | 0.01 - 1.0 | Wide bounds to avoid clipping |

## Training Progression

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_0000.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 0
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P4/intermediate_iter_0500.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    iter = 500
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

## Results and Discussion

The final model achieved approximately $20$ dB PSNR, which falls short of the target $23$ dB but represents a reasonable result for real-world data. The reconstruction is recognizable and captures the overall shape and appearance of the duck, though it appears noticeably blurrier compared to the Lego dataset.

Several factors contribute to this lower quality. First, real-world complexity introduces challenges such as varying lighting conditions across captures, potential motion blur from hand-held photography, and reflective surfaces on the table that create inconsistent appearances. Second, the limited training data of $74$ views, while sufficient for basic reconstruction, provides less coverage. Finally, camera calibration uncertainty from pose estimation introduces small errors in camera positions and orientations, unlike the perfect poses used for Lego.

This project yielded several important lessons about real-world NeRF training. Most significantly, uniform camera distances proved more critical than simply maximizing the total number of images—filtering out the outer camera loop dramatically improved results despite reducing image count. Additionally, wide near/far bounds $(0.01-1.0)$ proved safer for real-world scenes where exact depth ranges are uncertain, allowing the network to learn scene boundaries rather than being constrained by potentially inaccurate estimates. Overall, this experience demonstrated that real-world NeRF reconstruction is significantly more challenging, requiring careful attention to data quality and hyperparameter tuning to achieve acceptable results.

# Bells and Whistles

## Depth Map Rendering for Lego Scene

I implemented depth map visualization to reveal the 3D geometric structure of the Lego scene. Depth maps show the distance from the camera to each surface point, providing clear insight into the spatial layout learned by NeRF.

**Implementation:**

NeRF's volume rendering can be extended to render depth values using the same accumulation principle as color rendering. Instead of accumulating RGB colors, we accumulate the expected depth along each ray:

$$D(r) = \sum_{i=1}^{N} T_i \alpha_i t_i$$

where $D(r)$ is the expected depth, $T_i$ is transmittance, $\alpha_i$ is opacity, and $t_i$ is the distance from camera to sample point $i$.

**Visualization Process:**
1. Query the trained NeRF model to get density values along each ray
2. Compute depth using weighted sum (closer points with higher opacity contribute more)
3. Normalize depth values based on scene bounds
4. Apply grayscale mapping: darker values = closer surfaces, lighter values = farther surfaces

I generated a 360° rotating depth map video using the same spherical camera trajectory as the RGB rendering.

**Sample Depth Frames:**

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

**360° Depth Map Video:**

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/BW2.gif" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Complete 360° rotation showing depth map from all viewing angles
    </figcaption>
  </figure>
</div>