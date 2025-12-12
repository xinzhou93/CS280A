---
title: "Flow Matching from Scratch!"
tags: [project, cs280a]
---

# Overview


# Part 1: Training a Single-Step Denoising UNet

## 1.1 Implementing the UNet

The UNet architecture consists of an encoder-decoder structure with skip connections. We implement the following components:

- **Conv**: Conv2d(3,1,1) + BatchNorm + GELU
- **DownConv**: Conv2d(3,2,1) + BatchNorm + GELU (downsamples by 2)
- **UpConv**: ConvTranspose2d(4,2,1) + BatchNorm + GELU (upsamples by 2)
- **Flatten**: AvgPool(7) + GELU (7×7 → 1×1)
- **Unflatten**: ConvTranspose2d(7,7,0) + BatchNorm + GELU (1×1 → 7×7)

The unconditional UNet takes a noisy image and predicts the clean image directly.

## 1.2 Using the UNet to Train a Denoiser

The noising process adds Gaussian noise to clean images:
$$z = x + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_noisy_sigma0.0.png" alt="σ=0.0" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">σ=0.0</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_noisy_sigma0.2.png" alt="σ=0.2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">σ=0.2</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_noisy_sigma0.4.png" alt="σ=0.4" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">σ=0.4</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_noisy_sigma0.5.png" alt="σ=0.5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">σ=0.5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_noisy_sigma0.6.png" alt="σ=0.6" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">σ=0.6</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_noisy_sigma0.8.png" alt="σ=0.8" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">σ=0.8</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_noisy_sigma1.0.png" alt="σ=1.0" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">σ=1.0</figcaption>
  </figure>
</div>

### 1.2.1 Training

We train the UNet to denoise images with σ=0.5. The model learns to predict the clean image from the noisy input.

**Hyperparameters:**
- Batch size: 256
- Learning rate: 1e-4
- Hidden dimension D: 128
- Epochs: 5

**Training Loss:**

<img src="/P5B/part1_2_1_training_loss.png" alt="Training Loss" style="max-width: 600px;" />

**Results after Epoch 1:**

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch1_sample1_input.png" alt="Input" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">Input</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch1_sample1_noisy.png" alt="Noisy" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">Noisy (σ=0.5)</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch1_sample1_denoised.png" alt="Denoised" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">Output</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch1_sample2_input.png" alt="Input" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch1_sample2_noisy.png" alt="Noisy" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch1_sample2_denoised.png" alt="Denoised" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch1_sample3_input.png" alt="Input" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch1_sample3_noisy.png" alt="Noisy" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch1_sample3_denoised.png" alt="Denoised" style="width: 100%; height: auto; display: block;" />
  </figure>
</div>

**Results after Epoch 5:**

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch5_sample1_input.png" alt="Input" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">Input</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch5_sample1_noisy.png" alt="Noisy" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">Noisy (σ=0.5)</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch5_sample1_denoised.png" alt="Denoised" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">Output</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch5_sample2_input.png" alt="Input" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch5_sample2_noisy.png" alt="Noisy" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch5_sample2_denoised.png" alt="Denoised" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch5_sample3_input.png" alt="Input" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch5_sample3_noisy.png" alt="Noisy" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_1_epoch5_sample3_denoised.png" alt="Denoised" style="width: 100%; height: auto; display: block;" />
  </figure>
</div>

**Observations:**
- After epoch 1, the model already produces reasonable denoising results
- After epoch 5, the outputs are cleaner with sharper edges

### 1.2.2 Out-of-Distribution Testing

The model was trained with σ=0.5. How does it perform on different noise levels?

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.0_noisy.png" alt="σ=0.0" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.0</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.2_noisy.png" alt="σ=0.2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.2</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.4_noisy.png" alt="σ=0.4" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.4</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.5_noisy.png" alt="σ=0.5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.6_noisy.png" alt="σ=0.6" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.6</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.8_noisy.png" alt="σ=0.8" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.8</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma1.0_noisy.png" alt="σ=1.0" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=1.0</figcaption>
  </figure>
</div>

**Noisy inputs (top row) → Denoised outputs (bottom row):**

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.0_denoised.png" alt="σ=0.0" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.0</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.2_denoised.png" alt="σ=0.2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.2</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.4_denoised.png" alt="σ=0.4" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.4</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.5_denoised.png" alt="σ=0.5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.6_denoised.png" alt="σ=0.6" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.6</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma0.8_denoised.png" alt="σ=0.8" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=0.8</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_2_sigma1.0_denoised.png" alt="σ=1.0" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray;">σ=1.0</figcaption>
  </figure>
</div>

**Observations:**
- At σ=0.0 (no noise), the model slightly blurs the clean image
- At σ=0.5 (trained level), the model performs best
- At higher noise levels (σ=0.8, 1.0), the model struggles to recover details since it wasn't trained on those levels

### 1.2.3 Denoising Pure Noise

What if we try to denoise pure noise? We train a new model where the input is z = ε ~ N(0, I) and the target is a clean image.

**Training Loss:**

<img src="/P5B/part1_2_3_training_loss.png" alt="Training Loss" style="max-width: 600px;" />

**Samples after Epoch 1:**

<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_3_epoch1_sample1.png" alt="Sample 1" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_3_epoch1_sample2.png" alt="Sample 2" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_3_epoch1_sample3.png" alt="Sample 3" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_3_epoch1_sample4.png" alt="Sample 4" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_3_epoch1_sample5.png" alt="Sample 5" style="width: 100%; height: auto; display: block;" />
  </figure>
</div>

**Samples after Epoch 5:**

<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_3_epoch5_sample1.png" alt="Sample 1" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_3_epoch5_sample2.png" alt="Sample 2" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_3_epoch5_sample3.png" alt="Sample 3" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_3_epoch5_sample4.png" alt="Sample 4" style="width: 100%; height: auto; display: block;" />
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part1_2_3_epoch5_sample5.png" alt="Sample 5" style="width: 100%; height: auto; display: block;" />
  </figure>
</div>

**Observations:**
- All outputs look like a blurry oval/blob shape regardless of the input noise
- This is the **average (centroid)** of all training digits
- With MSE loss, the model learns to predict the point that minimizes squared distance to ALL training examples
- Since pure noise contains no information about which digit to generate, the optimal prediction is the mean of the dataset
- This demonstrates why single-step denoising from pure noise cannot work as a generative model

# Part 2: Training a Flow Matching Model

## 2.1 Adding Time Conditioning to UNet
## 2.2 Training the UNet
## 2.3 Sampling from the UNet
## 2.4 Adding Class-Conditioning to UNet
## 2.5 Training the UNet
## 2.6 Sampling from the UNet

# Part 3: Bells & Whistles