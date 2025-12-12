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
### 1.2.2 Out-of-Distribution Testing
### 1.2.3 Denoising Pure Noise

# Part 2: Training a Flow Matching Model

## 2.1 Adding Time Conditioning to UNet
## 2.2 Training the UNet
## 2.3 Sampling from the UNet
## 2.4 Adding Class-Conditioning to UNet
## 2.5 Training the UNet
## 2.6 Sampling from the UNet

# Part 3: Bells & Whistles