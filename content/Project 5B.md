---
title: "Flow Matching from Scratch!"
tags: [project, cs280a]
---

# Overview

In this project, I implemented generative models from scratch on the MNIST dataset. Part 1 explores single-step denoising with a UNet, demonstrating why directly predicting clean images from pure noise fails (the model outputs the dataset mean). Part 2 introduces Flow Matching, which learns to transport samples from noise to data by predicting velocity fields $u = x_1 - x_0$ along linear interpolation paths $x_t = (1-t)x_0 + tx_1$. I implemented both time-conditioned and class-conditioned UNets, using Classifier-Free Guidance (CFG) for improved sample quality.

# Part 1: Training a Single-Step Denoising UNet

## 1.1 Implementing the UNet

The UNet architecture consists of an encoder-decoder structure with skip connections. I implemented simple operations first, then composed them into blocks. D is the number of hidden channels.

**Simple Operations:**
- **Conv**: Conv2d(3,1,1) + BN + GELU — keeps resolution, changes channels
- **DownConv**: Conv2d(3,2,1) + BN + GELU — downsamples by 2×
- **UpConv**: ConvTranspose2d(4,2,1) + BN + GELU — upsamples by 2×
- **Flatten**: AvgPool(7) + GELU — 7×7 → 1×1
- **Unflatten**: ConvTranspose2d(7,7,0) + BN + GELU — 1×1 → 7×7

**Composed Operations:**
- **ConvBlock**: Conv → Conv (adds depth without changing dimensions)
- **DownBlock**: DownConv → ConvBlock
- **UpBlock**: UpConv → ConvBlock

The encoder downsamples while the decoder upsamples with skip connections (channel-wise `torch.cat()`).

## 1.2 Using the UNet to Train a Denoiser

The goal is to train a denoiser $D_\theta$ that maps noisy images $z$ back to clean images $x$, optimized with L2 loss: $L = \mathbb{E}_{z,x}||D_\theta(z) - x||^2$.

We generate training pairs $(z, x)$ using the noising process:
$$z = x + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Visualization of different noise levels:

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

For each training batch:
1. Take clean MNIST images $x$
2. Add noise with fixed $\sigma=0.5$: $z = x + 0.5 \cdot \epsilon$
3. Train the UNet to predict the clean image: $D_\theta(z) \rightarrow x$

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

After epoch 1, the model already produces reasonable denoising results. After epoch 5, the outputs are cleaner with sharper edges.

### 1.2.2 Out-of-Distribution Testing

Since the model was trained only with $\sigma=0.5$, I tested it on different noise levels to see how it generalizes.

Noisy inputs (top row) 

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

Denoised outputs (bottom row):

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

At $\sigma=0.0$ (no noise), the model slightly blurs the clean image since it expects some noise. At $\sigma=0.5$ (the trained level), the model performs best. At higher noise levels ($\sigma=0.8, 1.0$), the model struggles to recover details since it wasn't trained on those levels. This shows the limitation of training on a single noise level.

### 1.2.3 Denoising Pure Noise

To test whether a UNet can work as a generative model, I trained a new model where the input is pure noise $z = \epsilon \sim \mathcal{N}(0, I)$ and the target is a clean image $x$.

```
z ~ N(0, I)  # pure noise, no signal from x
output = D_θ(z)
loss = ||output - x||²
```

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

All outputs look like a blurry disk regardless of the input noise. This is the average of all training digits. With MSE loss, the model learns to predict the point that minimizes squared distance to all training examples. Since pure noise contains no information about which digit to generate, the optimal prediction is the mean of the dataset. 

# Part 2: Training a Flow Matching Model

## 2.1 Adding Time Conditioning to UNet

Flow matching uses a time-conditioned UNet that learns to predict the velocity field at each timestep $t \in [0, 1]$. I added FCBlocks that map scalar $t$ to embeddings, then modulate decoder features by element-wise multiplication:

The forward process interpolates between noise and data:
$$x_t = (1-t) \cdot x_0 + t \cdot x_1$$

where $x_0 \sim \mathcal{N}(0, I)$ is noise and $x_1$ is the clean image. The model learns to predict the velocity $u = x_1 - x_0$.

## 2.2 Training the UNet

For each training step:
1. Sample clean image $x_1$ from training set
2. Sample $t \sim \text{Uniform}(0,1)$ and noise $x_0 \sim \mathcal{N}(0,I)$
3. Compute $x_t = (1-t) \cdot x_0 + t \cdot x_1$
4. Take gradient descent step on $\nabla_\theta ||(x_1 - x_0) - u_\theta(x_t, t)||^2$

**Hyperparameters:**
- Batch size: 64
- Learning rate: 1e-2 with Exponential scheduler
- Hidden dimension D: 64
- Epochs: 10

**Training Loss:**

<img src="/P5B/part2_2_training_loss.png" alt="Training Loss" style="max-width: 600px;" />

## 2.3 Sampling from the UNet

Starting from pure noise $x_0 \sim \mathcal{N}(0, I)$, we iteratively apply the learned velocity field with $T=50$ timesteps:
$$x_{t+\Delta t} = x_t + \Delta t \cdot u_\theta(x_t, t)$$

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5B/part2_2_epoch1_samples.png" alt="Epoch 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part2_2_epoch5_samples.png" alt="Epoch 5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part2_2_epoch10_samples.png" alt="Epoch 10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 10</figcaption>
  </figure>
</div>

At epoch 1, samples are still noisy and blurry. By epoch 5, digits start to emerge with recognizable shapes. At epoch 10, clear and diverse digits are generated from pure noise. Unlike Part 1.2.3, flow matching can generate diverse samples because it learns the full trajectory from noise to data, not just a single-step mapping.

## 2.4 Adding Class-Conditioning to UNet

I extended the time-conditioned UNet to also condition on class labels (one-hot vector $c$), allowing generation of specific digits on demand. Following the spec, I added 2 more FCBlocks for class conditioning and modulate the decoder features:

```
fc1_t, fc1_c = FCBlock(1, D), FCBlock(num_classes, D)
fc2_t, fc2_c = FCBlock(1, 2D), FCBlock(num_classes, 2D)

t1, t2 = fc1_t(t), fc2_t(t)
c1, c2 = fc1_c(one_hot(c)), fc2_c(one_hot(c))

unflatten = c2 * unflatten + t2   # 7×7 level, 2D channels
up1 = c1 * up1 + t1               # 14×14 level, D channels
```

During training, with probability $p_{uncond}=0.1$, I drop the class conditioning (set $c$ to zero) so the model learns to work without it. This enables Classifier-Free Guidance at sampling time.

## 2.5 Training the UNet

Training is similar to 2.2, but with class labels and dropout for CFG:

```
repeat
    x_1, c ~ clean image and label from training set
    Make c into a one-hot vector
    with probability p_uncond set c to zero-vector
    t ~ Uniform([0, 1])
    x_0 ~ N(0, I)
    x_t = (1 - t) * x_0 + t * x_1
    Take gradient descent step on ∇_θ ||(x_1 - x_0) - u_θ(x_t, t, c)||²
until happy
```

**Hyperparameters:**
- Batch size: 64
- Learning rate: 1e-2 with ExponentialLR scheduler
- Hidden dimension D: 64
- Epochs: 10
- $p_{uncond}$: 0.1

**Training Loss:**

<img src="/P5B/part2_5_training_loss.png" alt="Training Loss" style="max-width: 600px;" />

## 2.6 Sampling from the UNet

For sampling, we use Classifier-Free Guidance (CFG) with $\gamma=5.0$:

```
input: one-hot vector c, classifier guidance scale γ
x_t = x_0 ~ N(0, I)
for t from 0 to 1, step size 1/T do
    u_uncond = u_θ(x_t, t, 0)
    u_cond = u_θ(x_t, t, c)
    u = u_uncond + γ * (u_cond - u_uncond)    ▷ Classifier-free guidance
    x_t = x_t + (1/T) * u
end for
return x_t
```

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5B/part2_6_epoch1_samples.png" alt="Epoch 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part2_6_epoch5_samples.png" alt="Epoch 5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part2_6_epoch10_samples.png" alt="Epoch 10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 10</figcaption>
  </figure>
</div>

At epoch 1, digits are recognizable but noisy, showing that class conditioning is already working. By epoch 5, the digits are much cleaner with clear class separation. At epoch 10, high-quality digits with consistent style within each class are generated. CFG ($\gamma=5.0$) helps sharpen the samples by amplifying the difference between conditional and unconditional predictions.

# Part 3: Bells & Whistles

## Improved Time-Conditioned Flow Matching

For bells & whistles, I trained an improved time-conditioned UNet with larger capacity and longer training:

**Improvements over baseline (Part 2.2):**
| Parameter | Baseline | Improved |
|-----------|----------|----------|
| Hidden dim D | 64 | 128 |
| Epochs | 10 | 20 |
| Timesteps T | 50 | 100 |

**Training Loss:**

<img src="/P5B/part3_training_loss.png" alt="Training Loss" style="max-width: 600px;" />

**Samples at Different Epochs:**

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5B/part3_epoch1_samples.png" alt="Epoch 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part3_epoch5_samples.png" alt="Epoch 5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part3_epoch10_samples.png" alt="Epoch 10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 10</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part3_epoch15_samples.png" alt="Epoch 15" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 15</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part3_epoch20_samples.png" alt="Epoch 20" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 20</figcaption>
  </figure>
</div>

The larger model (D=128) produces sharper, more detailed digits. Extended training (20 epochs) allows the model to learn finer details, and more timesteps (T=100) during sampling provides smoother trajectories. By epoch 20, the samples are noticeably cleaner than the baseline at epoch 10.