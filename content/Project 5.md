---
title: "Diffusion Models"
tags: [project, cs280a]
---

# Overview

In this project, I explored diffusion models through two parts. **Part A** uses DeepFloyd IF, a pretrained two-stage text-to-image model, to understand the diffusion process and implement sampling loops for image generation, inpainting, and visual illusions. **Part B** implements generative models from scratch on the MNIST dataset, starting with single-step denoising and progressing to Flow Matching with time-conditioned and class-conditioned UNets using Classifier-Free Guidance (CFG).

# Part A: The Power of Diffusion Models

# Part 0: Setup

DeepFloyd IF is a two-stage diffusion model: Stage 1 generates 64×64 images from text prompts, and Stage 2 upsamples them to 256×256. I created custom prompt embeddings using the Huggingface T5 Encoder and experimented with different `num_inference_steps` values. The `random seed` I used was 100.

Generated Images with `num_inference_steps=5`

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part0_steps5_a_cyberpunk_city_at_night.png" alt="Cyberpunk city" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps5_an_astronaut_riding_a_horse_on.png" alt="Astronaut on Mars" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps5_a_cozy_cabin_in_the_woods_with.png" alt="Cozy cabin" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

Generated Images with `num_inference_steps=20`

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part0_steps20_a_cyberpunk_city_at_night.png" alt="Cyberpunk city" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps20_an_astronaut_riding_a_horse_on.png" alt="Astronaut on Mars" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps20_a_cozy_cabin_in_the_woods_with.png" alt="Cozy cabin" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

Generated Images with `num_inference_steps=50`

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part0_steps50_a_cyberpunk_city_at_night.png" alt="Cyberpunk city" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps50_an_astronaut_riding_a_horse_on.png" alt="Astronaut on Mars" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps50_a_cozy_cabin_in_the_woods_with.png" alt="Cozy cabin" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

The prompts align well with the outputs—the model correctly interprets "cyberpunk" style with neon colors, places the astronaut on a Mars-like red landscape, and creates a winter cabin scene with snow. However, the created images with steps of 5 look a bit blurry and gray, compared to steps of 20 and 50 with colorful and sharp qualities.

# Part 1: Sampling Loops

## 1.1 Implementing the Forward Process

The forward process gradually adds Gaussian noise to a clean image according to a noise schedule. Given an image $x_0$, we can obtain a noisy version $x_t$ at any timestep $t$ using:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

where $\epsilon \sim \mathcal{N}(0, I)$ is random noise and $\bar{\alpha}_t$ is looked up from `alphas_cumprod` at timestep $t$.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_1_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Original (t=0)
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part1_1_noisy_t250.png" alt="t=250" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    t=250
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part1_1_noisy_t500.png" alt="t=500" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    t=500
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part1_1_noisy_t750.png" alt="t=750" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    t=750
    </figcaption>
  </figure>
</div>

## 1.2 Classical Denoising

This part tries to recover the original image using Gaussian blur filtering on the noisy images from Part 1.1.

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_1_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_1_noisy_t250.png" alt="Noisy t=250" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Noisy t=250</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_1_noisy_t500.png" alt="Noisy t=500" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Noisy t=500</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_1_noisy_t750.png" alt="Noisy t=750" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Noisy t=750</figcaption>
  </figure>

  <figure style="margin: 0;">
    <div style="height: 100%; display: flex; align-items: center; justify-content: center; color: gray; font-size: 0.9em;">—</div>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_2_blur_t250.png" alt="Blur t=250" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Blur k=5, σ=2.0</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_2_blur_t500.png" alt="Blur t=500" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Blur k=7, σ=2.0</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_2_blur_t750.png" alt="Blur t=750" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Blur k=9, σ=3.0</figcaption>
  </figure>
</div>

I used different kernel size and sigma value to get the best Gaussian-denoised version for each $t$. However, the images clearly demonstrate that the classical method denoises the image at the sacrifice of details. It is impossible to retrieve the tower structure when $t = 750$.

## 1.3 One-Step Denoising

Now let's use the pretrained diffusion model (UNet) to denoise. Given a noisy image $x_t$ and timestep $t$, the UNet predicts the noise $\epsilon$. We can then recover the original image using:

$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon}{\sqrt{\bar{\alpha}_t}}$$

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_1_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_1_noisy_t250.png" alt="Noisy t=250" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Noisy t=250</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_1_noisy_t500.png" alt="Noisy t=500" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Noisy t=500</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_1_noisy_t750.png" alt="Noisy t=750" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Noisy t=750</figcaption>
  </figure>

  <figure style="margin: 0;">
    <div style="height: 100%; display: flex; align-items: center; justify-content: center; color: gray; font-size: 0.9em;">—</div>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_2_blur_t250.png" alt="Blur t=250" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Gaussian Blur</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_2_blur_t500.png" alt="Blur t=500" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Gaussian Blur</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_2_blur_t750.png" alt="Blur t=750" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Gaussian Blur</figcaption>
  </figure>

  <figure style="margin: 0;">
    <div style="height: 100%; display: flex; align-items: center; justify-content: center; color: gray; font-size: 0.9em;">—</div>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_3_onestep_t250.png" alt="One-step t=250" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">One-Step UNet</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_3_onestep_t500.png" alt="One-step t=500" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">One-Step UNet</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_3_onestep_t750.png" alt="One-step t=750" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">One-Step UNet</figcaption>
  </figure>
</div>

The UNet dramatically outperforms Gaussian blur because it has learned the structure of natural images. However, one-step denoising can not still fully recover the original image when $t$ is large. From the images above, the tower at $t=750$ is still blurry and its appearance differs from the original image.

## 1.4 Iterative Denoising

Instead of denoising in one step, we can iteratively denoise by taking small steps from $x_t$ to $x_{t'}$ where $t' < t$. Using strided timesteps (stride=30) starting from t=690 (`i_start=10`), we apply the formula:

$$x_{t'} = \frac{\sqrt{\bar\alpha_{t'}}\beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t'})}{1 - \bar\alpha_t} x_t + v_\sigma$$

**Gradual Denoising Process (every 5th step):**

<div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_4_noisy_t690.png" alt="t=690" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">t=690 (start)</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_4_denoise_t660.png" alt="t=660" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">t=660</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_4_denoise_t510.png" alt="t=510" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">t=510</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_4_denoise_t360.png" alt="t=360" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">t=360</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_4_denoise_t210.png" alt="t=210" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">t=210</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_4_denoise_t60.png" alt="t=60" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">t=60</figcaption>
  </figure>
</div>

**Comparison: Original vs Iterative vs One-Step vs Gaussian Blur (all starting from t=690)**

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_4_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_4_iterative_clean.png" alt="Iterative" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Iterative Denoise</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_4_onestep.png" alt="One-Step" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">One-Step (t=690)</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_4_blur.png" alt="Gaussian Blur" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Gaussian Blur (t=690)</figcaption>
  </figure>
</div>

The gradual denoising shows the image progressively becoming cleaner as we step through timesteps. The result of iterative denoising
produces a clean, realistic tower—but I noticed it's not the original Campanile. Instead, it hallucinates a plausible tower based on the vague structure and the prompt "a high quality photo". One-step denoising at $t=690$ also hallucinates a different tower, but the result is blurry and distorted—too much noise to handle in one step
Gaussian blur $at t=690$ completely fails, producing an unrecognizable noise.

## 1.5 Diffusion Model Sampling

Instead of denoising a noisy image, we can generate images from scratch by starting from pure random noise and running `iterative_denoise` with `i_start=0`. The model "denoises" random noise into a coherent image guided by the prompt.

**5 Sampled Images** (prompt: "a high quality photo")

<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_5_sample_1.png" alt="Sample 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Sample 1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_5_sample_2.png" alt="Sample 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Sample 2</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_5_sample_3.png" alt="Sample 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Sample 3</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_5_sample_4.png" alt="Sample 4" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Sample 4</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_5_sample_5.png" alt="Sample 5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Sample 5</figcaption>
  </figure>
</div>

Each sample produces a completely different image since each starts from different random noise and the generic prompt "a high quality photo" produces varied natural-looking scenes. The images look gray and blurry because of using unconditional prompts.

## 1.6 Classifier-Free Guidance (CFG)

To improve image quality, we use Classifier-Free Guidance (CFG). The idea is to compute both:
- Conditional noise $\epsilon_c$: using the text prompt
- Unconditional noise $\epsilon_u$: using an empty prompt ""

Then combine them:
$$\epsilon = \epsilon_u + \gamma (\epsilon_c - \epsilon_u)$$

where $\gamma$ controls guidance strength. With $\gamma = 7$, we push strongly toward the prompt direction, producing sharper, more faithful images.

**5 CFG Samples** (prompt: "a high quality photo", $\gamma = 7$)

<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_6_cfg_sample_1.png" alt="CFG Sample 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Sample 1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_6_cfg_sample_2.png" alt="CFG Sample 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Sample 2</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_6_cfg_sample_3.png" alt="CFG Sample 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Sample 3</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_6_cfg_sample_4.png" alt="CFG Sample 4" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Sample 4</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_6_cfg_sample_5.png" alt="CFG Sample 5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Sample 5</figcaption>
  </figure>
</div>

Compared to Part 1.5 (without CFG), these images look more colorful with some details, since CFG amplifies the "prompt direction" by overshooting ($\gamma > 1$), making the model commit more strongly to generating prompt-faithful content.

## 1.7 Image-to-image Translation

### 1.7.1 SDEdit

Using the **SDEdit** algorithm, we can edit existing images by:
1. Adding noise to the original image (at different levels)
2. Denoising with CFG to "force" it back onto the natural image manifold

The noise level (`i_start`) controls how much the image changes:
- **Low i_start (1, 3)**: Heavy noise → model hallucinates more → big changes
- **High i_start (10, 20)**: Light noise → stays close to original

### Campanile

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_camp_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_camp_i1.png" alt="i=1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_camp_i3.png" alt="i=3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=3</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_camp_i5.png" alt="i=5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_camp_i7.png" alt="i=7" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=7</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_camp_i10.png" alt="i=10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=10</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_camp_i20.png" alt="i=20" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=20</figcaption>
  </figure>
</div>

### Web Image

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_web_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_web_i1.png" alt="i=1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_web_i3.png" alt="i=3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=3</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_web_i5.png" alt="i=5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_web_i7.png" alt="i=7" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=7</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_web_i10.png" alt="i=10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=10</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_web_i20.png" alt="i=20" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=20</figcaption>
  </figure>
</div>

### Hand-drawn Image 1

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_draw_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_draw_i1.png" alt="i=1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_draw_i3.png" alt="i=3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=3</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_draw_i5.png" alt="i=5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_draw_i7.png" alt="i=7" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=7</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_draw_i10.png" alt="i=10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=10</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_draw_i20.png" alt="i=20" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=20</figcaption>
  </figure>
</div>

### Hand-drawn Image 2

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_hand_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_hand_i1.png" alt="i=1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_hand_i3.png" alt="i=3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=3</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_hand_i5.png" alt="i=5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_hand_i7.png" alt="i=7" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=7</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_hand_i10.png" alt="i=10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=10</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_hand_i20.png" alt="i=20" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=20</figcaption>
  </figure>
</div>

The progression shows a smooth transition from "hallucinated" to "preserved". At **i_start=1** (most noise): The model almost completely reimagines the image, keeping only vague color/structure hints. However, at **i_start=20** (least noise): The result is similar to the original. Hand-drawn images and the campanile work particularly well, whereas the web image shows less creativity since the original image is already in the natural image manifold.

### 1.7.2 Inpainting

Using the **RePaint** algorithm, we can fill in masked regions of an image. At each denoising step, we keep the unmasked regions fixed (with appropriate noise added) and only let the model hallucinate inside the mask:

$$x_t \leftarrow \textbf{m} \cdot x_t + (1 - \textbf{m}) \cdot \text{forward}(x_{orig}, t)$$

### Campanile Inpainting

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_camp_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Campanile</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_camp_mask.png" alt="Mask" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Mask</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_camp_hole.png" alt="Hole to Fill" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Hole to Fill</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_camp_inpainted.png" alt="Inpainted" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Campanile Inpainted</figcaption>
  </figure>
</div>

### Custom Image 1

prompt: "an astronaut riding a horse on mars"

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_custom1_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_custom1_mask.png" alt="Mask" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Mask</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_custom1_hole.png" alt="Hole to Fill" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Hole to Fill</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_custom1_inpainted.png" alt="Inpainted" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Inpainted</figcaption>
  </figure>
</div>

### Custom Image 2

prompt: "a cyberpunk city at night"

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_custom2_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_custom2_mask.png" alt="Mask" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Mask</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_custom2_hole.png" alt="Hole to Fill" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Hole to Fill</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_2_custom2_inpainted.png" alt="Inpainted" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Inpainted</figcaption>
  </figure>
</div>

### 1.7.3 Text-Conditioned Image-to-image Translation

Instead of using the generic prompt `"a high quality photo"`, we can guide the image transformation with a specific text prompt. This gives us creative control over the style and content of the output.

### Campanile → "a cyberpunk city at night"

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_camp_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_camp_i1.png" alt="i=1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_camp_i3.png" alt="i=3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=3</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_camp_i5.png" alt="i=5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_camp_i7.png" alt="i=7" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=7</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_camp_i10.png" alt="i=10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=10</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_camp_i20.png" alt="i=20" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=20</figcaption>
  </figure>
</div>

The text prompt completely changes the style—instead of preserving the Campanile's identity, the model transforms it into cyberpunk cityscapes. At i=1 (heavy noise): it is almost fully reimagined as a neon-lit city; At i=20 (light noise): The tower structure is preserved but takes on a cyberpunk aesthetic with glowing lights

### Custom Image 1 → "a portrait of an ancient tree"

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom1_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom1_i1.png" alt="i=1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom1_i3.png" alt="i=3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=3</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom1_i5.png" alt="i=5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom1_i7.png" alt="i=7" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=7</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom1_i10.png" alt="i=10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=10</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom1_i20.png" alt="i=20" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=20</figcaption>
  </figure>
</div>

### Custom Image 2 → "a cozy cabin in the woods with snow"

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom2_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom2_i1.png" alt="i=1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom2_i3.png" alt="i=3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=3</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom2_i5.png" alt="i=5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom2_i7.png" alt="i=7" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=7</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom2_i10.png" alt="i=10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=10</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_7_3_custom2_i20.png" alt="i=20" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=20</figcaption>
  </figure>
</div>

## 1.8 Visual Anagrams

Visual Anagrams create optical illusions where an image looks like one thing, but when flipped upside down reveals something completely different. The algorithm works by averaging two noise estimates:
- $\epsilon_1$: Denoise normally with prompt $p_1$
- $\epsilon_2$: Flip image, denoise with prompt $p_2$, flip the noise back
- $\epsilon = (\epsilon_1 + \epsilon_2) / 2$

### Illusion 1: "an oil painting of a wise old wizard" ↔ "an oil painting of a mysterious forest"

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_8_illusion1_normal.png" alt="Normal" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Normal (wizard)</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_8_illusion1_flipped.png" alt="Flipped" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Flipped (forest)</figcaption>
  </figure>
</div>

### Illusion 2: "a watercolor painting of mountains" ↔ "a watercolor painting of ocean waves"

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_8_illusion2_normal.png" alt="Normal" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Normal (mountains)</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part1_8_illusion2_flipped.png" alt="Flipped" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Flipped (ocean waves)</figcaption>
  </figure>
</div>

The illusions work best when the two prompts have compatible structures such as wizard beard → tree roots, mountain peaks → wave crests). In addition, using matching art styles ("oil painting", "watercolor") helps unify the two views. The averaged noise estimate forces the model to find a compromise that satisfies both prompts

## 1.9 Hybrid Images

Factorized Diffusion creates hybrid images—images that look like one thing from far away but reveal something different up close (just like in Project 2).

The algorithm combines low and high frequency components from two different noise estimates:
- $\epsilon_1$: Denoise with prompt $p_1$ → apply low-pass filter
- $\epsilon_2$: Denoise with prompt $p_2$ → apply high-pass filter
- $\epsilon = f_{lowpass}(\epsilon_1) + f_{highpass}(\epsilon_2)$

### Hybrid 1: "an oil painting of a wise old wizard" (far) / "an oil painting of a mysterious forest" (close)

<div style="display: grid; grid-template-columns: repeat(1, 1fr); gap: 15px; text-align: center; max-width: 400px; margin: 0 auto;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_9_hybrid1.png" alt="Hybrid 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Squint or step back to see the cyberpunk city</figcaption>
  </figure>
</div>

### Hybrid 2: "a watercolor painting of mountains" (far) / "a watercolor painting of ocean waves" (close)

<div style="display: grid; grid-template-columns: repeat(1, 1fr); gap: 15px; text-align: center; max-width: 400px; margin: 0 auto;">
  <figure style="margin: 0;">
    <img src="/P5A/part1_9_hybrid2.png" alt="Hybrid 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Squint or step back to see the magical library</figcaption>
  </figure>
</div>

# Part A Bells & Whistles

## More Visual Anagrams

Beyond the upside-down flip in Part 1.8, we can create visual anagrams with other transformations.

### 90° Rotation Anagram: "a photo of a dog" ↔ "a photo of a cat"

Similar to the vertical flip in Part 1.8, we average two noise estimates:
- $\epsilon_1$: Denoise normally with prompt "dog"
- $\epsilon_2$: Rotate image 90°, denoise with prompt "cat", rotate noise back -90°

```python
image_rot = torch.rot90(image, k=1, dims=[2, 3])  # Rotate 90°
# ... denoise with prompt2 ...
noise_est_2 = torch.rot90(noise_est_2, k=-1, dims=[2, 3])  # Rotate back
```

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part2_rotation_normal.png" alt="Normal" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Normal (dog)</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_rotation_rotated.png" alt="Rotated" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Rotated 90° (cat)</figcaption>
  </figure>
</div>

### Quadrant Permutation Anagram: "a lithograph of waterfalls" ↔ "a lithograph of a skull"

This transformation shuffles the four quadrants of the image: [TL, TR, BL, BR] → [BR, BL, TR, TL]. The same permutation function reverses itself, so we apply it to both the image and the noise.

```python
def permute_quadrants(x):
    B, C, H, W = x.shape
    h2, w2 = H // 2, W // 2
    result = torch.zeros_like(x)
    result[:, :, :h2, :w2] = x[:, :, h2:, w2:]    # TL <- BR
    result[:, :, :h2, w2:] = x[:, :, h2:, :w2]    # TR <- BL
    result[:, :, h2:, :w2] = x[:, :, :h2, w2:]    # BL <- TR
    result[:, :, h2:, w2:] = x[:, :, :h2, :w2]    # BR <- TL
    return result

image_permuted = permute_quadrants(image)
# ... denoise with prompt2 ...
noise_est_2 = permute_quadrants(noise_est_2)  # Unpermute
```

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part2_permute_normal.png" alt="Normal" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Normal (waterfalls)</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_permute_permuted.png" alt="Permuted" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Permuted (skull)</figcaption>
  </figure>
</div>

90° rotation works well—the dog face transforms into a cat-like appearance when rotated. Quadrant permutation creates a puzzle-like effect where rearranging the four quadrants reveals a completely different image.

## UCB Logo Design

I used text-conditioned image-to-image translation (SDEdit) on a UCB logo to create a creative logo design.

### Logo 1: "a cyberpunk city at night"

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo_i1.png" alt="i=1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo_i3.png" alt="i=3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=3</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo_i5.png" alt="i=5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo_i7.png" alt="i=7" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=7</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo_i10.png" alt="i=10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=10</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo_i20.png" alt="i=20" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=20</figcaption>
  </figure>
</div>

### Logo 2: "a California bear"

<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo2_original.png" alt="Original" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">Original</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo2_i1.png" alt="i=1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo2_i3.png" alt="i=3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=3</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo2_i5.png" alt="i=5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo2_i7.png" alt="i=7" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=7</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo2_i10.png" alt="i=10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=10</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5A/part2_logo2_i20.png" alt="i=20" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.8em; color: gray; margin-top: 4px;">i=20</figcaption>
  </figure>
</div>

The intermediate noise levels (i=3 to i=7) produce the best logo designs—they retain the original structure while adding creative elements from the prompt. The "California bear" prompt works well with the UCB logo, enhancing the bear mascot while preserving the circular badge structure.

---

# Part B: Flow Matching from Scratch!

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

Training is similar to 2.3, but with class labels and dropout for CFG:

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
- Learning rate: 1e-3 (constant, no scheduler)
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
    <img src="/P5B/part2_5_epoch1_samples.png" alt="Epoch 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 1</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part2_5_epoch5_samples.png" alt="Epoch 5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 5</figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P5B/part2_5_epoch10_samples.png" alt="Epoch 10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.85em; color: gray; margin-top: 4px;">Epoch 10</figcaption>
  </figure>
</div>

Each image shows 4 instances of each digit (0-9) generated with CFG ($\gamma=5.0$). At epoch 1, digits are recognizable but noisy, showing that class conditioning is already working. By epoch 5, the digits are much cleaner with clear class separation. At epoch 10, high-quality digits with consistent style within each class are generated.

To simplify training, I removed the ExponentialLR scheduler and compensated by using a lower constant learning rate (1e-3 instead of 1e-2). The original scheduler decayed LR from 1e-2 to 1e-3 over 10 epochs, so using 1e-3 throughout provides stable training without the complexity of scheduling.

# Part B Bells & Whistles

## Improved Time-Conditioned Flow Matching

For bells & whistles, I trained an improved time-conditioned UNet with larger capacity and longer training:

**Improvements over baseline (Part 2.3):**
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
