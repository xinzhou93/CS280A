---
title: "The Power of Diffusion Models"
tags: [project, cs280a]
---

# Overview

In this project, I explored diffusion models using DeepFloyd IF, a two-stage text-to-image model. Part A focuses on understanding the diffusion process and implementing sampling loops for various creative tasks including image generation, inpainting, and visual illusions.

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

# Part 2: Bells & Whistles

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