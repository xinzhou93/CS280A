---
title: "The Power of Diffusion Models"
tags: [project, cs280a]
---

# Overview

In this project, I explored diffusion models using DeepFloyd IF, a two-stage text-to-image model. Part A focuses on understanding the diffusion process and implementing sampling loops for various creative tasks including image generation, inpainting, and visual illusions.

**Random Seed Used: 100**

# Part 0: Setup

DeepFloyd IF is a two-stage diffusion model: Stage 1 generates 64×64 images from text prompts, and Stage 2 upsamples them to 256×256. I created custom prompt embeddings using the Huggingface T5 Encoder and experimented with different `num_inference_steps` values.

**Random Seed: 100** (used for all subsequent parts)

### Generated Images with `num_inference_steps=5`

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part0_steps5_a_cyberpunk_city_at_night.png" alt="Cyberpunk city" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    "a cyberpunk city at night"
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps5_an_astronaut_riding_a_horse_on.png" alt="Astronaut on Mars" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    "an astronaut riding a horse on mars"
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps5_a_cozy_cabin_in_the_woods_with.png" alt="Cozy cabin" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    "a cozy cabin in the woods with snow"
    </figcaption>
  </figure>
</div>

### Generated Images with `num_inference_steps=20`

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part0_steps20_a_cyberpunk_city_at_night.png" alt="Cyberpunk city" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    "a cyberpunk city at night"
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps20_an_astronaut_riding_a_horse_on.png" alt="Astronaut on Mars" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    "an astronaut riding a horse on mars"
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps20_a_cozy_cabin_in_the_woods_with.png" alt="Cozy cabin" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    "a cozy cabin in the woods with snow"
    </figcaption>
  </figure>
</div>

### Generated Images with `num_inference_steps=50`

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P5A/part0_steps50_a_cyberpunk_city_at_night.png" alt="Cyberpunk city" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    "a cyberpunk city at night"
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps50_an_astronaut_riding_a_horse_on.png" alt="Astronaut on Mars" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    "an astronaut riding a horse on mars"
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P5A/part0_steps50_a_cozy_cabin_in_the_woods_with.png" alt="Cozy cabin" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    "a cozy cabin in the woods with snow"
    </figcaption>
  </figure>
</div>

Comparing different `num_inference_steps`:

- **5 steps**: Fast generation but images appear blurry and lack fine details. Shapes are recognizable but textures are muddy.
- **20 steps**: Good balance between speed and quality. Images are coherent with reasonable details.
- **50 steps**: Highest quality with sharper details and better textures. The cyberpunk city shows more defined buildings, the astronaut has clearer features, and the cabin scene has better snow and tree details.

The prompts align well with the outputs—the model correctly interprets "cyberpunk" style with neon colors, places the astronaut on a Mars-like red landscape, and creates a winter cabin scene with snow.


# Part 1: Sampling Loops

## 1.1 Implementing the Forward Process

The forward process gradually adds Gaussian noise to a clean image according to a noise schedule. Given an image $x_0$, we can obtain a noisy version $x_t$ at any timestep $t$ using:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

where $\epsilon \sim \mathcal{N}(0, I)$ is random noise and $\bar{\alpha}_t$ is the cumulative product of $(1 - \beta_t)$ values from the noise schedule.

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

**Observations:**
- At $t=250$: The image retains most of its structure with slight noise—the Campanile is clearly recognizable
- At $t=500$: Moderate noise level; major shapes are visible but details are obscured
- At $t=750$: Heavy noise dominates; the image is barely recognizable, approaching pure noise

## 1.2 Classical Denoising

Can we recover the original image using classical methods? Let's try Gaussian blur filtering on the noisy images from Part 1.1.

**Row 1: Noisy images (from 1.1) | Row 2: Gaussian blur denoised**

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

**Observations:**
- Gaussian blur faces an impossible trade-off: reduce noise vs. preserve detail
- At $t=250$: Blur removes some noise but softens edges—tower shape preserved but details lost
- At $t=500$: Larger kernel needed; result is a blurry blob with vague structure
- At $t=750$: No parameter choice can recover the image—classical denoising completely fails

This demonstrates why learned denoisers (diffusion models) are so powerful: they can remove noise while preserving—or even hallucinating plausible—image structure.
## 1.3 One-Step Denoising
## 1.4 Iterative Denoising
## 1.5 Diffusion Model Sampling
## 1.6 Classifier-Free Guidance (CFG)
## 1.7 Image-to-image Translation
## 1.8 Visual Anagrams
## 1.9 Hybrid Images

# Part 2: Bells & Whistles