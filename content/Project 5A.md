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

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/mlp_img.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    image source: CS180 website: https://cal-cs180.github.io/fa25/hw/proj4/index.html
    </figcaption>
  </figure>
</div>

## 1.2 Classical Denoising
## 1.3 One-Step Denoising
## 1.4 Iterative Denoising
## 1.5 Diffusion Model Sampling
## 1.6 Classifier-Free Guidance (CFG)
## 1.7 Image-to-image Translation
## 1.8 Visual Anagrams
## 1.9 Hybrid Images

# Part 2: Bells & Whistles