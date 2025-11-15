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

Default setting:
L = 10
hidden dimension = 256
layer numbers = 4
batch size = 10000
num of iterations: 2000
learning rate: 1e-2

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

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P4/P14.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P4/P16.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
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