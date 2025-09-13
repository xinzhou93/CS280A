---
title: "Colorizing the Prokudin-Gorskii photo collection"
tags: [project, cs280a]
---

# Overview

# Main Implementation

## Exhaustive Method
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
  <img src="/P1/j1_align.jpg" alt="img1" />
  <img src="/P1/j2_align.jpg" alt="img2" />
  <img src="/P1/j3_align.jpg" alt="img3" />
</div>

## Image Pyramid with Predefined borders

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P1/1_aligned.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      G(5, 65), R(-2, 133)<br>
      Runtime: 3.1 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/2_aligned.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      G(-12, 46), R(-20, 109)<br>
      Runtime: 3.3 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/3_aligned.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      G(3, 41), R(-12, 98)<br>
      Runtime: 3.15 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/4_aligned.jpg" alt="Image 4" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      G(15, 22), R(37,80)<br>
      Runtime: 3.21 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/5_aligned.jpg" alt="Image 5" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      G(-8, 90), R(-16, 158)<br>
      Runtime: 3.22 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/6_aligned.jpg" alt="Image 6" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      R(-10, -54), R(-21, 127)<br>
      Runtime: 3.25 sec
    </figcaption>
  </figure>
  
  <figure style="margin: 0;">
    <img src="/P1/7_aligned.jpg" alt="Image 7" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      R(-5, -54), R(-15, 119)<br>
      Runtime: 3.28 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/8_aligned.jpg" alt="Image 8" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      R(3, 26), R(3, 121)<br>
      Runtime: 3.19 sec
    </figcaption>
  </figure>
  
</div>



# Bells and Whistles
