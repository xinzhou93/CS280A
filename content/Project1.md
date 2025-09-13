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

### Gallery of required images
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P1/9_aligned.jpg" alt="Image 9" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      G(4, 25), R(-5, 59)<br>
      Runtime: 3.1 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/10_aligned.jpg" alt="Image 10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      G(24, 49), R(42, 104)<br>
      Runtime: 3.3 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/11_aligned.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      G(16, 61), R(13, 124)<br>
      Runtime: 3.11 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/12_aligned.jpg" alt="Image 12" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      G(17, 41), R(23, 90)<br>
      Runtime: 3.25 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/13_aligned.jpg" alt="Image 13" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      G(21, 38), R(35, 77)<br>
      Runtime: 3.22 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/14_aligned.jpg" alt="Image 14" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      R(-2, -3), R(-9, 77)<br>
      Runtime: 3.18 sec
    </figcaption>
  </figure>
  
  <figure style="margin: 0;">
    <img src="/P1/15_aligned.jpg" alt="Image 15" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      R(-13, 41), R(-28, 93)<br>
      Runtime: 3.3 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/16_aligned.jpg" alt="Image 16" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      R(9, 85), R(-12, 158)<br>
      Runtime: 3.17 sec
    </span><br>
      <span style="color: red; font-weight: bold;">Not Perfectly Aligned</span>
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/17_aligned.jpg" alt="Image 17" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      R(-1, 81), R(-3, 158)<br>
      Runtime: 3.3 sec
    </span><br>
      <span style="color: red; font-weight: bold;">Not Perfectly Aligned</span>
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/18_aligned.jpg" alt="Image 18" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      R(-7, 50), R(-25, 97)<br>
      Runtime: 3.35 sec
    </figcaption>
  </figure>
  
  <figure style="margin: 0;">
    <img src="/P1/19_aligned.jpg" alt="Image 19" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      R(12, 55), R(9, 112)<br>
      Runtime: 3.12 sec
    </figcaption>
  </figure>
</div>

### Gallery with other images
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
