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
      B(-41, -17), R(48, 5)<br>
      Runtime: 17 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/2_aligned.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      B(-49, -23), R(58, 17)<br>
      Runtime: 16 sec
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/3_aligned.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Best Shift:<br>
      B(-25, -4), R(33, -8)<br>
      Runtime: 15 sec
    </figcaption>
  </figure>
</div>



# Bells and Whistles
