---
title: "IMAGE WARPING and MOSAICING"
tags: [project, cs280a]
---

# Overview
 This project builds a image warping and mosaic pipeline using homography transformations. We compute homographies from manually selected point correspondences (A2), warp images using bilinear/Nearest Neighbor interpolation (A3), and stitch images into planar mosaics with weighted averaging (A4). 

In the second part, the project focuses on automatic feature based stitching. We first implement Harris corners detection with ANMS for spatial distributed features. Then the pipeline extracts feature descriptors with bias/gain normalization. In B3, we match features using the descriptors and Lowe's test. Finally the system estimates the homography using RANSAC. The final mosaic image can be obtained using weighted averaging blending again.

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

# Part 2: Fit a Neural Radiance Field from Multi-view Images

# Part 2.6: Training with Your Own Data

# Bells and Whistles
