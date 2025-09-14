---
title: "Colorizing the Prokudin-Gorskii photo collection"
tags: [project, cs280a]
---

# Overview
This project focuses on colorizing the photo collection made by Prokudin-Gorskii, a photographer travelling across Russian and took color photographs of everything he saw. The collections include portraits, scenery, architecture, etc, and each of them has three exposures of the scene onto a glass plate using a red, a green and a blue filter. Although the three images are black and white, when they overlay, a color photograph will magically appear. 

The goal of this project is to use 






# Methodology
The program takes a single plate image as input and should divide it into three equal parts. Using the channel $B$ as the basis, the program should output a single color image by aligning the Channel $G$ and $R$ to the base $B$ and stacking them together.  We need to handle two challenges:
- How to get the best alignment displacement for each glass plate image?
- How to get the displacement efficiently for high-resolution glass plate scans?

## $L2$ Norm Scoring Metric
To determine the best displacement, we need a metric function to quantify the score of the alignment. In this project, I used $L2$ norm (Euclidean Distance) to calculate the sum of differences, which measures the pixels-wise difference between the two overlapping images. 

$$
L2(I_{ref}, I_{Target}) = \sum (I_{ref}(x,y) - I_{Target}(x,y))^2
$$

The smaller $L2$ value indicates the better alignment and we want to find the smallest value in a search range of possible displacements.

## Exhaustive Search with Predefined Margin
Before performing the alignment search, the algorithm crops both the reference and target images by removing a **fixed** margin on each side. It becomes more convenient and robust to compute metrics on interior pixels only.  The margin is ensured to not exceed half of the minimum image dimension, preventing over-cropping.

```python
if not (0 <= self.margin < h_min/2 and 0 <= self.margin < w_min/2):  
    raise ValueError("Margin is too large for the image")
```

The exhaustive search explores all possible displacement vectors $(dx, dy)$ within a defined search region $[-15, 15]$. For each candidate displacement $(dx, dy)$, the algorithm computes the overlapping region between the images using the boundary constraints:

```python
x_lower_overlap = max(0, dx)  
x_upper_overlap = min(self.img_ref.shape[1], img_other.shape[1]+dx)  
y_lower_overlap = max(0, dy)  
y_upper_overlap = min(self.img_ref.shape[0], img_other.shape[0]+dy)
```

Then, the program only computes the $L2$ norm for the valid overlapping region and take the displacement $(dx, dy)$ with the smallest score.  

Although this method is simple to implement, it becomes expensive and slow when handling high-resolution glass plate scans.

## Image Pyramid Optimization
The image pyramid uses multi-scale images from coarse to fine resolutions to approach to find the best displacement for alignment. 

The pyramid uses a simple $2 \times 2$ box filter for downsampling. At the beginning, the program used nested for loops to extract $2 \times 2$ four pixels sequentially and average them to reduce resolution. Later on, it was replaced by the vectorized method which extracts even and odd rows and columns directly.

The pyramid employs a recursive structure:
- Base Case:
	- When either image dimension goes under $100$ pixels, the program performs **exhaustive search with the predefined margin** to get the optimal displacement at this coarse level. Since the image size is really small, the exhaustive research can be really fast.
- Recursive Case:
	- The program downsamples both reference and target images using the $2 \times 2$ box filter and recursively calls alignment on downsampled images. In the meantime, it receives the coarse displacement $(dx_c, dy_c)$ from lower resolution. To match current finer resolution, the algorithm scales the coarse displacement up by 2 and performs small window search with $radius = 2$ pixels centered at the scaled displacement $dx_u = dx_c × 2, dy_u = dy_c × 2$. This offsets errors from the downsampling process.
	- 
Using this method can efficiently process high-resolution images because most searching happens at the coarse level.


<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
  <img src="/P1/j1_align.jpg" alt="img1" />
  <img src="/P1/j2_align.jpg" alt="img2" />
  <img src="/P1/j3_align.jpg" alt="img3" />
</div>

## Image Pyramid with Predefined borders

### Gallery of required images
<div style="background-color: #222; padding: 20px; border-radius: 8px;">
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
</div>

### Gallery with other images
<div style="background-color: #222; padding: 20px; border-radius: 8px;">
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
</div>



# Bells and Whistles

## Auto-Cropping
<div style="background-color: #222; padding: 10px; border-radius: 8px;">
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P1/10_aligned.jpg" alt="Image 10" style="width: 100%; height: auto; display: block;" />
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/10_intensity_borders.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
  </figure>

</div>
</div>

## Auto Contrast
<div style="background-color: #222; padding: 10px; border-radius: 8px;">
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P1/10_intensity_borders.jpg" alt="Image 10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Auto-Cropping applied
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/10_rescale_contrast.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
      <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Auto-Contrast applied
    </figcaption>
  </figure>

</div>
</div>

## Auto White Balance

<div style="background-color: #222; padding: 10px; border-radius: 8px;">
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P1/10_intensity_borders.jpg" alt="Image 10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Auto-Cropping applied
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/10_max_white_wb.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
      <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Auto White Balance applied
    </figcaption>
  </figure>

</div>
</div>

## Color Mapping

<div style="background-color: #222; padding: 10px; border-radius: 8px;">
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P1/10_intensity_borders.jpg" alt="Image 10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Auto-Cropping applied
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/10_matrix_transform_colormap.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
      <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Color mapping applied
    </figcaption>
  </figure>

</div>
</div>

## Edge-based Alignment

<div style="background-color: #222; padding: 10px; border-radius: 8px;">
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P1/10_intensity_borders.jpg" alt="Image 10" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Auto-Cropping applied
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P1/10_sobel_edge.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
      <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Sobel Detector applied
    </figcaption>
  </figure>

</div>
</div>