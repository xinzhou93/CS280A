---
title: "IMAGE WARPING and MOSAICING"
tags: [project, cs280a]
---

# Overview
 This project explores fundamental image processing techniques including finite difference operators, derivative of Gaussian filters, image sharpening, hybrid images, and multi-resolution blending using Gaussian and Laplacian pyramids. Through implementing these classic computer vision algorithms, I gained hands-on experience with frequency domain analysis and multi-scale image processing techniques.     

# Part 1: 
## Part 1.1: Shoot the Pictures
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/IMG_1883.jpeg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P3/IMG_1884.jpeg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P3/IMG_1885.jpeg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/IMG_1887.jpeg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P3/IMG_1888.jpeg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/IMG_1889.jpeg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P3/IMG_1890.jpeg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>


## Part 1.2: Recover Homographies
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/2_1.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

Assume 
$$
H = \begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & 1
\end{bmatrix}
$$
where the lower right corner is a scaling factor set to 1. We also have 8 points for both the original image $M_1$ and the target image $M_2$ respectively.

$$
M_1=
\begin{bmatrix}
64.96 & 1019.53 \\
37.19 & 2721.60 \\
517.26 & 1955.86 \\
1108.42 & 1864.61 \\
1747.20 & 1475.79 \\
1957.47 & 1479.76 \\
1965.41 & 1721.78 \\
1743.23 & 1717.81 \\
\end{bmatrix}

M_2=
\begin{bmatrix}
1605.36 & 1031.43 \\
1656.93 & 2554.96 \\
2006.08 & 1868.58 \\
2545.66 & 1793.19 \\
3176.50 & 1392.47 \\
3426.45 & 1376.60 \\
3434.39 & 1642.43 \\
3180.47 & 1638.46 \\
\end{bmatrix}
$$

However, to get the $3*3$ homography matrix, we need to transfer them into homogeneous coordinates. Thus the matrices become
$$
M_1=
\begin{bmatrix}
64.96 & 1019.53 & 1 \\
37.19 & 2721.60 & 1 \\
517.26 & 1955.86 & 1 \\
1108.42 & 1864.61 & 1 \\
1747.20 & 1475.79 & 1 \\
1957.47 & 1479.76 & 1 \\
1965.41 & 1721.78 & 1 \\
1743.23 & 1717.81 & 1 \\
\end{bmatrix}

M_2=
\begin{bmatrix}
1605.36 & 1031.43 & 1 \\
1656.93 & 2554.96 & 1 \\
2006.08 & 1868.58 & 1 \\
2545.66 & 1793.19 & 1 \\
3176.50 & 1392.47 & 1 \\
3426.45 & 1376.60 & 1 \\
3434.39 & 1642.43 & 1 \\
3180.47 & 1638.46 & 1 \\
\end{bmatrix}
$$
If we start with point in $M_1$, the first point is $p_1 = [64.96, 1019.53, 1]^T$ and the target point is $p_1' = [1605.36, 1031.43, 1]^T$ and we want to solve
$$
Hp_1 = p_1'
$$

Expanding the equation we can get
$$
\begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & 1
\end{bmatrix} 
\begin{bmatrix}
64.96 \\
1019.53 \\
1
\end{bmatrix} =
\begin{bmatrix}
1605.36 \\
1031.43 \\
1
\end{bmatrix}
$$

We want to solve for $M_1H = M2$

$$
H = 
\begin{bmatrix}
0.000401 & 0.000023 & 0.995941 \\
-0.000108 & 0.000577 & 0.090006 \\
-0.000000 & -0.000000 & 0.000658
\end{bmatrix}
$$


## Part 1.3: Warp the Images
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/IMG_1895.jpeg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Original Image 1
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P3/IMG_1896.jpeg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Original Image 2 (target)
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/3_1.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Result - Bilinear
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P3/3_2.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Result - Nearest Neighbor
    </figcaption>
  </figure>
</div>

## Part 1.4: Blend the Images into a Mosaic

## Part 1.5: Bells & Whistles


# Part 2: 
