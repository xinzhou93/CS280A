---
title: "IMAGE WARPING and MOSAICING"
tags: [project, cs280a]
---

# Overview
 This project builds a image warping and mosaic pipeline using homography transformations. We compute homographies from manually selected point correspondences (A2), warp images using bilinear/Nearest Neighbor interpolation (A3), and stitch images into planar mosaics with weighted averaging (A4). 

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
The recovered homography matrix is
$$
H = 
\begin{bmatrix}
0.000401 & 0.000023 & 0.995941 \\
-0.000108 & 0.000577 & 0.090006 \\
-0.000000 & -0.000000 & 0.000658
\end{bmatrix}
$$

### System of equations
Assume 
$$
H = \begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & 1
\end{bmatrix}
$$
where the lower right corner is a scaling factor set to 1. We also have 8 points for both the original image $M_1$ and the target image $M_2$ respectively. However, to get the $3*3$ homography matrix, we need to transfer each point $[x,y]$ into homogeneous coordinates. For example, If we start with point in $M_1$, the first point is $p_1 = [64.96, 1019.53, 1]^T$ and the target point is $p_1' = [1605.36, 1031.43, 1]^T$.

To derive the system of equations, assume $p = [x, y, 1]^T$ and 
$p' = [x', y', 1]^T$. Then we have the transformation
$$
Hp_1 = wp_1'
$$
where $w$ is a scaling factor and $w \ne 0$.

Expanding the equation we can get
$$
\begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & 1
\end{bmatrix} 
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix} =
\begin{bmatrix}
wx' \\
wy'\\
w
\end{bmatrix}
$$

Then we can get three equations:
$$
\begin{align*}
ax + by + c = wx' \\
dx + ey + f = wy' \\
gx + hy + 1 = w
\end{align*}
$$

Rearrange the equation, we can get
$$
\begin{align*}
ax + by + c = (gx + hy + 1)x' \\
dx + ey + f = (gx + hy + 1)y' \\
ax + by + c - gxx' - hyx' = x'\\
dx + ey + f - gxy' - hyy' = y'
\end{align*}
$$

put the 2 equations into the matrix form, we get
$$
\overbrace{
\begin{bmatrix}
x & y & 1 & 0 & 0 & 0 & -xx' & -yx' & -x' \\
0 & 0 & 0 & x & y & 1 & -xy' & -yy' & -y'
\end{bmatrix}}^{\text{Matrix M}}
\begin{bmatrix}
a \\b \\c \\d \\e \\ f\\g\\h
\end{bmatrix} = 
\begin{bmatrix}
x'\\y'
\end{bmatrix}
$$

From the system of equation above for one point, we can see that each point can contribute 2 rows. If we have 8 points, the left matrix $M$ will expand to a $16 * 8$ matrix. Then we can solve it with least squares: $$\mathbf{h} = (M^T M)^{-1} M^T \mathbf{b}$$

I also tested other images.
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/3_1.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

$$
M_1=
\begin{bmatrix}
791.02 & 2217.72 \\
1243.32 & 1860.64 \\
2762.88 & 1876.51 \\
3155.67 & 2110.60 \\
3084.25 & 452.17 \\
3262.79 & 317.27 \\
3068.38 & 793.38 \\
3064.42 & 971.92
\end{bmatrix}

M_2=
\begin{bmatrix}
6.44 & 2459.74 \\
534.12 & 2003.47 \\
2204.45 & 1852.71 \\
2502.02 & 2007.44 \\
2386.96 & 646.58 \\
2517.89 & 567.23 \\
2414.73 & 940.18 \\
2414.73 & 1083.01
\end{bmatrix}
$$
The recovered homography matrix is
$$
H = 
\begin{bmatrix}
1.426295 & 0.009679 & -1163.549961 \\
0.091272 & 1.063630 & 133.598303 \\
0.000118 & -0.000021 & 1.000000

\end{bmatrix}
$$
## Part 1.3: Warp the Images
From A2, we get the parameters of the homography. Assume $p = [x, y, 1]^T$ and $p' = [x', y', 1]^T$ are points using homogeneous coordinates. Then if homography $H$ is known, we can have $Hp_1 = wp_1'$, which is known as forward warping. Nevertheless, forward warping can leave holes in the output image if several input pixels map to the same output pixel or some output pixels do not receive any input pixel colors.

Instead, we can use inverse warping. For each pixel location $[x', y', 1]^T$ in the output image, we find the corresponding location in the input image by computing:

$$
\begin{bmatrix} u \\ v \\  w \end{bmatrix} = H^{-1} \begin{bmatrix} x' \\  y' \\ 1 \end{bmatrix}
$$

Since we iterate through each pixel of the output image, there is a guarantee that we can find a corresponding value.

Then we can convert the homogeneous coordinates by doing 
$$
\begin{bmatrix} 
x \\ y
\end{bmatrix} = 
\begin{bmatrix} 
u \over w \\
v \over w
\end{bmatrix}
$$

If the updated pixel location is in-between, we need to interpolate pixel values from neighbors using Nearest Neighbor  or Bilinear Interpolation.

### Nearest Neighbor Interpolation
Nearest Neighbor is relatively easy, we just need to round the floating number and take the integer of it. Besides, it is always good to check the valid bounds.
```python
x_in = int(round(x_in))
y_in = int(round(y_in))

# Check if within bounds
if 0 <= x_in < width and 0 <= y_in < height:
warped[y_out, x_out] = img[y_in, x_in]
```

For pixel location that is outside the bound, the pixel color will be black.

### Bilinear Interpolation
Assume $(0,0)$ is at the "top-left" of the first pixel, If the pixel location lands inside a pixel, for bilinear interpolation, we need to find the 4 surrounding integer pixel vertices and interpolate between them as the diagram shown below.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/d.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

![[Pasted image 20251008224616.png]]

Then we can firstly apply 1D linear interpolation
Linear interpolation formula:
$$
\text{lerp}(t, v_0, v_1) = v_0 + t(v_1 - v_0) = (1-t)v_0 + t \cdot v_1
$$

Horizontally (along y=y0), interpolate between top-left and top-right:
$$
u_0 = \text{lerp}(dx, \text{img}[y_0, x_0], \text{img}[y_0, x_1])
$$

Horizontally (along y=y1), interpolate between bottom-left and bottom-right:
$$
u_1 = \text{lerp}(dx, \text{img}[y_1, x_0], \text{img}[y_1, x_1])
$$
where $dx = x - x_0$

Vertically, interpolate between $u_0$ and $u_1$:
$$
\text{result} = \text{lerp}(dy, u_0, u_1)
$$

where $dy = y - y_0$

When testing the both interpolation methods, I found that bilinear interpolation takes longer time since it needs to find four neighbors and interpolate among them. 

### Rectification
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/4_1.png" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Original Image 2 (target)
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/4_b.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Warped - Bilinear
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P3/4_n.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Warped - Nearest Neighbor
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/5_1.png" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Original Image 2 (target)
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/5_b.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Warped - Bilinear
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P3/5_n.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Warped - Nearest Neighbor
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/6_1.png" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Original Image 2 (target)
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/6_b.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Warped - Bilinear
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P3/6_c.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Warped - Nearest Neighbor
    </figcaption>
  </figure>
</div>

## Part 1.4: Blend the Images into a Mosaic
Following A3, we want to blend the images into a single mosaic.
- Similar to A3, we manually select 8 correspondence points for both images and calculate homography using 
`H = computeH(im1_pts,im2_pts)`.
- Then I use the one-shot procedure to determine the final mosaic canvas.
	- For both the original image and the target images, I find the 4 corners and determine the warped corners using `H @ 4_corners`. Then we can get 8 warped corners.
	- For those corners, we can find a bounding box by finding `min_x, min_y, max_x, max_y` and the output mosaic size is `max_y-min_y, max_x-min_x`.
	- We also need to calculate the offset $(tx, ty)$ from the bounding box to shift all coordinates to positive values. We also use this offset in the translation transformation. 
	$$
	\begin{align*}
	t_x &= \min_x(\text{bbox}) \\
	t_y &= \min_y(\text{bbox})
	\end{align*}
	$$
- After that, we need warp each image to the ouput mosaic:
	- For image 1, which is the reference image, we only need to apply a translation for offset since the homography is just an identity matrix.
	$$
	H_T= T\cdot H = T\cdot I = T = 
	\begin{bmatrix}
	1 & 0 & t_x \\ 
	0 & 1 & t_y \\ 
	0 & 0 & 1
	\end{bmatrix}
	$$
	- For image 2, we need to apply both homography and translation.
	$$H_T= T \cdot H$$
	- Similar to A3, we still use the inverse warping technique to avoid holes. For each pixel location $[x', y', 1]^T$ in the output image, we find the corresponding location in the input image by computing:$$\begin{bmatrix} u \\ v \\  w \end{bmatrix} = H_T^{-1} \begin{bmatrix} x' \\  y' \\ 1 \end{bmatrix}$$
	- For identitfying the overlapped area of the two images, we also have a binary variable to keep track if the pixel has a color(255) or it is black(0) since each warped image only occupies small part of the output image. This is helpful to do blending in the next step.
- Finally, we need to blend the overlapped area with weighted averaging. 
	- We normalize the binary variable: black = 0, color = 1.
	- If the pixel location receives color from only one image, then that pixel gets that single color from that image.
	- If the pixel location get colors from both images, then we need to average the both colors in RGB respectively. 
	- If the pixel location get black colors, then it remains black.


<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/7_1.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/7_2.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Result with weighted averaging
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/8_1.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/8_2.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Result with weighted averaging
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/9_1.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/9_2.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Result with weighted averaging
    </figcaption>
  </figure>
</div>

## Part 1.5: Bells & Whistles
In this task, I tried to extend A4 by using cylindrical projection. I firstly attempted manual implementation based on A4 code:
- Select points on the original image and target images.
- Then project images to cylindrical coordinates
- Transform the points to cylindrical coordinates
- Compute homography between projected points
- Finally warp and blend the images as A4 does

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/9_11.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

However, we can see that the images are not stitched correctly since there are black areas between inside the warped images. I choose focal length of 800 pixels but this value may not represent the actual camera length. In addition, I assume the cylindrical coordination implementation is wrong and thus the subsequent homography cannot be computed correctly. 

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/9_21.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Result with Cylindrical mapping
    </figcaption>
  </figure>
</div>

Then I tried to use OpenCV's built in stitcher function `stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)` and the result is pretty good. I will look into my code to figure out what caused the mismatches.
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/9_41.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Cylindrical mapping using OpenCV's built in method
    </figcaption>
  </figure>
</div>

# Part 2: 
## Part B.1: Harris Corner Detection

From the lecture, we know that if we have a window $W$ for the shift $[u,v]$, only the corner can have significant change in all directions and the change can be expressed as 
$$E(u,v) = \sum_{x,y \in W}[I(x+u, y+v) - I(x,y)]^2$$
Using the first-order Taylor approximation, we can simplify the equation:
$$
E(u,v) = \sum_{x,y \in W}
\begin{bmatrix}
u & v
\end{bmatrix}
\begin{bmatrix}
I_x^2 & I_xI_y \\
I_xI_y & I_y^2
\end{bmatrix}
\begin{bmatrix}
u \\ v
\end{bmatrix}
$$

The quadratic approximation further simplifies to
$$
E(u,v) \approx 
\begin{bmatrix}
u & v
\end{bmatrix} M
\begin{bmatrix}
u \\ v
\end{bmatrix}
$$

Where $M$ is a second moment matrix computed from image derivatives:
$$
M = \sum_{x,y \in W}
\begin{bmatrix}
I_x^2 & I_xI_y \\
I_xI_y & I_y^2
\end{bmatrix}
$$

Then, we use the Measure of corner response
$$R = \frac{\text{det} M}{\text{Trace} M + \epsilon}$$
to find points with larger corner response $R> \text{Threshold}$

Ensuring to get corner points rather than clusters of corner regions, I also use `local maxima` with window size `10`to find the peaks.
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/B1_1.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

To distribute points evenly over the image, Adaptive Non-maximal Suppression is applied.


<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/B1_2.png" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

## Part B.2: Feature Descriptor Extraction

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/B2_1.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Cylindrical mapping using OpenCV's built in method
    </figcaption>
  </figure>
</div>

## Part B.3: Feature Matching

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/B3_1.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Cylindrical mapping using OpenCV's built in method
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/B3_2.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Cylindrical mapping using OpenCV's built in method
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/B3_3.png" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Cylindrical mapping using OpenCV's built in method
    </figcaption>
  </figure>
</div>

## Part B.4: RANSAC for Robust Homography

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/B4f_1.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/B4f_2.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P3/B4f_3.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

## Part B.5: Bells & Whistles
