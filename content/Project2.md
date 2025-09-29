---
title: "Fun with Filters and Frequencies"
tags: [project, cs280a]
---

# Overview
 This project explores fundamental image processing techniques including finite difference operators, derivative of Gaussian filters, image sharpening, hybrid images, and multi-resolution blending using Gaussian and Laplacian pyramids. Through implementing these classic computer vision algorithms, I gained hands-on experience with frequency domain analysis and multi-scale image processing techniques.     

# Part 1: Fun with Filters
## Part 1.1: Convolutions From Scratch
For this section, I implemented basic edge detection using finite difference operators to compute image gradients. My approach involved applying simple convolution kernels `[1, -1]` for horizontal differences and `[[1], [-1]]` for vertical differences to detect edges by finding regions of rapid intensity change. I computed the gradient magnitude using the formula `√(dx² + dy²)` to combine both horizontal and vertical edge information into a single edge strength map. To create clean edge maps, I experimented with different threshold values to binarize the gradient magnitude, finding that higher thresholds produced cleaner but potentially incomplete edge detection while lower thresholds captured more detail but introduced noise.

```python
# Four nested loops  
for i in range(img_height):  
    for j in range(img_width): 
        for ki in range(kernel_height): 
            for kj in range(kernel_width): 
                output[i, j] += padded_image[i + ki, j + kj] * kernel_2d[ki, kj]
```

```python
# two nested loop
for i in range(img_height):
    for j in range(img_width):  
        if direction == 'x':  
            output[i, j] = np.sum(padded_image[i + pad_size, j:j+filter_size] * filter_1d)  
        else:  
            output[i, j] = np.sum(padded_image[i:i+filter_size, j + pad_size] * filter_1d)
```
The implementation in `part11_finite_difference.py` includes functions for computing directional gradients, combining them into magnitude maps, and applying thresholds for edge detection. I found that the raw finite difference approach, while simple and fast, was quite sensitive to image noise, which motivated the need for the more sophisticated approach in the next section.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P1_smaller.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      Original Image
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/P1_smaller_box_filtered.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      9*9 box filter
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P1_smaller_dx_edges.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Dx
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P2/P1_smaller_dy_edges.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Dy
    </figcaption>
  </figure>
</div>

## Part 1.2: Finite Difference Operator

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P2_x.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Dx
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P2/P2_y.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Dy
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P2_p85.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Top 15%<br>
	    Thresh = 0.108
    </figcaption>
  </figure>
  
  <figure style="margin: 0;">
    <img src="/P2/P2_p90.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Top 10%<br>
	    Thresh = 0.161
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/P2_p92.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Top 8%<br>
	    Thresh = 0.199
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P2_p95.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Top 5%<br>
	    Thresh = 0.315
    </figcaption>
  </figure>
  
  <figure style="margin: 0;">
    <img src="/P2/P2_p97.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Top 3%<br>
	    Thresh = 0.424
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/P2_p99.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Top 1%<br>
	    Thresh = 0.612
    </figcaption>
  </figure>
</div>

## Part 1.3: Derivative of Gaussian (DoG) Filter

Building on the limitations observed in Part 1.1, I implemented Derivative of Gaussian filters to achieve more robust edge detection by combining smoothing and differentiation in a single operation. My approach involved first creating 2D Gaussian kernels with controllable sigma values to determine the amount of smoothing, then applying these to blur the image before computing finite differences. This two-step process significantly reduced noise sensitivity while preserving important edge information.

I also demonstrated the mathematical principle that convolution is associative by showing that blurring an image first and then applying difference operators produces identical results to convolving the image with pre-computed derivative of Gaussian kernels. This insight led me to implement both approaches in `part12_derivative_gaussian.py`, allowing for either sequential processing or direct application of DoG filters. The Gaussian pre-smoothing proved essential for practical edge detection, as it eliminated high-frequency noise that would otherwise dominate the gradient computation while preserving the structural edges that define object boundaries.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P3.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Original
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P2/P3_smooth.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    DoG filter applied
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P3_filters.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    DoG filters
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P3_x_twostep.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Dx -- two step DoG
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P2/P3_x_dog.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Dx -- one step DoG
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P3_y_twostep.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Dy -- two step DoG
    </figcaption>
  </figure>
  <figure style="margin: 0;">
    <img src="/P2/P3_y_dog.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Dy -- one step DoG
    </figcaption>
  </figure>
</div>


## Part1.4: Bells and Whistles

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P4.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    Image Gradient Orientations with HSV color space
    </figcaption>
  </figure>
</div>

# Part 2: Fun with Frequencies
## Part 2.1: Image "Sharpening"

In this section, I implemented the unsharp masking technique to enhance image sharpness by amplifying high-frequency details. My approach was based on the principle that subtracting a blurred version of an image from itself isolates the high-frequency components, which can then be added back to the original image with increased amplitude. I used the formula `sharpened = original + α × (original - blurred)` where α controls the strength of the sharpening effect.

The implementation involved experimenting with different Gaussian blur sigma values to control which frequencies are considered "high frequency" and testing various alpha multipliers to achieve the desired level of enhancement without introducing artifacts. I discovered that moderate alpha values (typically 0.5-2.0) provided pleasing enhancement while higher values could create unrealistic over-sharpening or ringing artifacts around edges. To validate the technique's effectiveness, I also tested it on intentionally blurred images to demonstrate that sharpening could partially recover lost detail, though it cannot fully restore information that was eliminated by the original blurring process.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P5_o.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Original
    </figcaption>
  </figure>
  
  <figure style="margin: 0;">
    <img src="/P2/P5_blur.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Blurred
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/P5_s.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Sharpened
    </figcaption>
  </figure>
</div>


<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P1_smaller.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Original
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/P6_o.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Blurred
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/P6_s.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Sharpened
    </figcaption>
  </figure>
</div>
  
## Part 2.2: Hybrid Images

The hybrid images section involved creating images that appear different depending on viewing distance by combining the low-frequency content of one image with the high-frequency content of another. My approach required careful frequency separation: I extracted low frequencies using Gaussian blur with a large sigma value and obtained high frequencies by subtracting a Gaussian-blurred version from the original image using a smaller sigma. The key challenge was finding the optimal sigma values for each image to create a convincing hybrid effect.

Image alignment proved crucial for successful hybrid images, so I implemented an interactive alignment system using matplotlib's ginput functionality, allowing precise manual registration of corresponding features between the two source images. This involved geometric transformations including translation, rotation, and scaling to ensure that key features like eyes, mouth, or other structural elements were properly aligned before frequency mixing.

For the bells and whistles component, I extended the basic grayscale approach to explore color hybrid images by testing four different combinations: standard grayscale, high-frequency component in color, low-frequency component in color, and both components in color. This exploration revealed that different color strategies can enhance or diminish the hybrid effect, with some combinations making the transition between near and far viewing more dramatic. I implemented this in `part22_color_hybrid_bells_whistles.py` with comprehensive analysis of which color approaches work best for different image pairs.


<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_5.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_3.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_2.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    The hybrid Image of the samples
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_4.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

## Part 2.2B: Bells and Whistles

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_6.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_7.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

## Part 2.3: Gaussian and Laplacian Stacks
Gaussian stack is a sequence of progressively blurred images at the same resolution as the original. Different from the pyramid structure, there is no need to subsample the image in each level. However, to effectively blur the images, we do need to apply the Gaussian kernels with increasing $\sigma$ values that have wider spread outs.

In the kernel implementation, I double the $\sigma$ in each level ($\sigma, 2\sigma, 4\sigma,..$) and use the rule of thumb `kernel size = 6 * sigma + 1` from the lecture and make sure the size is odd. Then, I use the customized function `create_gaussian_kernel(kernel_size, current_sigma)` from Part 1 to create the new wider Gaussian kernel.

The total level of the stack is $5$ and I start the $\sigma$ at $2$. In a for loop, I create a Gaussian kernel for the current $\sigma$ and use convolution from Part 1 to get the target blurred image. The result in each level is saved into a list `gaussian_stack` for Laplacian stack implementation.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P9_1.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Apple Gaussian Stack
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P9_3.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Orange Gaussian Stack
    </figcaption>
  </figure>
</div>

A Laplacian stack is derived from the Gaussian stack by subtracting the blurred image at the coarser level ($G_{i+1}$) from the current image ($G_i$), which gives the band pass filtered image at the specific level.

$$L_i = G_i - G_{i+1}$$

In the implementation, I create a function that passes the `Gaussian_stack` list as a parameter and use a for loop to store the differences between the current and next image. The result `Laplacian_stack` list has 4 band-pass filtered images and the most blurred image at the last Gaussian level since we need all of them to retrieve the original image by collapsing the stack, mentioned in the lecture.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P9_2.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Apple Laplacian Stack
    </figcaption>
  </figure>
</div>


<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P9_4.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Orange Laplacian Stack
    </figcaption>
  </figure>
</div>

## Part 2.4: Multi-resolution Blending
I followed the steps of image blending with the Laplacian stack:
- Build Laplacian stack for both images
	- I directly use Part 2.3 to get the Laplacian stacks for both images.
- Build Gaussian stack for a mask.
	- I directly use Gaussian stack from Part 2.3 for the masks.
- Build a combined Laplacian stack $L$.
	- The blending function is applied from the lecture: $l_k = l_k^A * m_k + l_i^B * (1-m_k)$
- Collapse $L$ to obtain the blended image.

The image below shows the image blending at each Laplacian level.


<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P11_1.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Orange Laplacian Stack
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P11_2.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Orange Laplacian Stack
    </figcaption>
  </figure>
</div>


##  Part 2.4B: Bells and Whistles

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/blend_horizontal.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Original
    </figcaption>
  </figure>
  
  <figure style="margin: 0;">
    <img src="/P2/blend_vertical.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Blurred
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/blend_wave.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Sharpened
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P10_1.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P10_2.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P10_3.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

## Technical Implementation and Learnings
Throughout this project, I encountered several technical challenges that deepened my understanding of image processing fundamentals. The pyramid construction required careful attention to dimension handling, particularly ensuring that upsampling operations correctly reconstructed the expected sizes when dealing with odd and even dimensions. I solved broadcasting issues when applying 2D masks to 3D color images by expanding mask dimensions appropriately.

The interactive components taught me about building user-friendly tools for computer vision tasks, as precise alignment and custom mask creation significantly impact final results. I learned that successful hybrid images require not just technical implementation but also artistic intuition about which features should be emphasized at different viewing distances.

The frequency domain analysis revealed how different processing techniques affect various scales of image content, providing insight into why multi-resolution approaches often outperform single-scale methods for complex image processing tasks like seamless blending and hybrid image creation.