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
    <img src="/P2/P7_2.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    The hybrid Image of the samples
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_3.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    The hybrid Image of the samples
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_4.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
	    The hybrid Image of the samples
    </figcaption>
  </figure>
</div>

## Part 2.3: Gaussian and Laplacian Stacks
For the stacks implementation, I built multi-scale image representations without downsampling to analyze how image content varies across different frequency bands. My approach involved creating Gaussian stacks by iteratively applying Gaussian blur with increasing sigma values, effectively removing progressively higher frequencies at each level. The corresponding Laplacian stack was computed by taking differences between consecutive Gaussian stack levels, isolating specific frequency bands at each scale.

The implementation required careful handling of the mathematical relationship `L[i] = G[i] - G[i+1]` to ensure perfect reconstruction when summing all Laplacian levels plus the final Gaussian residual. I validated this by verifying that reconstructed images matched the originals within numerical precision. Applying these stacks to hybrid images revealed fascinating insights about how the hybrid effect manifests at different scales - typically, one source image dominates at finer scales while the other becomes more apparent at coarser scales, providing a quantitative explanation for why hybrid images work.

**Results:**
[Add your stack results here - Gaussian stacks, Laplacian stacks, reconstruction verification, hybrid image stack analysis]

---
## Part 2.4: Multi-resolution Blending
The multi-resolution blending section implemented the classic Burt and Adelson 1983 algorithm for seamlessly combining images using Laplacian pyramid decomposition. My approach involved building Laplacian pyramids for both input images through iterative Gaussian smoothing and downsampling, creating a multi-scale representation where each level captures different frequency bands. Simultaneously, I constructed a Gaussian pyramid for the blending mask, allowing the transition boundary to be represented at multiple scales.

The blending process occurred at each pyramid level using the formula `Blended[i] = Mask[i] × Image1[i] + (1-Mask[i]) × Image2[i]`, where the mask values smoothly interpolate between the two source images. The final result was obtained by reconstructing the image from the blended Laplacian pyramid through a series of upsample and add operations. This multi-scale approach eliminates the harsh transitions that would occur with simple alpha blending while preserving fine details at boundaries.

I implemented both a standard grayscale version following the original paper (`part24_multiresolution_blending_grayscale.py`) and an advanced color version with extensive mask options (`part24_advanced_blending.py`). The grayscale version converts all inputs to luminance values before processing, maintaining the classic single-channel approach, while the color version processes each RGB channel independently to preserve color information throughout the blending process.

For the bells and whistles component, I created an extensive library of blending masks beyond simple vertical seams, including horizontal transitions, circular masks with controllable feathering, various gradient patterns (radial, diagonal), and mathematically-defined custom shapes like hearts and stars. I also implemented an interactive mask creation tool allowing users to define arbitrary blending boundaries by clicking points to create polygonal regions. This comprehensive mask library demonstrates how different boundary shapes can create vastly different artistic effects while using the same underlying multi-resolution blending algorithm.

**Results:**

[Add your blending results here - mask library visualization, blending results for different masks, comparison between grayscale and color approaches]

---

  

## Technical Implementation and Learnings
Throughout this project, I encountered several technical challenges that deepened my understanding of image processing fundamentals. The pyramid construction required careful attention to dimension handling, particularly ensuring that upsampling operations correctly reconstructed the expected sizes when dealing with odd and even dimensions. I solved broadcasting issues when applying 2D masks to 3D color images by expanding mask dimensions appropriately.

The interactive components taught me about building user-friendly tools for computer vision tasks, as precise alignment and custom mask creation significantly impact final results. I learned that successful hybrid images require not just technical implementation but also artistic intuition about which features should be emphasized at different viewing distances.

The frequency domain analysis revealed how different processing techniques affect various scales of image content, providing insight into why multi-resolution approaches often outperform single-scale methods for complex image processing tasks like seamless blending and hybrid image creation.