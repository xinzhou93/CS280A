---
title: "Fun with Filters and Frequencies"
tags: [project, cs280a]
---

# Overview
 This project explores fundamental image processing techniques including finite difference operators, derivative of Gaussian filters, image sharpening, hybrid images, and multi-resolution blending using Gaussian and Laplacian pyramids. Through implementing these classic computer vision algorithms, I gained hands-on experience with frequency domain analysis and multi-scale image processing techniques.     

# Part 1: Fun with Filters
## Part 1.1: Convolutions From Scratch
Let $F$ be the image, $H$ be the kernel($2k+1 \times 2k+1$) and $G$ be the output image. A convolution operation is a cross-correlation where the filter is flipped both horizontally and vertically before being applied to the image:

$$
G[i,j] = \sum_{u=-k}^{k} \sum_{v=-k}^{k} H[u,v] \, F[i-u, \, j-v]
$$

It is written:

$$
G = H \star F
$$

For this section, I implemented convolution using cross-correlation instead since I do not need to worry about the flipping operations.

$$
G[i,j] = \sum_{u = -k}^{k} \sum_{v = -k}^{k} H[u,v] \, F[i+u, j+v]$$

To compute the output pixel at location $(i, j)$:
1. Place the kernel $H$ centered at that location.
2. Multiply each kernel value $H[u, v]$ with the corresponding image pixel $F[i+u, j+v]$.
3. Add up all those products.

To get convolution work for each pixel of the image, I firstly implemented zero padding by using the formula `pad size = (kernel_height/width - 1) // 2` and add the pad to each side of the image using `np.pad()`.

 For the four-loop convolution, the first two loops iterate through each position $[i,j]$ in the output image. For each output position, the inner two loops iterate through all kernel positions $[ki,kj]$, accumulating the sum of element-wise products between the kernel and the corresponding image patch `padded_image[i+ki, j+kj]`. I was confused about finding the correct neighboring pixels but the zero padding does the shift for us. For example, assume the pad size is 1, after padding, the starting pixel of the padded image is $(0,0)$ but in the original image, that position is mapped to $(1,1)$ and it is easier for us to find the top left, top right ..... pixels.

```python
padded_image = add_zero_padding(image, kernel_2d)
# Four nested loops  
output = np.zeros((img_height, img_width))

for i in range(img_height):  
    for j in range(img_width): 
        for ki in range(kernel_height): 
            for kj in range(kernel_width): 
                output[i, j] += padded_image[i + ki, j + kj] * kernel_2d[ki, kj]
```
  
  Instead of applying a 2D filter in one step, we can apply two 1D filters sequentially:
  - Step 1: Apply 1D filter horizontally (along rows)
  - Step 2: Apply 1D filter vertically (along columns)
 
```python
# Get dimensions  
img_height, img_width = image.shape  
filter_size = len(filter_1d)  
  
# Use padding function  
padded_image = add_zero_padding(image, filter_1d)  
pad_size = (filter_size - 1) // 2

# two nested loop
for i in range(img_height):
    for j in range(img_width):  
        if direction == 'x':  
            output[i, j] = np.sum(padded_image[i + pad_size, j:j+filter_size] * filter_1d)  
        else:  
            output[i, j] = np.sum(padded_image[i:i+filter_size, j + pad_size] * filter_1d)
```


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

In this section, I implemented the unsharp masking technique to enhance image sharpness by amplifying high-frequency details. My approach was based on the principle that subtracting a blurred version of an image from itself isolates the high-frequency components, which can then be added back to the original image with increased amplitude. 

$$\text{High Frequency = Original - Low Frequency}$$

Then I used the formula 

$$\text{sharpened = original + } \alpha \times \text{(original - blurred)}$$

where $\alpha$ controls the strength of the sharpening effect.

The implementation has 3 steps:
- Get low frequency image:
	- I create a function that passes an image and an abstract $\sigma$ as the parameters. Then through the rule of thumb from the lecture `kernel size = 6 * sigma + 1` and `create_gaussian_kernel(kernel_size1, sigma1)` from Part 1, I can get a Gaussian kernel and convolve it with the image to get the target low frequency version.
- Get high frequency image.
	- The high frequency image can be obtained by subtracting a Gaussian-blurred version from the original image: $\text{High Frequency = Original - Low Frequency}$.
- Sharpen the image.
	- Add amplified high frequencies back to original using $\text{sharpened = original + } \alpha \times \text{(original - blurred)}$

The images below show that the technique of image sharpening can effectively recover the original image from a blurred version with the specific $\alpha$ value.

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
    Blurred with sigma = 2.0
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/P5_s.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Sharpened with alpha = 1.5
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
    Blurred with sigma = 2.0
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/P6_s.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    Sharpened with alpha = 1.5
    </figcaption>
  </figure>
</div>

The implementation involved experimenting with different alpha multipliers to achieve the desired level of enhancement without introducing artifacts. I found that moderate $\alpha$ values (typically $0.5-2.0$) provided improvement while higher values could create artifacts around edges. For example, with $\alpha = 5$, the bumpy wall in the background stands out and the dark circles around my eyes become obvious, which made me really sad.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P6_com.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>


## Part 2.2: Hybrid Images
The hybrid images section intends to guide us how to create a famous special effect from "Gala Contemplating the Mediterranean Sea, which at 30 meters becomes the portrait of Abraham Lincoln”, 1976". In the lecture, we knew that from a distance a portrait of Lincoln appeared whereas Gala showed up when looking at it closely.

This part involves creating images exactly the same way that appear different depending on viewing distance by combining the low-frequency content of one image with the high-frequency content of another. 

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_3.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

Take the images of me and a tiger for example, I want myself to be visible from far away and the tiger to show up close. The steps are shown as follows:
- Get low frequency image from me.
	- I create a function that passes my photo and an abstract $\sigma$ as the parameters. Then through the rule of thumb from the lecture `kernel size = 6 * sigma + 1` and `create_gaussian_kernel(kernel_size1, sigma1)` from Part 1, I can get a Gaussian kernel and convolve it with the image to get the target low frequency version.
- Get high frequency image from the tiger.
	- Using the same process to get a low frequency tiger, the high frequency image can be obtained by subtracting a Gaussian-blurred version from the original image: $\text{High Frequency = Original - Low Frequency}$.
- Combine two images.
	- for the result hybrid image, we use `hybrid = low_frequencies + high_frequencies`.

I used the starter code, which includes an interactive alignment feature for the two images.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P12_1.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

The image above demonstrates the relationship between each image and the corresponding Fourier transform visualization. I use the default values for the two images. (Me: $\sigma = 10$, Tiger: $\sigma = 5$). In the input images, due to alignment, there are sharp black boundaries. As a result, in the FFT visualizations, we can see bright lines both horizontally and vertically and noisy white dots scattered in the images. The low-pass filter (column 3) preserves the center frequencies and wipes out the high frequency noise, creating a blurry version of me. The bright lines in FFT are also blurred.(I do not understand why the background becomes black). However, the high-pass filter (column 4) removes center frequencies but keeps edge information from the tiger. The FFT of it is really bright, indicating the high frequency information kept in the image. When combined in the hybrid image (column 5), the center frequencies from me and peripheral frequencies from the tiger coexist in the same image. 

The key challenge was finding the optimal sigma values for each image to create a smooth hybrid effect. I tested several combinations. From my perspective, the bottom right image is the worse than others since the low frequency image is too blurry and the hybrid image is dominant by the tiger whatever the distance is. I prefer the default combo (top left) because I can clearly see me when looking at it a bit far away.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P12_2.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      sigma1 = 10 <br>
      sigma2 = 5
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/P12_8_4.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      sigma1 = 8 <br>
      sigma2 = 4
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P12_12_6.jpg" alt="Image 1" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      sigma1 = 12 <br>
      sigma2 = 6
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/P12_15_3.jpg" alt="Image 2" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
      sigma1 = 15 <br>
      sigma2 = 3
    </figcaption>
  </figure>
</div>

Here are some other hybrid images I tested.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_1.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_44.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P7_2.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    </figcaption>
  </figure>
</div>

## Part 2.2B: Bells and Whistles

For the bells and whistles component, I extended the basic grayscale approach to explore color hybrid images. In my opinion, the hybrid image works better to use color for both the high-frequency and the low-frequency component since colors bring us another layer of information, such as texture and areas of highlight.

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
The Laplacian level 0 shows the clear edges of the apple and orange that represent the highest frequencies. There is a clear seam in the mask, indicating the window size of the mask is relatively short since high frequency details do not need a wide transition. In contrast, level 3 demonstrates the blurred images of apple, orange and mask because low frequency needs more window size to create a natural blending effect.

In the visualization process, I had difficulty imitating the colors of Laplacian stack in the paper since the values of Laplacian stack can be negative. To show it in RGB, I just clamp the value to $[0,1]$, displaying them in a grayscale style.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P11_1.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    The process of Laplacian Blending at each level
    </figcaption>
  </figure>
</div>

Using multiresolution blending, even there is a mask with a hard boundary in the middle, the result can still create a nice transition effect when blending two images.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/P11_2.png" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    The result of Image blending with Laplacian stack
    </figcaption>
  </figure>
</div>

I also tried some other masks including irregular ones:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: center;">
  <figure style="margin: 0;">
    <img src="/P2/blend_horizontal.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    mask - horizontal
    </figcaption>
  </figure>
  
  <figure style="margin: 0;">
    <img src="/P2/blend_vertical.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    mask - vertical
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/blend_wave.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    mask - vertical wave
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/blend_heart.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    mask - heart
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/blend_circular.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    mask - circle
    </figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="/P2/blend_star.jpg" alt="Image 3" style="width: 100%; height: auto; display: block;" />
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 6px; line-height: 1.4;">
    mask - star
    </figcaption>
  </figure>
  
</div>

Here are some blended images I found interesting.
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

##  Part 2.4B: Bells and Whistles
