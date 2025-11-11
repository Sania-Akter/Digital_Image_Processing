# -*- coding: utf-8 -*-
"""
Digital Image Processing Assignment
-----------------------------------
DCT, DFT, and DWT Implementation in Python

This program:
1. Loads a grayscale image.
2. Applies DCT, DFT, and DWT.
3. Displays frequency-domain and wavelet-domain results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt  # For Discrete Wavelet Transform (install using: pip install PyWavelets)

# ========== Function to normalize image ==========
def normalize_img(x):
    x = x.astype(np.float64)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-12)

# ========== Function to compute magnitude (for visualization) ==========
def magnitude_spectrum(x):
    return np.log(np.abs(x) + 1.0)

# ========== Main Function ==========
def main():
    # ---- Step 1: Load image ----
    img_path = r"E:\dipimage\sapla.jpg"  
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Image not found! Check the path.")
        return
    
    img32 = img.astype(np.float32)

    # ---- Step 2: Apply DCT ----
    dct = cv2.dct(img32)
    dct_vis = normalize_img(magnitude_spectrum(dct))

    # ---- Step 3: Apply DFT ----
    F = np.fft.fft2(img32)
    F_shift = np.fft.fftshift(F)
    dft_vis = normalize_img(magnitude_spectrum(F_shift))

    # ---- Step 4: Apply DWT (using Haar wavelet) ----
    coeffs2 = pywt.dwt2(img32 / 255.0, 'haar')
    LL, (LH, HL, HH) = coeffs2
    LL_v = normalize_img(LL)
    LH_v = normalize_img(np.abs(LH))
    HL_v = normalize_img(np.abs(HL))
    HH_v = normalize_img(np.abs(HH))

    # ---- Step 5: Display all results ----
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.title("DCT")
    plt.imshow(dct_vis, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.title("DFT")
    plt.imshow(dft_vis, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.title("DWT - LL (Approximation)")
    plt.imshow(LL_v, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.title("DWT - LH (Vertical Details)")
    plt.imshow(LH_v, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.title("DWT - HL (Horizontal Details)")
    plt.imshow(HL_v, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.title("DWT - HH (Diagonal Details)")
    plt.imshow(HH_v, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# ========== Run the code ==========
if __name__ == "__main__":
    main()
