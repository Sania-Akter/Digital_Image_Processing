import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --------- Contrast Adjustment ---------
def adjust_contrast(img, contrast_type='normal'):
    if contrast_type == 'low':
        alpha, beta = 0.5, 0    # low contrast
    elif contrast_type == 'high':
        alpha, beta = 2.0, 0    # high contrast
    else:
        alpha, beta = 1.0, 0    # normal contrast
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# --------- DFT and Inverse DFT ---------
def compute_fft(img):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.log(np.abs(dft_shift) + 1)
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(dft_shift)))
    return dft_shift, magnitude, img_back

# --------- Create Filter Masks ---------
def low_pass_mask(shape, radius):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
    return mask

def high_pass_mask(shape, radius):
    return 1 - low_pass_mask(shape, radius)

def band_pass_mask(shape, low_radius, high_radius):
    return low_pass_mask(shape, high_radius) - low_pass_mask(shape, low_radius)

# --------- Apply Filter ---------
def apply_filter(dft_shift, mask):
    filtered = dft_shift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    magnitude = np.log(np.abs(filtered) + 1)
    return img_back, magnitude

# --------- Display Images and Spectra ---------
def display_images(imgs, titles, rows=2, cols=4, figsize=(20,10)):
    plt.figure(figsize=figsize)
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        if isinstance(img, np.ndarray) and img.ndim == 2:
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        else:  # histogram or other array
            plt.bar(np.arange(256), img, color='blue')
            plt.xlim([0,256])
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()

# --------- Main Processing Function ---------
def process_images(image_folder, image_names):
    contrast_types = ['low', 'normal', 'high']
    
    for img_name in image_names:
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        img_gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        for contrast in contrast_types:
            img_gray = adjust_contrast(img_gray_original, contrast)
            
            # Compute FFT
            dft_shift, mag, img_back = compute_fft(img_gray)
            
            # Filter radii
            radius_low = img_gray.shape[0] // 4
            radius_high = img_gray.shape[0] // 6
            
            # Low-pass filter
            lp_mask = low_pass_mask(dft_shift.shape, radius_low)
            img_lp, mag_lp = apply_filter(dft_shift, lp_mask)
            
            # High-pass filter
            hp_mask = high_pass_mask(dft_shift.shape, radius_high)
            img_hp, mag_hp = apply_filter(dft_shift, hp_mask)
            
            # Band-pass filter
            bp_mask = band_pass_mask(dft_shift.shape, radius_high, radius_low)
            img_bp, mag_bp = apply_filter(dft_shift, bp_mask)
            
            # Display results
            imgs_to_show = [img_gray, mag, img_back, img_lp, mag_lp, img_hp, mag_hp, img_bp, mag_bp]
            titles = [f'{contrast} Input', 'FFT Magnitude', 'Reconstructed',
                      'Low-pass', 'Low-pass Magnitude', 
                      'High-pass', 'High-pass Magnitude', 
                      'Band-pass', 'Band-pass Magnitude']
            
            display_images(imgs_to_show, titles, rows=3, cols=3)

# --------- Run ---------
if __name__ == '__main__':
    image_folder = r"E:\dipimage"  # folder containing your images
    image_names = ['g1.jpeg','g2.jpeg','g3.jpeg','source.jpg','tulip.jpg']
    process_images(image_folder, image_names)
