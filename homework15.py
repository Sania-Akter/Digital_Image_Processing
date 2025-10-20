import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Helper function ----------
def apply_filter(img, H):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    G = fshift * H
    g = np.fft.ifftshift(G)
    img_back = np.abs(np.fft.ifft2(g))
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)

# ---------- Filter creation ----------
def butterworth_filter(shape, D0, n, type='low', W=10):
    """Butterworth LPF, HPF, BPF, BSF"""
    M, N = shape
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
            if type == 'low':
                H[u, v] = 1 / (1 + (D / D0)**(2 * n))
            elif type == 'high':
                H[u, v] = 1 / (1 + (D0 / D)**(2 * n)) if D != 0 else 0
            elif type == 'bandpass':
                # Butterworth Bandpass = HPF * LPF (with width W)
                H[u, v] = 1 / (1 + ((D*W) / (D**2 - D0**2))**(2 * n)) if D != D0 else 0
    return H

def gaussian_filter(shape, D0, type='low', W=10):
    """Gaussian LPF, HPF, BPF, BSF"""
    M, N = shape
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
            if type == 'low':
                H[u, v] = np.exp(-(D**2) / (2 * D0**2))
            elif type == 'high':
                H[u, v] = 1 - np.exp(-(D**2) / (2 * D0**2))
            elif type == 'bandpass':
                H[u, v] = np.exp(-((D**2 - D0**2)**2) / (D**2 * W**2)) if D != 0 else 0
        
    return H

# ---------- Main ----------
img = cv2.imread(r"E:\dipimage\sapla.jpg", cv2.IMREAD_GRAYSCALE)

# Create low, normal, high contrast versions
low_contrast = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
normal_contrast = img
high_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

images = {'Low': low_contrast, 'Normal': normal_contrast, 'High': high_contrast}

D0 = 40
n = 2
W = 20  # Bandwidth for band filters

for label, im in images.items():
    plt.figure(figsize=(12, 8))
    plt.suptitle(f'{label} Contrast Image Filtering (Butterworth + Gaussian)', fontsize=14)

    # Butterworth Filters
    H_blpf = butterworth_filter(im.shape, D0, n, type='low')
    H_bhpf = butterworth_filter(im.shape, D0, n, type='high')
    H_bbpf = butterworth_filter(im.shape, D0, n, type='bandpass', W=W)

    # Gaussian Filters
    H_glpf = gaussian_filter(im.shape, D0, type='low')
    H_ghpf = gaussian_filter(im.shape, D0, type='high')
    H_gbpf = gaussian_filter(im.shape, D0, type='bandpass', W=W)


    # Apply filters
    img_blpf = apply_filter(im, H_blpf)
    img_bhpf = apply_filter(im, H_bhpf)
    img_bbpf = apply_filter(im, H_bbpf)
    

    img_glpf = apply_filter(im, H_glpf)
    img_ghpf = apply_filter(im, H_ghpf)
    img_gbpf = apply_filter(im, H_gbpf)

    # Display results
    filters = [
        ('Original', im),
        ('Butterworth LPF', img_blpf),
        ('Butterworth HPF', img_bhpf),
        ('Butterworth BPF', img_bbpf),
        ('Gaussian LPF', img_glpf),
        ('Gaussian HPF', img_ghpf),
        ('Gaussian BPF', img_gbpf)
    ]

    for i, (title, fimg) in enumerate(filters, 1):
        plt.subplot(3, 3, i)
        plt.imshow(fimg, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ---------- Effect of Butterworth Filter Order ----------
plt.figure(figsize=(12, 6))
plt.suptitle("Effect of Butterworth Filter Order (n)", fontsize=14)

n_values = [1, 2, 4, 8]
for i, n in enumerate(n_values, 1):
    H = butterworth_filter(normal_contrast.shape, D0, n=n, type='low')
    img_f = apply_filter(normal_contrast, H)
    plt.subplot(2, 4, i)
    plt.imshow(img_f, cmap='gray')
    plt.title(f'n = {n}')

plt.tight_layout()
plt.show()
