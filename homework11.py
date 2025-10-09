import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread(r"E:\dipimage\sapla.jpg", cv2.IMREAD_GRAYSCALE)
N, M = img.shape
s = 64  # Grid size

# Function to split image into grids
def split_into_grids(image, grid_size):
    grids = []
    for i in range(0, image.shape[0], grid_size):
        for j in range(0, image.shape[1], grid_size):
            grids.append(image[i:i+grid_size, j:j+grid_size])
    return grids

# Function to merge grids back to image
def merge_grids(grids, img_shape, grid_size):
    merged = np.zeros(img_shape, dtype=np.uint8)
    idx = 0
    for i in range(0, img_shape[0], grid_size):
        for j in range(0, img_shape[1], grid_size):
            merged[i:i+grid_size, j:j+grid_size] = grids[idx]
            idx += 1
    return merged

# Split image into grids
grids = split_into_grids(img, s)

# Apply different operations on each grid
he_grids = [cv2.equalizeHist(g) for g in grids]  # Histogram Equalization
ahe_grids = [cv2.createCLAHE(clipLimit=40.0).apply(g) for g in grids]  # Using high clip limit as AHE
clahe_grids_1 = [cv2.createCLAHE(clipLimit=2.0).apply(g) for g in grids]  # CLAHE low clip
clahe_grids_2 = [cv2.createCLAHE(clipLimit=8.0).apply(g) for g in grids]  # CLAHE higher clip

# Merge grids back to images
he_img = merge_grids(he_grids, img.shape, s)
ahe_img = merge_grids(ahe_grids, img.shape, s)
clahe_img_1 = merge_grids(clahe_grids_1, img.shape, s)
clahe_img_2 = merge_grids(clahe_grids_2, img.shape, s)

# Optional: Bi-linear interpolation on AHE image
ahe_img_interpolated = cv2.resize(ahe_img, (M, N), interpolation=cv2.INTER_LINEAR)

# Display results
plt.figure(figsize=(10,6))
plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(he_img, cmap='gray') 
plt.title("HE")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(ahe_img, cmap='gray')
plt.title("AHE")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(clahe_img_1, cmap='gray')
plt.title("CLAHE clip=2")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(clahe_img_2, cmap='gray')
plt.title("CLAHE clip=8")
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(ahe_img_interpolated, cmap='gray')
plt.title("AHE + Bilinear Interp")
plt.axis('off')


plt.tight_layout()
plt.show()
