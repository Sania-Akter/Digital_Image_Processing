import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Custom implementations
# ---------------------------

def erosion_scratch(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    # Padding with black (0) for erosion
    padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                cv2.BORDER_CONSTANT, value=0)
    out = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            if np.all(region[kernel==1] == 255):
                out[i, j] = 255
            else:
                out[i, j] = 0
    return out

def dilation_scratch(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    # Padding with black (0) for dilation is correct
    padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                cv2.BORDER_CONSTANT, value=0)
    out = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            if np.any(region[kernel==1] == 255):
                out[i, j] = 255
            else:
                out[i, j] = 0
    return out

def opening_scratch(img, kernel):
    return dilation_scratch(erosion_scratch(img, kernel), kernel)

def closing_scratch(img, kernel):
    return erosion_scratch(dilation_scratch(img, kernel), kernel)

def tophat_scratch(img, kernel):
    return cv2.subtract(img, opening_scratch(img, kernel))

def blackhat_scratch(img, kernel):
    return cv2.subtract(closing_scratch(img, kernel), img)

# ---------------------------
# Structuring Elements
# ---------------------------

def diamond_kernel(size=5):
    kernel = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    for i in range(size):
        for j in range(size):
            if abs(i-mid) + abs(j-mid) <= mid:
                kernel[i, j] = 1
    return kernel

kernels = {
    "Rectangular": cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),
    "Elliptical": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
    "Cross": cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)),
    "Diamond": diamond_kernel(5)
}

# ---------------------------
# Main
# ---------------------------

def main():
    # Load grayscale and threshold to binary
    img = cv2.imread(r"E:\dipimage\bs.jpg", 0)   # Use raw string for path
    if img is None:
        print("Error: Image not found. Check the path.")
        return

    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    for name, kernel in kernels.items():
        print(f"\n--- Using {name} Kernel ---")

        # Built-in
        erosion_cv = cv2.erode(img, kernel)
        dilation_cv = cv2.dilate(img, kernel)
        opening_cv = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        closing_cv = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        tophat_cv = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        blackhat_cv = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

        # Scratch
        erosion_sc = erosion_scratch(img, kernel)
        dilation_sc = dilation_scratch(img, kernel)
        opening_sc = opening_scratch(img, kernel)
        closing_sc = closing_scratch(img, kernel)
        tophat_sc = tophat_scratch(img, kernel)
        blackhat_sc = blackhat_scratch(img, kernel)

        titles = [
            "Original", "Erosion CV", "Erosion Scratch",
            "Dilation CV", "Dilation Scratch",
            "Opening CV", "Opening Scratch",
            "Closing CV", "Closing Scratch",
            "Top-hat CV", "Top-hat Scratch",
            "Black-hat CV", "Black-hat Scratch"
        ]
        images = [
            img, erosion_cv, erosion_sc,
            dilation_cv, dilation_sc,
            opening_cv, opening_sc,
            closing_cv, closing_sc,
            tophat_cv, tophat_sc,
            blackhat_cv, blackhat_sc
        ]

        # Show all results
        plt.figure(figsize=(15,10))
        for i in range(len(images)):
            plt.subplot(4,4,i+1)
            plt.imshow(images[i], cmap="gray")
            plt.title(titles[i], fontsize=9)
            plt.axis("off")

        plt.suptitle(f"Results with {name} Kernel", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Prevent overlap
        plt.show()

if __name__ == "__main__":
    main()
