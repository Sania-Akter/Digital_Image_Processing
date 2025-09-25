import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.exposure import match_histograms

#================= Histogram Calculation ==============================
def histogram(img_2D):
    # Ensure image is uint8 before computing histogram
    if img_2D.dtype != np.uint8:
        img_2D = cv2.normalize(img_2D, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.calcHist([img_2D], [0], None, [256], [0, 256]).flatten()

def pdf_f(hist):
    return hist / hist.sum()

def cdf_f(pdf):
    return np.cumsum(pdf)

#================= Mapping-based Histogram Matching ===================
def match_histogram_custom(src_cdf, ref_cdf):
    mapping = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        diff = np.abs(ref_cdf - src_cdf[src_val])
        mapping[src_val] = np.argmin(diff)
    return mapping

def img_conv(img_gray, mapping):
    return mapping[img_gray]

#================= Display Function ===================================
def display_results(title, src, ref, matched_builtin, matched_custom):
    # Normalize all to uint8 for display
    imgs = [src, ref, matched_builtin, matched_custom]
    imgs = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for img in imgs]

    plt.figure(figsize=(16, 12))
    labels = ["Source", "Reference", "Matched (Skimage)", "Matched (Custom)"]

    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        # Image
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.title(lbl)

        # Histogram
        plt.subplot(2, 4, i + 5)
        plt.plot(histogram(img), color='black')
        plt.title(f"{lbl} Histogram")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

#================= Generate Contrast Variants =========================
def adjust_contrast(img, level="normal"):
    if level == "low":
        return cv2.normalize(img, None, 100, 150, cv2.NORM_MINMAX)  # compress range
    elif level == "high":
        return cv2.equalizeHist(img)  # histogram equalization
    else:
        return img  # normal

#================= Main ===============================================
def main():
    src_path = r"E:\dipimage\source.jpg"
    ref_path = r"E:\dipimage\refer.jpg"
    src_img = cv2.imread(src_path, 0)
    ref_img = cv2.imread(ref_path, 0)

    if src_img is None or ref_img is None:
        print("Error: Could not load images. Check the paths.")
        return

    # Contrast cases
    contrast_levels = ["low", "normal", "high"]

    for src_level in contrast_levels:
        for ref_level in contrast_levels:
            src_mod = adjust_contrast(src_img, src_level)
            ref_mod = adjust_contrast(ref_img, ref_level)

            # Built-in scikit-image histogram matching
            matched_builtin = match_histograms(src_mod, ref_mod)

            # Custom mapping method
            src_hist = histogram(src_mod)
            ref_hist = histogram(ref_mod)
            src_cdf = cdf_f(pdf_f(src_hist))
            ref_cdf = cdf_f(pdf_f(ref_hist))
            mapping = match_histogram_custom(src_cdf, ref_cdf)
            matched_custom = img_conv(src_mod, mapping)

            # Show results
            title = f"Source: {src_level} contrast | Reference: {ref_level} contrast"
            display_results(title, src_mod, ref_mod, matched_builtin, matched_custom)

#================= Run Script =========================================
if __name__ == "__main__":
    main()
