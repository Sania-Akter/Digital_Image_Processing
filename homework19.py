import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

IMG_PATH = r"E:\dipimage\sapla.jpg"


def quantize_gray(gray: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits
    q = np.floor(gray.astype(np.float32) * levels / 256.0).astype(np.uint8)
    q = np.clip(q, 0, levels - 1)
    step = 256.0 / levels
    rec = q.astype(np.float32) * step + step / 2.0
    return np.clip(rec, 0, 255).astype(np.uint8)


def palette_8bit(img_bgr: np.ndarray, K: int = 256) -> np.ndarray:
    Z = img_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        Z, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = centers.astype(np.uint8)
    quant = centers[labels.flatten()]
    return quant.reshape(img_bgr.shape)


def quantize_3_3_2(img_bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img_bgr)

    r3 = np.round(r.astype(np.float32) * 7.0 / 255.0).astype(np.uint8)
    g3 = np.round(g.astype(np.float32) * 7.0 / 255.0).astype(np.uint8)
    b2 = np.round(b.astype(np.float32) * 3.0 / 255.0).astype(np.uint8)

    r_rec = np.round(r3.astype(np.float32) * 255.0 / 7.0).astype(np.uint8)
    g_rec = np.round(g3.astype(np.float32) * 255.0 / 7.0).astype(np.uint8)
    b_rec = np.round(b2.astype(np.float32) * 255.0 / 3.0).astype(np.uint8)

    return cv2.merge([b_rec, g_rec, r_rec])


def pct_change(orig_bits: int, new_bits: int) -> float:
    return 100.0 * (1.0 - new_bits / max(1, orig_bits))


# ---------- SSIM / MSE / PSNR (grayscale) ----------

def ssim_gray(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel_size = (11, 11)
    sigma = 1.5

    mu_x = cv2.GaussianBlur(x, kernel_size, sigma)
    mu_y = cv2.GaussianBlur(y, kernel_size, sigma)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(x * x, kernel_size, sigma) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, kernel_size, sigma) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, kernel_size, sigma) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = num / (den + 1e-12)
    return float(ssim_map.mean())


def mse_psnr_ssim(ref: np.ndarray, test: np.ndarray):
    ref = ref.astype(np.float32)
    test = test.astype(np.float32)
    mse = np.mean((ref - test) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10((255.0 ** 2) / mse)
    s = ssim_gray(ref, test)
    return mse, psnr, s


def main():
    img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not read image: " + IMG_PATH)

    H, W, _ = img_bgr.shape
    orig_bits = H * W * 24

    # ---------- compressions ----------
    img_palette = palette_8bit(img_bgr, K=256)
    bits_palette = H * W * 8
    pct_palette = pct_change(orig_bits, bits_palette)

    img_332 = quantize_3_3_2(img_bgr)
    bits_332 = H * W * 8
    pct_332 = pct_change(orig_bits, bits_332)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    gray_8 = quantize_gray(img_gray, 8)
    bits_8 = H * W * 8
    pct_8 = pct_change(orig_bits, bits_8)

    gray_5 = quantize_gray(img_gray, 5)
    bits_5 = H * W * 5
    pct_5 = pct_change(orig_bits, bits_5)

    gray_4 = quantize_gray(img_gray, 4)
    bits_4 = H * W * 4
    pct_4 = pct_change(orig_bits, bits_4)

    gray_2 = quantize_gray(img_gray, 2)
    bits_2 = H * W * 2
    pct_2 = pct_change(orig_bits, bits_2)

    gray_1 = quantize_gray(img_gray, 1)
    bits_1 = H * W * 1
    pct_1 = pct_change(orig_bits, bits_1)

    # ---------- bit summary ----------
    print("\n=========== Bit Compression Summary (vs 24-bit) ===========")
    print(f"Original 24-bit color   : {orig_bits} bits   | Change: 0.00%")
    print("-----------------------------------------------------------")
    print(f"8-bit palette (256 col) : {bits_palette} bits | Change: {pct_palette:.2f}%")
    print(f"8-bit 3-3-2 RGB         : {bits_332} bits     | Change: {pct_332:.2f}%")
    print(f"8-bit grayscale         : {bits_8} bits      | Change: {pct_8:.2f}%")
    print(f"5-bit grayscale         : {bits_5} bits      | Change: {pct_5:.2f}%")
    print(f"4-bit grayscale         : {bits_4} bits      | Change: {pct_4:.2f}%")
    print(f"2-bit grayscale         : {bits_2} bits      | Change: {pct_2:.2f}%")
    print(f"1-bit grayscale         : {bits_1} bits      | Change: {pct_1:.2f}%")
    print("===========================================================\n")

    # ---------- quality metrics (all vs original grayscale) ----------
    pal_gray   = cv2.cvtColor(img_palette, cv2.COLOR_BGR2GRAY)
    img_332_g  = cv2.cvtColor(img_332,   cv2.COLOR_BGR2GRAY)

    print("=========== Quality Metrics (vs original grayscale) ===========")
    print("Method           |    MSE      |  PSNR (dB) |   SSIM")
    print("---------------  | ----------  | ---------- | ------")

    mse, psnr, s = mse_psnr_ssim(img_gray, pal_gray)
    print(f"8-bit palette    | {mse:10.2f} | {psnr:10.2f} | {s:6.4f}")

    mse, psnr, s = mse_psnr_ssim(img_gray, img_332_g)
    print(f"8-bit 3-3-2 RGB  | {mse:10.2f} | {psnr:10.2f} | {s:6.4f}")

    mse, psnr, s = mse_psnr_ssim(img_gray, gray_8)
    print(f"8-bit grayscale  | {mse:10.2f} | {psnr:10.2f} | {s:6.4f}")

    mse, psnr, s = mse_psnr_ssim(img_gray, gray_5)
    print(f"5-bit grayscale  | {mse:10.2f} | {psnr:10.2f} | {s:6.4f}")

    mse, psnr, s = mse_psnr_ssim(img_gray, gray_4)
    print(f"4-bit grayscale  | {mse:10.2f} | {psnr:10.2f} | {s:6.4f}")

    mse, psnr, s = mse_psnr_ssim(img_gray, gray_2)
    print(f"2-bit grayscale  | {mse:10.2f} | {psnr:10.2f} | {s:6.4f}")

    mse, psnr, s = mse_psnr_ssim(img_gray, gray_1)
    print(f"1-bit grayscale  | {mse:10.2f} | {psnr:10.2f} | {s:6.4f}")

    print("===============================================================\n")

    # ---------- visualization ----------
    img_rgb      = cv2.cvtColor(img_bgr,     cv2.COLOR_BGR2RGB)
    pal_rgb      = cv2.cvtColor(img_palette, cv2.COLOR_BGR2RGB)
    img_332_rgb  = cv2.cvtColor(img_332,     cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 4, 1)
    plt.title("Original\n24-bit")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.title(f"8-bit palette\n{pct_palette:.1f}%")
    plt.imshow(pal_rgb)
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.title(f"8-bit 3-3-2\n{pct_332:.1f}%")
    plt.imshow(img_332_rgb)
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.title(f"8-bit gray\n{pct_8:.1f}%")
    plt.imshow(gray_8, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.title(f"5-bit gray\n{pct_5:.1f}%")
    plt.imshow(gray_5, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.title(f"4-bit gray\n{pct_4:.1f}%")
    plt.imshow(gray_4, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.title(f"2-bit gray\n{pct_2:.1f}%")
    plt.imshow(gray_2, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.title(f"1-bit gray\n{pct_1:.1f}%")
    plt.imshow(gray_1, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()