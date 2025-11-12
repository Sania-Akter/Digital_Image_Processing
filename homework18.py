import cv2, numpy as np, heapq
from collections import Counter
import matplotlib.pyplot as plt

# ====================== SETTINGS ===========================================
IMG_PATH	= r"E:\dipimage\tulip1.jpg"
DCT_QSTEP	= 10.0 
DWT_QSTEP	= 1.0  
WAVELET		= "haar"
# ===========================================================================

# Optional PyWavelets for DWT
try:
	import pywt
	HAS_PYWT = True
except ImportError:
	HAS_PYWT = False

def huff_bits(arr: np.ndarray) -> int:
	flat = arr.ravel().tolist()
	if not flat: return 0
	freq = Counter(flat)
	h=[]; uid=0
	for s,f in freq.items():
		heapq.heappush(h, (f, uid, (s,None,None))); uid+=1
	while len(h)>1:
		f1,_,n1 = heapq.heappop(h)
		f2,_,n2 = heapq.heappop(h)
		heapq.heappush(h, (f1+f2, uid, (None,n1,n2))); uid+=1
	root = h[0][2]
	codes={}
	def walk(node,p=""):
		s,l,r = node
		if s is not None: codes[s]=p or "0"; return
		walk(l,p+"0"); walk(r,p+"1")
	walk(root)
	return sum(len(codes[v]) for v in flat)

def pct_smaller(orig_bits, comp_bits):
	return max(0.0, 100.0*(1.0 - comp_bits/max(1,orig_bits)))

# 1) Huffman (lossless)
def do_huffman(img):
	bits = huff_bits(img)
	return bits, img.copy()

# 2) DCT (cv2) + uniform quant + Huffman
def do_dct(img, qstep=DCT_QSTEP):
	x = img.astype(np.float32) - 128.0
	C = cv2.dct(x)
	Cq = np.rint(C / qstep).astype(np.int32)
	bits = huff_bits(Cq)
	y = cv2.idct(Cq.astype(np.float32) * qstep) + 128.0
	return bits, np.clip(y, 0, 255).astype(np.uint8)

# 3) DWT (pywt) + uniform quant + Huffman — simpler
def do_dwt(img, wave=WAVELET, qstep=DWT_QSTEP):
	if not HAS_PYWT:
		raise RuntimeError("PyWavelets not installed. pip install pywavelets")
	x = img.astype(np.float32)
	cA, (cH, cV, cD) = pywt.dwt2(x, wave)
	cAq = np.rint(cA / qstep).astype(np.int32)
	cHq = np.rint(cH / qstep).astype(np.int32)
	cVq = np.rint(cV / qstep).astype(np.int32)
	cDq = np.rint(cD / qstep).astype(np.int32)
	bits = huff_bits(np.concatenate([cAq.ravel(), cHq.ravel(), cVq.ravel(), cDq.ravel()]))
	y = pywt.waverec2((cAq * qstep, (cHq * qstep, cVq * qstep, cDq * qstep)), wave)
	y = y[:img.shape[0], :img.shape[1]]
	return bits, np.clip(y, 0, 255).astype(np.uint8)


def main():
	img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
	if img is None:
		raise ValueError(f"Could not read image: {IMG_PATH}")
	H,W = img.shape
	orig_bits = H*W*8

	hb, hrecon = do_huffman(img)
	db, drecon = do_dct(img, DCT_QSTEP)
	if HAS_PYWT:
		wb, wrecon = do_dwt(img, WAVELET, DWT_QSTEP)
	else:
		wb = wrecon = None

	ph	= pct_smaller(orig_bits, hb)
	pdct= pct_smaller(orig_bits, db)
	print("\n=== Bits & % Compressed ===")
	print(f"Original bits: {orig_bits}")
	print(f"Huffman bits : {hb}\t| % compressed: {ph:.1f}%")
	print(f"DCT bits     : {db}\t| % compressed: {pdct:.1f}%")
	if HAS_PYWT:
		pwdt = pct_smaller(orig_bits, wb)
		print(f"DWT bits     : {wb}\t| % compressed: {pwdt:.1f}%")
	else:
		print("DWT bits     : (PyWavelets not installed)")

	cols = 4 if HAS_PYWT else 3
	plt.figure(figsize=(4*cols,4))
	plt.subplot(1, cols, 1); plt.title("Original"); plt.imshow(img, cmap='gray'); plt.axis('off')
	plt.subplot(1, cols, 2); plt.title(f"Huffman (lossless)\n{ph:.1f}% smaller"); plt.imshow(hrecon, cmap='gray'); plt.axis('off')
	plt.subplot(1, cols, 3); plt.title(f"DCT\n{pdct:.1f}% smaller"); plt.imshow(drecon, cmap='gray'); plt.axis('off')
	if HAS_PYWT:
		plt.subplot(1, cols, 4); plt.title(f"DWT\n{pwdt:.1f}% smaller"); plt.imshow(wrecon, cmap='gray'); plt.axis('off')
	plt.tight_layout(); plt.show()

if __name__ == "__main__":
	main()