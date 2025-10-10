from PIL import Image
from collections import Counter
import heapq
import cv2
import matplotlib.pyplot as plt

def huffman_image(img, top_n=15):
	
	pixels=img.flatten().tolist()
	total=len(pixels)
	# frequency
	freq=Counter(pixels) 

	# build Huffman tree using tuples (sym,left,right)
	h=[]; uid=0
	for s,f in freq.items():
		heapq.heappush(h,(f,uid,(s,None,None))); uid+=1
	while len(h)>1:
		f1,_,n1=heapq.heappop(h); f2,_,n2=heapq.heappop(h)
		heapq.heappush(h,(f1+f2,uid,(None,n1,n2))); uid+=1
	root=h[0][2]

	#  make codes by DFS walk (left=0, right=1)
	codes={}
	def walk(node,p=""):
		s,l,r=node
		if s is not None: codes[s]=p or "0"; return
		walk(l,p+"0"); walk(r,p+"1")
	walk(root)

	# encode & print simple stats
	bits=''.join(codes[v] for v in pixels)
	orig=total*8
	comp=max(1,len(bits))
	print("Original bits:",orig)
	print("Compressed bits:",len(bits))
	print("Compression ratio:",round(orig/comp,2))

	# show frequency + code (top-N most frequent)
	print("\nValue  Count     Prob    Len  Code")
	for v,c in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]:
		prob=c/total; code=codes[v]
		print(f"{v:>5}  {c:>6}  {prob:>8.5f}  {len(code):>4}  {code}")

	#main function
def main():	
	img_path = r"E:\dipimage\sapla.jpg"
	img_path1 = r"E:\dipimage\sapla0.png"
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
	
	print("Result for .jpg image")	
	huffman_image(img, top_n=15)
	print("Result for .png image")	
	huffman_image(img1, top_n=15)
		
	plt.subplot(121);plt.title(".jpg image"); plt.imshow(img,cmap = 'gray')
	plt.subplot(122);plt.title(".png image"); plt.imshow(img1,cmap = 'gray')
	plt.show()
	
	# Call the main function
if __name__== "__main__":
	main()