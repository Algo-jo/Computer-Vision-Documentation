import cv2

def load_image_gs(path: str):
    img = cv2.imread(path)
    if img is None:
        print("Image path not found")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# Gaussian Blur -> mengurangi noise umum pada gambar
image = load_image_gs('./tree.jpg')
gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)

# Median Blur 
median_filtered = cv2.medianBlur(image, 5)

bilateral_filtered = cv2.bilateralFilter(
    image, 5, 0, 2
)
# Thresholding
# Simple thresholding -> Ubah semua intensitas di atas 127 ke warna hitam
# print(median_filtered)
# ret, simple_thresh = cv2.threshold(
#     median_filtered, 30, 255, cv2.THRESH_BINARY
#     )

# Adaptive thresholding -> menentukan thresholding satu titik berdasarkan area sekitarnya
# ret, adaptive_thresh = cv2.adaptiveThreshold(
#     median_filtered, 30, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2
#     )

# Otsu's Thresholding -> mencari nilai batas paling optimal secara otomatis
# otsu_ret, otsu_thresh = cv2.threshold(
#     0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
# )

# otsu_ret_filtered, otsu_thresh_filtered = cv2.threshold(
#     median_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
# )

# cv2.imshow("pohon", image)
# cv2.imshow("pohon", gaussian_filtered)
cv2.imshow("pohon", bilateral_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()