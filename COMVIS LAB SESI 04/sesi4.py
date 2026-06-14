import cv2
import numpy as np
import matplotlib.pyplot as plt


#===========================================================================================
img = cv2.imread("67.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# LAPLACIAN 
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
laplacian_uint = np.uint8(np.absolute(laplacian))
# cv2.imshow("Image", laplacian_uint)
# cv2.waitKey(0)

# SOBEL
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
# cv2.imshow("Image", sobel_x)
# cv2.waitKey(0)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
# cv2.imshow("Image", sobel_y)
# cv2.waitKey(0)
sobel_xy = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=5)
# cv2.imshow("Image", sobel_xy)
# cv2.waitKey(0)

# CANNY
canny = cv2.Canny(img_gray, 100, 200)
# cv2.imshow("Sixty Nine", canny)
# cv2.waitKey(0)

images = [
    img_gray, laplacian, laplacian_uint, sobel_x, sobel_y, sobel_xy, canny
]
titles = [
    "Original", "Laplacian", "Laplacian_Uint", "Sobel_x", "Sobel_y", "Sobel_xy", "Canny"
]

plt.figure(figsize=(8, 8))
for i, (curr_image, curr_title) in enumerate(zip(images, titles)):
    plt.subplot(3, 3, (i+1))
    plt.imshow(curr_image, 'gray')
    plt.title(curr_title)
    plt.axis(False)
plt.show()