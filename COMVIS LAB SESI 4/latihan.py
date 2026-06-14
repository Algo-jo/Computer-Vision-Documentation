import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('picture.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Image", img_gray)
# cv2.waitKey(0)

canny = cv2.Canny(img_gray, 100, 200)
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)

images = [canny, sobel_x, sobel_y, laplacian]
titles = ["Canny", "Sobel_x", "Sobel_y", "Laplacian"]

plt.figure(figsize=(10, 10))
for i, (curr_image, curr_title) in enumerate(zip(images, titles)):
    plt.subplot(3,3, i+1)
    plt.imshow(curr_image, 'gray')
    plt.title(curr_title)
    plt.axis(False)
plt.show()