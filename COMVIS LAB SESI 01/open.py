import cv2
import numpy as np

cat = cv2.imread("images.jpg")
# cv2.imshow('Cat', cat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cat_black = cv2.cvtColor(cat, cv2.COLOR_BGRA2GRAY)
# print(cat_black)
# cv2.imshow('Cat', cat_black)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

pixel = cat[100, 100]
print(pixel)
cat[1:50, 1:200] = [0, 255, 0]
cv2.imshow('Cat', cat)
cv2.waitKey(0)
cv2.destroyAllWindows()
