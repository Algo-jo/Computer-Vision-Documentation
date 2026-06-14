import cv2
import matplotlib.pyplot as plt

def load_image_gs(path: str):
    img = cv2.imread(path)
    if img is None:
        print("Image path not found")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

image = load_image_gs('./mariobros.png')

ret, binary = cv2.threshold(
    image,
    50,
    255,
    cv2.THRESH_BINARY
)

plt.figure(figsize=(15, 10))
plt.subplots(1,1,1)
plt.imshow(image[0])
plt.subplots(1,1,2)
plt.imshow(image[1])
plt.show()

cv2.imshow("Binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
