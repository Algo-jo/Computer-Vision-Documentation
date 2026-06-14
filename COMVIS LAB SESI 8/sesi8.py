import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import colormaps

# 1. Convert Image to Bag of Visual Words
#================================================================================
def generate_bovw(image_list, k=50):
    orb = cv2.ORB_create()
    desc_list = []
    
    # Detect Image Descriptors
    for image in image_list:
        _, desc = orb.detectAndCompute(image, None)
        if desc is not None:
            desc_list.append(desc)
    
    # Flatten Array
    descriptors = np.vstack(desc_list)
    
    # K-Means
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=45)
    kmeans.fit(descriptors)
    
    # Histogram for First Image
    bag = kmeans.predict(desc_list[0])
    hist = np.histogram(bag, bins=(k+1), density=True)
    
    return hist, kmeans

def compute_depth(img_left, img_right):
    img_left_size = img_left.shape[:2]
    img_right_size = img_right.shape[:2]
    
    smallestWidth = min(img_left_size[0], img_right_size[0])
    smallestHeight = min(img_left_size[1], img_right_size[1])

    img_left = cv2.resize(
        img_left, 
        (smallestWidth, smallestHeight), 
        interpolation=cv2.INTER_AREA
    )
    
    img_right = cv2.resize(
        img_right, 
        (smallestWidth, smallestHeight),
        interpolation=cv2.INTER_AREA
    )
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64, # Must be divided by 16
        blockSize=5, # Odd number
        P1=8 * 3 * 5 ** 2, # Penalty for small disparity changes  
        P2=32 * 3 * 5 ** 2, # Penalty for big disparity changes
         
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    disparity = stereo.compute(img_left, img_right)
    visualize_depth_map = cv2.normalize(
        disparity, None, 0, 255, 
        cv2.NORM_MINMAX,
        cv2.CV_8U
    )
    
    return visualize_depth_map

images = []
for image in os.listdir("images"):
    images.append(
        cv2.imread(os.path.join("images", image))
    )

hist, kmeans = generate_bovw(images)
# plt.hist(hist)
# plt.show()
# print(list(colormaps))

depth = compute_depth(images[0], images[2])
plt.imshow(depth, cmap='inferno')
plt.show()