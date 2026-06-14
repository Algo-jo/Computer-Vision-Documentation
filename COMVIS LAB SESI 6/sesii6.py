import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'COMVIS LAB SESI 6\box.png')
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ORB
#==========================================================================================
orb = cv2.ORB_create()
keypoints, descriptor = orb.detectAndCompute(img, None)
orb_result = img.copy()
cv2.drawKeypoints(img, keypoints, orb_result, [0,255,0])
orb_result = cv2.cvtColor(orb_result, cv2.COLOR_BGR2RGB)

# cv2.imshow("Image", orb_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# AKAZE
#==========================================================================================
Akaze = cv2.AKAZE_create()
keypoints, descriptor = Akaze.detectAndCompute(img, None)
Akaze_result = img.copy()
cv2.drawKeypoints(img, keypoints, Akaze_result, [0,255,0])
Akaze_result = cv2.cvtColor(Akaze_result, cv2.COLOR_BGR2RGB)

# cv2.imshow("Image", Akaze_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# KAZE
#==========================================================================================
kaze = cv2.KAZE_create()
keypoints, descriptor = kaze.detectAndCompute(img, None)
kaze_result = img.copy()
cv2.drawKeypoints(img, keypoints, kaze_result, [0,255,0])
kaze_result = cv2.cvtColor(kaze_result, cv2.COLOR_BGR2RGB)

# cv2.imshow("Image", kaze_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# SIFT
#==========================================================================================
sift = cv2.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(img, None)
sift_result = img.copy()
cv2.drawKeypoints(img, keypoints, sift_result, [0,255,0])
sift_result = cv2.cvtColor(sift_result, cv2.COLOR_BGR2RGB)

# cv2.imshow("Image", sift_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#==========================================================================================
title = ['ORB', 'AKAZE', 'KAZE', 'SIFT']
img_list = [orb_result, Akaze_result, kaze_result, sift_result]

plt.figure(figsize = (6,6))
for i, (curr_image, curr_title) in enumerate(zip(img_list, title)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(curr_image)
    plt.title(curr_title)
    plt.axis(False)
plt.show()


#FLANN
#==========================================================================================
img_object = cv2.imread(r'COMVIS LAB SESI 6\box.png')
img_scene = cv2.imread(r'COMVIS LAB SESI 6\box_in_scene.png')

img_object = cv2.cvtColor(img_object, cv2.COLOR_BGR2RGB)
img_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2RGB)

sift = cv2.SIFT_create()
kp_object, desc_object = sift.detectAndCompute(img_object, None)
kp_scene, desc_scene = sift.detectAndCompute(img_scene, None)

FLANN_INDEX = 1
index_param = dict(
    algorithm = FLANN_INDEX,
    tree = 5
)
search_param = dict(check = 50)

flann = cv2.FlannBasedMatcher(index_param, search_param)
matches = flann.knnMatch(desc_object, desc_scene, k=2)

matches_mask = []
for _ in range(len(matches)):
    matches_mask.append([0, 0])

for i, (m,n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matches_mask[i] = [1, 0]

img_res2 = cv2.drawMatchesKnn(
    img_object, kp_object,
    img_scene, kp_scene,
    matches, None,
    matchColor= [0, 255, 0],
    singlePointColor = [255, 0, 0],
    matchesMask=matches_mask
)

plt.imshow(img_res2)
plt.show()

#BRUTE FORCE
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc_object, desc_scene, k = 2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])
        
img_res3 = cv2.drawMatchesKnn(
    img_object, kp_object,
    img_scene, kp_scene,
    good, None,
    matchColor= [0, 255, 0],
    singlePointColor = [255, 0, 0],
)

plt.imshow(img_res3)
plt.show()