import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# PREPROCESSING
#==============================================================================================
# Listing images
source_dir = r'PRACTICE\images\source'
source_images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpeg', '.jpg'))]

# Load images
img = cv2.imread(r'PRACTICE\images\target\hina.png')

# Smoothing Images
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pro = cv2.GaussianBlur(gray, (3, 3), 0)
    return img_pro

img_pro = preprocess(img)

# Feature Detection
#==============================================================================================
sift = cv2.SIFT_create()
kp_target, desc_target = sift.detectAndCompute(img_pro, None)

if desc_target is not None and desc_target.dtype != np.float32:
    desc_target = desc_target.astype(np.float32)

# sift_res = img_pro.copy()
# cv2.drawKeypoints(img, kp_target, None, [0, 255, 0])
# sift_res = cv2.cvtColor(sift_res, cv2.COLOR_BGR2RGB)

# cv2.imshow('SIFT', sift_res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Feature Matching
#==============================================================================================
best_score = -1
best_match_data = None
best_matches = 0

for file_name in source_images:
    img_raw = cv2.imread(os.path.join(source_dir, file_name))
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_processed = preprocess(img_raw)
    kp_img, desc_img = sift.detectAndCompute(img_processed, None)

    if desc_img is not None:
        if desc_img.dtype != np.float32:
            desc_img = desc_img.astype(np.float32)

        # MATCH
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(check = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_target, desc_img, k=2)

        # FILTER MATCH

        matches_mask = []
        for _ in range(len(matches)):
            matches_mask.append([0, 0])

        for i, (m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[i] = [1, 0]
        current_match = 0

        score = 0
        if current_match > 0:
            valid_distances = [m.distance for i, (m, n) in enumerate(matches) if matches_mask[i][0] == 1]
            score = sum(valid_distances) / len(valid_distances) if valid_distances else 0

        if best_matches < current_match:
            best_matches = current_match
            best_match_data = {
                'image_data' : img_raw,
                'keypoint' : kp_img,
                'descriptor' : desc_img,
                'match' : matches,
                'matchesmask': matches_mask,
                'score' : score 
         }

# RESULT
#==============================================================================================
    # img_target_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_rgb = cv2.cvtColor(best_match_data['img'], cv2.COLOR_BGR2RGB)

    # DRAW MATCH
    img_res = cv2.drawMatchesKnn(
        img_raw, kp_target,
        best_match_data['image_data'], best_match_data['keypoint'],
        best_match_data['match'], None, matchesMask= best_match_data['matchesmask'],
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # SHOW MATCH
    plt.figure(figsize=(15, 7))
    plt.imshow(img_res)
    plt.title(f"Best Match Result : {best_match_data['score']}")
    plt.axis('off')
    plt.show()