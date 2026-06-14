import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# PREPROCESSING
#===========================================================================
# Listing Images
source_dir = r'Documentation\images\source'
source_images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Load Image
img_raw = cv2.imread(r'Documentation\images\target\hina.png')

# Smoothing Image
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return blur

img_processed = preprocess(img_raw)

# FEATURE DETECTION
#===========================================================================

sift = cv2.SIFT_create()
kp_target, desc_target = sift.detectAndCompute(img_processed, None)
if desc_target is not None:
    if desc_target.dtype != np.float32:
        desc_file = desc_target.astype(np.float32)

best_score = -1
best_match_data = None

# FEATURE MATCHING
#===========================================================================
for file in source_images:
    file_raw = cv2.imread(os.path.join(source_dir, file))
    file_processed = preprocess(file_raw)
    kp_file, desc_file = sift.detectAndCompute(file_processed, None)
    if desc_file is not None:
        if desc_file.dtype != np.float32:
            desc_file = desc_file.astype(np.float32)

        # MATCH
        index_params = dict(algorithm=1, tree=5)
        search_params = dict(check=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        match = flann.knnMatch(desc_target, desc_file, k=2)

        good_matches = []
        for m, n in match:
            if m.distance < 0.7 * n.distance:
                good_matches.append([m])

        if len(good_matches) > best_score:
            best_score = len(good_matches)
            best_match_data = {
                'img': file_raw,
                'kp': kp_file,
                'name': file,
                'matches': good_matches
            }

# RESULT
if best_match_data:
    img_target_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(best_match_data['img'], cv2.COLOR_BGR2RGB)

    # DRAW MATCH
    img_res = cv2.drawMatchesKnn(
        img_target_rgb, kp_target,
        img_rgb, best_match_data['kp'],
        best_match_data['matches'], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # SHOW MATCH
    plt.figure(figsize=(15, 7))
    plt.imshow(img_res)
    plt.title(f"Best Match Result {best_match_data['name']}: {best_score}")
    plt.axis('off')
    plt.show()