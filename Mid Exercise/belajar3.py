import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# PREPROCESS
source_dir = r'PRACTICE\images\source'
source_images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

img_raw = cv2.imread(r'PRACTICE\images\target\hina.png')

def precrocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    return blur

img_processed = precrocess(img_raw)

# FEATURE DETECTION
sift = cv2.SIFT_create()
kp_target, desc_target = sift.detectAndCompute(img_processed, None)
if desc_target is not None:
    if desc_target.dtype != np.float32:
        desc_target = desc_target.astype(np.float32)

best_score = -1
best_match_data = None

# FEATURE MATCHING
for file in source_images:
    file_raw = cv2.imread(os.path.join(source_dir, file))
    file_processed = precrocess(file_raw)

    kp_file, desc_file = sift.detectAndCompute(file_processed, None)
    if desc_file is not None:
        if desc_file.dtype != np.float32:
            desc_file = desc_file.astype(np.float32)
        
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(check=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_target, desc_file, k=2)

        matches_mask = []
        for _ in range(len(matches)):
            matches_mask.append([0,0])
        
        for i, (m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[i] = [1,0]

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append([m])

        if len(good_matches) > best_score:
            best_score = len(good_matches)
            best_match_data = {
                'img': file_raw,
                'kp': kp_file,
                'name': file,
                'match': good_matches
            }

# RESULT
if best_match_data:
    img_target_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(best_match_data['img'], cv2.COLOR_BGR2RGB)

    img_res = cv2.drawMatchesKnn(
        img_target_rgb, kp_target,
        img_rgb, best_match_data['kp'],
        best_match_data['match'], None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )  
    plt.figure(figsize=(12, 7))
    plt.imshow(img_res)
    plt.title(f"Best Match {best_match_data['name']}: {best_score}")
    plt.show()

