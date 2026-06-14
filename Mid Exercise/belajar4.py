import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

source_dir = r'PRACTICE\images\source'
source_images = [f for f in os.listdir(source_dir)]
target_raw = cv2.imread(r'PRACTICE\images\target\hina.png')

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    return blur
target_processed = preprocess(target_raw)

sift = cv2.SIFT_create()
kp_target, desc_target = sift.detectAndCompute(target_processed, None)
if desc_target is not None:
    if desc_target.dtype != np.float32:
        desc_target = desc_target.astype(np.float32)

best_score = -1
best_match_data = None

for file in source_images:
    file_raw = cv2.imread(os.path.join(source_dir, file))
    file_processed = preprocess(file_raw)

    kp_file, desc_file = sift.detectAndCompute(file_processed, None)
    if desc_file is not None:
        if desc_file.dtype != np.float32:
            desc_file = desc_file.astype(np.float32)
        
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(check=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_target, desc_file, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append([m])
        
        if best_score < len(good_matches):
            best_score = len(good_matches)
            best_match_data = {
                'img': file_raw,
                'kp': kp_file,
                'name': file,
                'matches': good_matches
            }

if best_match_data:
    img_target_rgb = cv2.cvtColor(target_raw, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(best_match_data['img'], cv2.COLOR_BGR2RGB)

    img_res = cv2.drawMatchesKnn(
        img_target_rgb, kp_target,
        img_rgb, best_match_data['kp'],
        best_match_data['matches'], None,
        matchColor=[255,0,0],
        singlePointColor=[0,0,255],
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    plt.figure(figsize=(10,5))
    plt.imshow(img_res)
    plt.title(f"{best_score}")
    plt.show()
else:
    print("No Image found")