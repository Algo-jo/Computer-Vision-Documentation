import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. PREPROCESSING SECTION [cite: 14, 35]
# ==========================================================================================
# [Listing Image - 10 pts] [cite: 14, 34]
source_dir = r'Documentation\images\source'
target_dir = r'Documentation\images\target'

# Mengambil semua gambar kandidat (Student Images) 
student_images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
# Mengambil satu target (Target Object Image) 
target_files = [f for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

if not target_files or not student_images:
    print("Folder target atau source kosong!")
    exit()

# [Load Image - 10 pts] [cite: 14]
target_path = os.path.join(target_dir, target_files[0])
img_target_raw = cv2.imread(target_path)

# [Smoothing Image - 10 pts] [cite: 14, 36]
# Spesifikasi SEVeros: Grayscale -> Gaussian Blur (3x3, SigmaX: 0) [cite: 36, 37, 38, 39, 40]
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0) 
    return blur

img_target_proc = preprocess_image(img_target_raw)

# 2. FEATURE DETECTION [cite: 14, 41]
# ==========================================================================================
# Menggunakan SIFT sesuai anjuran SEVeros 
detector = cv2.SIFT_create()
kp_target, desc_target = detector.detectAndCompute(img_target_proc, None)

# Konversi tipe data agar kompatibel dengan FLANN 
if desc_target is not None and desc_target.dtype != np.float32:
    desc_target = desc_target.astype(np.float32) 

best_score = -1
best_match_data = None

# 3. FEATURE MATCHING [cite: 14, 43]
# ==========================================================================================
for file_name in student_images:
    img_student_raw = cv2.imread(os.path.join(source_dir, file_name))
    img_student_proc = preprocess_image(img_student_raw)
    
    kp_student, desc_student = detector.detectAndCompute(img_student_proc, None)
    
    if desc_student is not None:
        if desc_student.dtype != np.float32:
            desc_student = desc_student.astype(np.float32) 

        # [Match - 15 pts] Menggunakan FLANN Algorithm [cite: 14, 44]
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_target, desc_student, k=2)

        # [Filter Match - 20 pts] Lowe's Ratio Test [cite: 14, 45]
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append([m])
        
        # Menentukan best match result [cite: 45]
        if len(good_matches) > best_score:
            best_score = len(good_matches)
            best_match_data = {
                'img': img_student_raw,
                'kp': kp_student,
                'matches': good_matches,
                'name': file_name
            }

# 4. RESULT [cite: 14, 28]
# ==========================================================================================
if best_match_data:
    # Output harus dalam RGB 
    img_target_rgb = cv2.cvtColor(img_target_raw, cv2.COLOR_BGR2RGB)
    img_student_rgb = cv2.cvtColor(best_match_data['img'], cv2.COLOR_BGR2RGB)

    # [Draw Match - 10 pts] Sertakan semua garis kecocokan [cite: 14, 46]
    res_img = cv2.drawMatchesKnn(
        img_target_rgb, kp_target,
        img_student_rgb, best_match_data['kp'],
        best_match_data['matches'], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # [Show Match - 10 pts] [cite: 14]
    plt.figure(figsize=(15, 7))
    plt.imshow(res_img)
    plt.title(f"Best Match: {best_match_data['name']} ({best_score} points)")
    plt.axis('off')
    plt.show()