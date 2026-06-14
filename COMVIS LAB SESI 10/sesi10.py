import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

DATASET = "./Dataset/"
TRAIN_SET = DATASET + "train"
TEST_SET = DATASET + "test"
MODEL_PATH = "./haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(MODEL_PATH)
if face_cascade.empty():
    raise RuntimeError("Face Cascade Model Not Found")

def read_img_grayscale(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image doesn't exists")
    return img

def read_img_from_folder(folder_path):
    result = []
    for file in os.listdir(folder_path):
        result.append(
            read_img_grayscale(
                os.path.join(folder_path, file)
            )
        )
    return result

def preprocess_img(face_img, target_size=(100,100)):
    resized = cv2.resize(face_img, target_size)
    equalize = cv2.equalizeHist(resized)
    return equalize

# Test
# img_test = read_img_grayscale("./Dataset/test/s1/8.pgm")
# preprocess_img(img_test)

def detect_single_face(img):
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=4,                    # 2^n
        minSize=(30,30)
    )
    
    if len(faces) == 1:
        x, y, w, h = faces[0]
        face_region = img[y:y+h, x:x+w]
        detected_box = (x, y, w, h)
    else:
        h, w = img.shape[:2]
        face_region = img
        detected_box = (0, 0, h, w)
    
    preprocess_face = preprocess_img(face_region)
    return preprocess_face, detected_box, len(faces)

# Test
detect_single_face(
    read_img_grayscale("./Dataset/test/s1/10.pgm")
)

def load_imgs(folder, ):
    faces = []
    labels = []
    img = []
    img_path = []
    detection_result = []

    subject_folder = [
        folder for folder in os.listdir(DATASET + folder)
    ]
    
    subject_folder = subject_folder[:10:]
    labels = subject_folder
    
    for label in labels:
        read = read_img_from_folder(
            os.path.join(DATASET, folder, label)
        )
        
        face, box, length = detect_single_face(read)
        if length != 1:
            continue
        
        labels.append(label)
        img.append(read)
        img_path(os.path.join(DATASET, folder, label))
        faces.append(face)
        
        detection_result.append({
            "path": os.path.join(DATASET, folder, label),
            "subject": label,
            "box": box,
            "face_length": length
        })
        
    return faces, labels, img_path, detection_result
   
for path in load_imgs("test"):
    print(path)
 
def trainAndTest():
    pass

def predict():
    pass

def main_loop():
    while True:
        print("""
              1. Train and Test Model
              2. Predict
              3. Exit
              """)
        user_input = int(input("Enter Choice: "))
        
        if user_input == 1:
            trainAndTest()
        elif user_input == 2:
            predict()
        else:
            return