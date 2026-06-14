import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

TRAIN_DIR = "COMVIS LAB SESI 10\Dataset\\train"
TEST_DIR = "COMVIS LAB SESI 10\Dataset\\test"
MODEL_PATH = "MODEL_PATH"
LABEL_MAP_PATH = "label_map.json"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocessing(img):
    img = cv2.resize(img, (100,100))
    img = cv2.equalizeHist(img)
    return img

def extract_face(img):
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    if len(faces) > 0:
        x,y,w,h = faces[0]
        face = img[y:y+h, x:x+w]
        return preprocessing(face), (x,y,w,h)
    h,w = img.shape
    return preprocessing(img), (0,0,w,h)

def load_dataset(DATASET_DIR):
    faces = []
    labels = []
    label_map = {}
    subjects = sorted(os.listdir(DATASET_DIR))
    for subject in subjects:
        subject_path = os.path.join(DATASET_DIR, subject)
        if not os.path.isdir(subject_path):
            continue
        label_id = len(label_map)
        label_map[subject] = label_id

        for filename in os.listdir(subject_path):
            img_path = os.path.join(subject_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            face, _ = extract_face(img)
            faces.append(face)
            labels.append(label_id)
    
    return faces, np.array(labels), label_map

def train_and_test_model():
    train_faces, train_label, label_map = load_dataset(TRAIN_DIR)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_faces, train_label)
    reversed_label_map = {value: key for key, value in label_map.items()}

    correct = 0
    total = 0

    subjects = sorted(os.listdir(TEST_DIR))
    for subject in subjects:
        subject_path = os.path.join(TEST_DIR, subject)
        if not os.path.isdir(subject_path):
            continue
        true_label = label_map[subject]

        for filename in os.listdir(subject_path):
            img_path = os.path.join(subject_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            face, _ = extract_face(img)

            predicted_label, confidence = recognizer.predict(face)
            predicted_subject = reversed_label_map[predicted_label]

            if predicted_label == true_label:
                correct+=1
            total +=1

            print(f"{img_path}: {subject}" 
            f"{predicted_subject}: {confidence}")
    
    acc = correct/total * 100
    print(acc)
    recognizer.write(MODEL_PATH)
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_map, f)

def predict_img(img_path):
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    reversed_label_map = {value: key for key, value in label_map.items()}

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    face, box = extract_face(img)
    predicted_label, confidence = recognizer.predict(face)
    predicted_subject = reversed_label_map[predicted_label]

    x,y,w,h = box
    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(
        display_img,
        (x,y),
        (x+w, y+h),
        (0,255,0),
        2
    )

    cv2.putText(
        display_img,
        f"{predicted_subject}: {confidence}",
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

    print(f"{img_path} predicted as {predicted_subject} with confidence {confidence}")
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def main_menu():
    while True:
        choice = input("Select an option: ")
        if choice == '1':
            train_and_test_model()
        elif choice == '2':
            img = input("Img Path: ")
            predict_img(img)
        elif choice == '3':
            break
main_menu()
    