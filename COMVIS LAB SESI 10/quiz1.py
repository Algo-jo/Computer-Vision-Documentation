import os
import json

import cv2
import cv2.data
import matplotlib.pyplot as plt
import numpy as np

TRAIN_DIR = os.path.join('Dataset', 'train')
TEST_DIR = os.path.join('Dataset', 'test')

MODEL_PATH = "MODEL_PATH"
LABEL_MAP_PATH = "label_map.json"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_img(img):
    img = cv2.resize(img, (100,100))
    img = cv2.equalizeHist(img)
    return img

def extract_face(img):
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    if len(faces) > 0:
        x,y,w,h = faces[0]
        face = img[y:y+h, x:x+w]
        return preprocess_img(face), (x,y,w,h)
    
    h,w = img.shape
    return preprocess_img(face), (0,0,w,h)

def load_dataset(dataset_dir):
    faces = []
    labels = []
    label_map = {}
    subjects = sorted(os.listdir(dataset_dir))
    
    for subject in subjects:
        subject_path = os.path.join(dataset_dir, subjects)
        if not os.path.isdir(subject_path): continue
        label_id = len(label_map)
        label_map[subject] = label_id
        
        for filename in os.listdir(subject_path):
            img_path = os.path.join(subject_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None: continue

            face, _ = extract_face(img)
            faces.append(face)
            labels.append(label_id)
        
        return faces, np.array(labels), label_map
    
def train_and_test_model():
    train_faces, train_label, label_map = load_dataset(TRAIN_DIR)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_faces, train_label)
    reverse_label_map = {value: key for key, value in label_map.items()}
    
    correct = 0
    total = 0
            
            
            