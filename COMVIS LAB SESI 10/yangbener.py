import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

TRAIN_DIR = "COMVIS LAB SESI 10\Dataset\\train"
TEST_DIR = "COMVIS LAB SESI 10\Dataset\\test"


MODEL_PATH = "MODEL_PATH"
LABEL_MAP_PATH = "label_map.json"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_image(image):
    image = cv2.resize(image, (100, 100))
    image = cv2.equalizeHist(image)
    return image


def extract_face(image):
    faces = face_cascade.detectMultiScale(image, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        return preprocess_image(face), (x, y, w, h)

    h, w = image.shape
    return preprocess_image(image), (0, 0, w, h)


def load_dataset(dataset_dir):
    faces = []
    labels = []
    label_map = {}

    subjects = sorted(os.listdir(dataset_dir))

    for subject in subjects:
        subject_path = os.path.join(dataset_dir, subject)

        if not os.path.isdir(subject_path):
            continue

        label_id = len(label_map)
        label_map[subject] = label_id

        for filename in os.listdir(subject_path):
            image_path = os.path.join(subject_path, filename)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            face, _ = extract_face(image)

            faces.append(face)
            labels.append(label_id)

    return faces, np.array(labels), label_map


def train_and_test_model():
    train_faces, train_labels, label_map = load_dataset(TRAIN_DIR)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_faces, train_labels)

    reverse_label_map = {value: key for key, value in label_map.items()}

    correct = 0
    total = 0

    for subject in sorted(os.listdir(TEST_DIR)):
        subject_path = os.path.join(TEST_DIR, subject)

        if not os.path.isdir(subject_path):
            continue

        true_label = label_map[subject]

        for filename in os.listdir(subject_path):
            image_path = os.path.join(subject_path, filename)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if image is None:
                continue

            face, _ = extract_face(image)

            predicted_label, confidence = recognizer.predict(face)

            predicted_subject = reverse_label_map[predicted_label]

            print(
                f"{image_path} | Actual: {subject} | "
                f"Predicted: {predicted_subject} | Confidence: {confidence:.2f}"
            )

            if predicted_label == true_label:
                correct += 1

            total += 1

    accuracy = correct / total * 100
    print(f"\nAverage Accuracy: {accuracy:.2f}%")

    recognizer.write(MODEL_PATH)

    with open(LABEL_MAP_PATH, "w") as file:
        json.dump(label_map, file)

    print("\nModel saved successfully.")


def predict_image(image_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABEL_MAP_PATH, "r") as file:
        label_map = json.load(file)

    reverse_label_map = {value: key for key, value in label_map.items()}

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face, box = extract_face(image)

    predicted_label, confidence = recognizer.predict(face)

    predicted_subject = reverse_label_map[predicted_label]

    x, y, w, h = box

    display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(
        display_image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_image,
        f"{predicted_subject} | {confidence:.2f}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    print(f"{image_path} detected as {predicted_subject}")
    print(f"Confidence: {confidence:.2f}")

    plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def main_menu():
    while True:
        print("\n1. Train and test model.")
        print("2. Predict.")
        print("3. Exit.")

        choice = input("Enter your choice: ")

        if choice == "1":
            train_and_test_model()

        elif choice == "2":
            image_path = input(
                "Enter the path to the image for testing: "
            )
            predict_image(image_path)

        elif choice == "3":
            print("Program terminated.")
            break

        else:
            print(
                "Invalid menu choice. Please choose 1, 2, or 3."
            )


main_menu()