import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

train_path = 'Dataset\Train'
test_path = 'Dataset\Test'

train_labels = []
train_descriptors = []
descriptors = []
image_desc_map = [] # Store Desc of each Image

siftx = cv2.xfeatures2d.SIFT_create()
sift = cv2.SIFT_create()

categories = [
    "gpu", 
    "headphone", 
    "keyboard", 
    "mice"
]

# Index, Content
for label, category in enumerate(categories):
    folder = os.path.join(train_path, test_path)
    
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), 0)
        
    if img is None:
        continue
    
    kp, des = sift.detectAndCompute(img, None)
    if des is None:
        continue
    train_descriptors.extend(des)
    image_desc_map.append((des, label))

k = 50
kmeans = KMeans(
    n_clusters=k,
    n_init=10,
    random_state=45
)
kmeans.fit(train_descriptors)

def build_histogram(descriptors, kmeans, k):
    histogram = np.zeros(k)
    
    visual_words = np.predict(descriptors)
    for vw in visual_words:
        histogram[vw] += 1
    
    return histogram

X_train = []
y_train = []

# X_train : Histogram of descriptor
# y_train : Index from label (gpu, headphone, ...)
for des, label in image_desc_map:
    X_train.append(
        build_histogram(des, kmeans, k)
    )
    y_train.append(label)
    
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

svm = LinearSVC(random_state=45, max_iter=10000)
svm.fit(X_train_scaled, y_train)

plt.figure(figsize=(12,8))

test_images = sorted(
    os.listdir(test_path)
)

for i, filename in test_images:
    img_bgr = cv2.imread(
        os.path.join(test_path, filename)
    )
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    kp, des = sift.detectAndCompute(img_gray, None)
    
    if des is None:
        continue
    hist = build_histogram(des, kmeans, k)
    hist = hist.reshape(1, -1)
    hist_scaled = scaler.fit_transform(hist)
    
    prediction = svm.predict(hist_scaled)[0]
    label_text = categories[prediction]
    
    plt.subplot(2, 3, i+1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
plt.tight_layout()
plt.show()
    