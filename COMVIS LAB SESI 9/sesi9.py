import os
import cv2
from scipy.spatial.distance import euclidean

image_dir = 'images' 
features = []
for filename in os.listdir(image_dir):
    if '.db' in filename: 
        continue
    
    image_name = filename.split('.')[0]
    image_bgr = cv2.imread(image_dir + '/' + filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    hist = cv2.calcHist([image_rgb], 
                        [0,1,2], 
                        None, 
                        [8,8,8],
                        [0,256,0,256,0,256]
                        )
    
    normalize_hist = cv2.normalize(hist, None)
    flatten_hist = normalize_hist.flatten()
    features.append((image_name, flatten_hist))
    
test_image = cv2.imread('images/Rivendell-001.png')
test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_hist = cv2.calcHist([test_image_rgb], 
                         [0,1,2], 
                         None, 
                         [8,8,8], 
                         [0,256,0,256,0,256])
test_normalized = cv2.normalize(test_hist, None)
test_flatten = test_normalized.flatten()

result = []
for name, distance in features:
    dist = euclidean(test_flatten, distance)
    result.append((dist, name))

sorted_result = sorted(result)
best_distance, best_name = sorted_result[0]
best_image = cv2.imread(f'images/{best_name}.png')
cv2.imshow('best match', best_image)
cv2.waitKey(0)

for distance, name in sorted_result:
    print(f'{name}: {distance}')
