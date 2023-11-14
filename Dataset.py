import numpy as np
import os 
import matplotlib as plt
import cv2
import random
import pickle

directory = r'Dataset/'
type = ['Satvik', 'Nancy', 'Marco', 'Sukrit', 'Ari_big_sig']

data = []

for t in type:
    path = os.path.join(directory, t)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        print("Image path:", img_path)  # Add this line for debugging
        label = type.index(t)
        arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if arr is not None:
            new_arr = cv2.resize(arr, (224, 224))
            data.append([new_arr, label])
        else:
            print("Failed to load image:", img_path)

data[0][1]

X = []
Y = []

for feature, label in data:
    X.append(feature)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(Y, open('Y.pkl', 'wb'))