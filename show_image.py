import numpy as np
import pickle
import cv2

def pickle_retrieve(name):
    pickle_in = open(name, 'rb')
    file = pickle.load(pickle_in)
    return file

dataset_train_features = pickle_retrieve('dataset_train_features.pickle')
for i in range(0,20):
    img = []
    img.append(dataset_train_features[i])
    img = np.array(img, dtype='uint8')
    img = img.reshape((48,48))
    print(img.shape)

    cv2.imshow('image', img)  # show the image
    cv2.waitKey(0)