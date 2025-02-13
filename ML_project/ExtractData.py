

import os
import cv2
import numpy as np


#The dataset can be downloaded from https://www.kaggle.com/datasets/msambare/fer2013

emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
train_dataset_path="FER2013_PATH/train" #replace with your path
test_dataset_path="FER2013_PATH/test" #replace with your path

label_dict = {emotion: idx for idx, emotion in enumerate(emotion_labels)}

#Loads the data
def get_data():

    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for emotion in emotion_labels:

        folder_path=os.path.join(train_dataset_path,emotion)

        #For every image in train folder, resize the image turn it into greyscale and extract the label
        for img_name in os.listdir(folder_path):

            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            train_images.append(img)
            train_labels.append(label_dict[emotion])

        folder_path=os.path.join(test_dataset_path,emotion)

        # For every image in test folder, resize the image turn it into greyscale and extract the label
        for img_name in os.listdir(folder_path):

            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            test_images.append(img)
            test_labels.append(label_dict[emotion])


    #Transform to np arrays and set images to type float32
    train_images_nu= np.array(train_images, dtype=np.float32)
    train_labels_nu = np.array(train_labels)

    test_images_nu= np.array(test_images, dtype=np.float32)
    test_labels_nu= np.array(test_labels)

    return train_images_nu,train_labels_nu,test_images_nu,test_labels_nu






