
import numpy as np
from ExtractData import get_data
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
import cv2


#This is the "main" function of this python file.Calling it starts a chain reaction of functions
#calculating the HOG and LBP features while also performing PCA to reduce dimensionality
#The features are saved in the project's directory for easier usability
def extract_features():

    train_dataset=[]
    test_dataset=[]

    train_images,train_labels,test_images,test_labels=get_data()


    for image in train_images:
        lpb_hist=lbp_features(image)
        hog=hog_features(image)
        combined_features = np.hstack((hog, lpb_hist))
        train_dataset.append(combined_features)

    for image in test_images:
        lpb_hist=lbp_features(image)
        hog=hog_features(image)
        combined_features = np.hstack((hog, lpb_hist))
        test_dataset.append(combined_features)

    final_train_dataset,final_test_dataset=pca(np.vstack(train_dataset),np.vstack(test_dataset))
    save_datasets_labels(final_train_dataset, final_test_dataset,train_labels,test_labels)



#Calculates HOG features with the optimized parameters
def hog_features(image):

    hog_features= hog( image,pixels_per_cell=(8, 8),cells_per_block=(2, 2),orientations=9,visualize=False,block_norm='L2-Hys'
    )
    return hog_features

#Calculates LBP features with the optimized parameters
def lbp_features(image):
    lbp = local_binary_pattern(image.astype("uint8"), P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

#PCA is applied to both train/test numpy feature arrays to achieve the same number of principal components
def pca(train_dataset,test_dataset):
    pca = PCA(n_components=0.95)
    final_train_dataset = pca.fit_transform(train_dataset)
    final_test_dataset=pca.transform(test_dataset)
    return final_train_dataset,final_test_dataset


#Saves the features/Labels
def save_datasets_labels(final_train_dataset,final_test_dataset,train_labels,test_labels):

    np.save("Dataset.npy",final_train_dataset)
    np.save("Test_Dataset.npy",final_test_dataset)
    np.save("Labels.npy",train_labels)
    np.save("Test_Labels.npy",test_labels)















