from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import joblib

#Load data
X_test=np.load("Test_Dataset.npy")
y_test=np.load("Test_Labels.npy")

Labels=np.load("Labels.npy")

#The main function of this python file.Gets as input the model's name, loads and uses it to make predictions
#The results are input for the model_perfomance function
def get_data_from_model(model_name):

    model= joblib.load(model_name)
    y_pred = model.predict(X_test)
    models_perfomance(model,y_pred)


#Gets as input a saved model and outputs a plot with key metrics visualized
def models_perfomance(model, y_pred):
    report_dict = classification_report(y_test, y_pred, output_dict=True)


    labels = list(report_dict.keys())[:-3]
    metrics = ["precision", "recall", "f1-score"]

    report_matrix = np.array([[report_dict[label][metric] for metric in metrics] for label in labels])


    cm = confusion_matrix(y_test, y_pred)


    accuracy = accuracy_score(y_test, y_pred)


    fig, axes = plt.subplots(1, 2, figsize=(15, 5))


    sns.heatmap(report_matrix, annot=True, cmap="Blues", fmt=".2f", xticklabels=metrics, yticklabels=labels, linewidths=0.5, ax=axes[0])
    axes[0].set_xlabel("Metrics")
    axes[0].set_ylabel("Class Labels")
    axes[0].set_title("Classification Report Heatmap")

    sns.heatmap(cm, annot=True, fmt='d', cmap="Oranges", linewidths=0.5, xticklabels=sorted(np.unique(y_test)), yticklabels=sorted(np.unique(y_test)) , ax=axes[1])
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_title("Confusion Matrix")

    plt.figtext(0.5, 0.02, f"Accuracy: {accuracy:.4f}", fontsize=14, color="black", ha="center", bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


#Used to create a plot depicting the class distribution
def class_distribution():
    unique_classes, counts = np.unique(Labels, return_counts=True)

    plt.figure(figsize=(8, 5))
    plt.bar(unique_classes, counts, color='skyblue', edgecolor='black')

    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.xticks(unique_classes)

    for i, count in enumerate(counts):
        plt.text(unique_classes[i], count, str(count), ha='center', va='bottom', fontsize=12)
    plt.show()








