
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV

import numpy as np
import joblib

#Load data
Dataset=np.load("Dataset.npy")
Labels=np.load("Labels.npy")

X_test=np.load("Test_Dataset.npy")
y_test=np.load("Test_Labels.npy")

#Used to create weights to fight undersampling so we can achieve better classification
def optimize_data():

    weights = compute_class_weight("balanced", classes=np.unique(Labels), y=Labels)
    weights_dictionary= dict(zip(np.unique(Labels), weights))
    return weights_dictionary

#Dummy classifier.The fitted model is saved in the project's directory
def dummy_classifier( X_train, y_train):

    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(X_train, y_train)
    y_dummy_pred = dummy_clf.predict(X_test)
    print(classification_report(y_test, y_dummy_pred))

    joblib.dump(dummy_clf, "dummy_stratified.pkl")


#Svm classifier.The fitted model is saved in the project's directory
def svm_classifier(X_train, X_test, y_train, y_test,weights_dictionary):

    svm_model = SVC(kernel='linear', C=1.0, class_weight=weights_dictionary)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    accuracy= accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(accuracy)
    print(classification_report(y_test, y_pred))
    print(cm)

    joblib.dump(svm_model,"svm_linear_weighted.pkl")


#Random forest classifier.The fitted model is saved in the project's directory
def random_forest(X_train, y_train,weights_dictionary):

    rf = RandomForestClassifier(n_estimators=100,class_weight=weights_dictionary)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(rf, "random_forest_weighted.pkl")



#Preprocesses the data before being fed to the classifiers
#It calls optimize_data() and also shuffles the data
def data_preprocessing(Dataset,Labels):

    weights_dictionary=optimize_data()
    combined = np.column_stack((Dataset, Labels))

    np.random.seed(69)
    np.random.shuffle(combined)
    Dataset = combined[:, :-1]
    Labels = combined[:, -1]

    return weights_dictionary,Dataset,Labels


#Tunes the hyperparameters for svm.In case of a crash it saves  the current progress
#If the function runs without any issues, it saves the best model in the project's directory
def tune_hyperparameters_svm(X_train, y_train, weights_dictionary):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 0.01, 0.001]
    }
    grid_search = GridSearchCV(SVC(kernel='rbf', class_weight=weights_dictionary), param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=2)

    try:
        grid_search.fit(X_train, y_train)
        print("GridSearch Completed!")


        joblib.dump(grid_search, "gridsearch_checkpoint.pkl")
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv("gridsearch_results.csv", index=False)


        best_model = grid_search.best_estimator_
        joblib.dump(best_model, "best_svm_model.pkl")
        print("Best model saved!")
        print("Results saved! Best Params:", grid_search.best_params_)
        return grid_search.best_params_

    except KeyboardInterrupt:
        print(" Interrupted! Saving partial results...")
        joblib.dump(grid_search, "gridsearch_checkpoint.pkl")
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv("gridsearch_results.csv", index=False)
        print(" Partial results saved! Resume training using `gridsearch_checkpoint.pkl`.")
        raise

#Tunes the hyperparameters of random forest.In case of a crash it saves  the current progress
#If the function runs without any issues, it saves the best model in the project's directory
def tune_hyperparameters_rf(X_train, y_train):

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(RandomForestClassifier(class_weight=None),param_grid, cv=2, scoring='f1_weighted',n_jobs=-1, verbose=2)

    try:
        grid_search.fit(X_train, y_train)
        print(" GridSearch Completed!")

        # Save GridSearch progress
        joblib.dump(grid_search, "gridsearch_rf_checkpoint.pkl")
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv("gridsearch_rf_results.csv", index=False)

        # Save only the best model
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, "best_rf_model.pkl")
        print("Best model saved!")
        print("Results saved! Best Params:", grid_search.best_params_)
        return grid_search.best_params_

    except KeyboardInterrupt:
        print(" Interrupted! Saving partial results...")
        joblib.dump(grid_search, "gridsearch_rf_checkpoint.pkl")
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv("gridsearch_rf_results.csv", index=False)
        print(" Partial results saved! Resume training using `gridsearch_rf_checkpoint.pkl`.")
        raise 


#The "main" function of this python file.As input it recieves a string
#and calls the appopriate function
def call_classifier(classifier):
    weights_dictionary, X_train, y_train = data_preprocessing(Dataset, Labels)

    if classifier=="svm":
        print('Svm!')
        svm_classifier(X_train, X_test, y_train, y_test, weights_dictionary)
    elif classifier=="dummy":
        print("Dummy!")
        dummy_classifier(X_train, y_train)
    elif classifier=="forest":
        print("Random Forest!")
        random_forest(X_train, y_train,weights_dictionary)
    elif classifier=="svmHyper":
        print("Tuning Svm's hyperparameters!")
        tune_hyperparameters_svm(X_train, y_train, weights_dictionary)
    else:
        print("Tuning Random Forest hyperparameters!")
        tune_hyperparameters_rf(X_train, y_train)






