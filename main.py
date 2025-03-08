from assets.imports import *
from assets.data_preprocessing import preprocess_data
from assets.model_training import train_lr, train_rf, train_svm

def run_ml_pipeline():
    # Preprocess the data
    X, y = preprocess_data() 

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate Logistic Regression
    print("Training Logistic Regression...")
    lr_accuracy, lr_report = train_lr(X_train, y_train, X_test, y_test)
    print(f"Logistic Regression Accuracy: {lr_accuracy}")
    print(f"Logistic Regression Report: {lr_report}\n")

    # Train and evaluate Random Forest
    print("Training Random Forest...")
    rf_accuracy, rf_report = train_rf(X_train, y_train, X_test, y_test)
    print(f"Random Forest Accuracy: {rf_accuracy}")
    print(f"Random Forest Report: {rf_report}\n")

    # Train and evaluate SVM
    print("Training Support Vector Machine (SVM)...")
    svm_accuracy, svm_report = train_svm(X_train, y_train, X_test, y_test)
    print(f"SVM Accuracy: {svm_accuracy}")
    print(f"SVM Report: {svm_report}\n")

if __name__ == "__main__":
    run_ml_pipeline()