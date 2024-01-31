import numpy as np
import pandas as pd
import os
from src.exception import securelinkException
import pickle
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    Parameters:
    file_path (str): The path where the object should be saved.
    obj (object): The Python object to be saved.
    """
    try:
        # Extract directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
       
        # Open the file in write-binary mode and save the object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as ex:
        # Raise custom exception in case of error
        raise securelinkException(ex, sys)
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models on the given training and test data.

    Parameters:
    X_train (array): Training data features
    y_train (array): Training data labels
    X_test (array): Test data features
    y_test (array): Test data labels
    models (dict): Dictionary of models to be evaluated

    Returns:
    DataFrame: A Pandas DataFrame containing the evaluation results of each model
    """
    try:
        results = []

        for model_name, model in models.items():
            logging.info(f"Training and evaluating model: {model_name}")

            # Train the model
            model.fit(X_train, y_train)

            # Predict on training and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate training and test accuracies
            train_accuracy = np.mean(y_train_pred == y_train) * 100  # Convert to percentage
            test_accuracy = np.mean(y_test_pred == y_test) * 100  # Convert to percentage

            # Generate confusion matrix for test predictions
            cm = confusion_matrix(y_test, y_test_pred)

            # Append the results for the current model
            results.append({
                "Model": model_name,
                "Train Accuracy (%)": train_accuracy,
                "Test Accuracy (%)": test_accuracy,
                "Confusion Matrix": cm
            })

            logging.info(f"Completed {model_name}: Train Accuracy = {train_accuracy}%, Test Accuracy = {test_accuracy}%")
        
        # Return the results as a Pandas DataFrame
        return pd.DataFrame(results)

    except Exception as ex:
        # Raise custom exception in case of error
        raise securelinkException(ex, sys)