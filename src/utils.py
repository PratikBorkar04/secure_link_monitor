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
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as ex:
        raise securelinkException(ex, sys)
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        results = []

        for model_name, model in models.items():
            logging.info(f"Training and evaluating model: {model_name}")

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = np.mean(y_train_pred == y_train) * 100  # Convert to percentage
            test_accuracy = np.mean(y_test_pred == y_test) * 100  # Convert to percentage

            cm = confusion_matrix(y_test, y_test_pred)

            results.append({
                "Model": model_name,
                "Train Accuracy (%)": train_accuracy,
                "Test Accuracy (%)": test_accuracy,
                "Confusion Matrix": cm
            })

            logging.info(f"Completed {model_name}: Train Accuracy = {train_accuracy}%, Test Accuracy = {test_accuracy}%")

        return pd.DataFrame(results)

    except Exception as ex:
        raise securelinkException(ex, sys)