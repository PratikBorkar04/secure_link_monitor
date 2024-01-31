import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from src.utils import save_object, evaluate_models
from src.logger import logging
from src.exception import securelinkException
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class ModelTrainerConfig:
    # Configuration class for the model trainer
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        # Initializing the model trainer configuration
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            # Splitting the input data into features and labels for both training and testing
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            # Dictionary of models to train
            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB()
            }
            # Evaluating each model and storing their reports
            model_report_df = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Find the best model based on test accuracy
            best_model_row = model_report_df.loc[model_report_df["Test Accuracy (%)"].idxmax()]
            best_model_name = best_model_row["Model"]
            best_model = models[best_model_name]

            logging.info(f"Model accuracies and confusion matrices:\n{model_report_df}")
            logging.info(f"Best model found on both training and testing dataset: {best_model_name}")

            # Saving the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            cm = best_model_row["Confusion Matrix"]
            logging.info(f"Confusion Matrix for best model:\n{cm}")

            # Return the model report dataframe
            return model_report_df
        except Exception as ex:
            # Handling exceptions and logging
            raise securelinkException(ex, sys)

