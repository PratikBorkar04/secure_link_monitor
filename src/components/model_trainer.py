import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from src.utils import save_object, evaluate_models
from src.logger import logging
from src.exception import securelinkException
from sklearn.svm import SVC

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=100),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(max_iter=100)
            }
            model_report_df = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Find the best model based on test accuracy
            best_model_row = model_report_df.loc[model_report_df["Test Accuracy (%)"].idxmax()]
            best_model_name = best_model_row["Model"]
            best_model = models[best_model_name]

            logging.info(f"Model accuracies and confusion matrices:\n{model_report_df}")
            logging.info(f"Best model found on both training and testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            cm = best_model_row["Confusion Matrix"]
            logging.info(f"Confusion Matrix for best model:\n{cm}")
            return model_report_df
        except Exception as ex:
            raise securelinkException(ex, sys)

