import os
import sys
import json
import joblib
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import plot_confusion_matrix

from sklearn.metrics import (classification_report, 
                             accuracy_score, f1_score,
                             confusion_matrix)

@dataclass
class ModelEvaluationConfig:
    root_dir: str
    model_path: str

class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config

    def model_evaluator(self, test_set:pd.DataFrame) -> None:
        try:
            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)

            X_test, y_test = test_set

            logging.info("Loading Model")
            loaded_model = joblib.load(self.config.model_path)

            y_pred = loaded_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logging.info("Saving Results...")
            pd.DataFrame({
                'accuracy_score': [accuracy],
                'f1_score': [f1]
            }).to_csv(os.path.join(root_dir, "results.csv"))

            logging.info("Results saved")

            clf_report = classification_report(y_test, y_pred, output_dict=True)
            with open(os.path.join(root_dir, "classification_report.json"), "w") as f:
                json.dump(clf_report, f, indent=4)

            logging.info("Classification Report saved")

            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(root_dir, cm)

            logging.info("Confusion Matrix Saved")

        except Exception as e:
            logging.error(CustomException(e, sys))
