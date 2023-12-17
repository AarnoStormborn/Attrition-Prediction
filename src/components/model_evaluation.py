import os
import sys
import json
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import plot_confusion_matrix, get_latest_run_id

from sklearn.metrics import (classification_report, 
                             accuracy_score, f1_score,
                             confusion_matrix)

import mlflow
import mlflow.sklearn

@dataclass
class ModelEvaluationConfig:
    experiment_id: str
    experiment_dir_path: str

class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config

    def model_evaluator(self, test_set:pd.DataFrame) -> None:
        try:

            X_test, y_test = test_set

            latest_run_id = get_latest_run_id(experiment_name=self.config.experiment_name)

            model_uri = self.config.experiment_dir_path.format(str(latest_run_id))
            
            loaded_model = mlflow.sklearn.load_model(model_uri)

            y_pred = loaded_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("test - accuracy", accuracy)
            mlflow.log_metric("test - f1_score", f1)

            logging.info("Results saved")

            clf_report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_table(clf_report, "classification_report.json")

            logging.info("Classification Report saved")

            cm = confusion_matrix(y_test, y_pred)
            cm_fig = plot_confusion_matrix(cm)

            mlflow.log_figure(cm_fig, "confusion_matrix.png")

            logging.info("Confusion Matrix Saved")

        except Exception as e:
            logging.error(CustomException(e, sys))

        finally:
            mlflow.end_run()
