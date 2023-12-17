import os
import sys
import joblib

from pandas import DataFrame
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import get_latest_run_id, read_config
from src.constant import MLFLOW_SETUP_FILE

import mlflow
import mlflow.sklearn

@dataclass
class PredictionConfig:
    preprocessor_path: str
    experiment_id: str
    experiment_dir_path: str

class PredictionPipeline:

    def __init__(self, config:PredictionConfig):
        self.config = config
        self.preprocessor = joblib.load(self.config.preprocessor_path)

        mlflow_setup = read_config(MLFLOW_SETUP_FILE).mlflow_setup

        mlflow.set_tracking_uri(mlflow_setup.mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_setup.mlflow_experiment_name)
        
        latest_run_id = get_latest_run_id(experiment_name=self.config.experiment_name)
        
        model_uri = self.config.experiment_dir_path.format(str(latest_run_id))
        
        with mlflow.start_run(run_id=latest_run_id):
            self.model = mlflow.sklearn.load_model(model_uri=model_uri)

            logging.info("Model Loaded")

    def predict(self, data:DataFrame) -> str:

        try:
            logging.info("Prediction Data Obtained")

            prediction_data = self.preprocessor.transform(data)
            logging.info("Prediction Data Preprocessed")

            prediction = self.model.predict_proba(prediction_data)
            logging.info("Model Prediction Complete")

            return prediction

        except Exception as e:
            logging.error(CustomException(e,sys))  

        finally:
            mlflow.end_run()