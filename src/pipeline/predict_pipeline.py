import os
import sys
import joblib

from pandas import DataFrame
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import get_latest_run_id

import mlflow
import mlflow.sklearn

@dataclass
class PredictionConfig:
    preprocessor_path: str
    experiment_id: str

class PredictionPipeline:

    def __init__(self, config:PredictionConfig):
        self.config = config
        self.preprocessor = joblib.load(self.config.preprocessor_path)
        
        latest_run_id = get_latest_run_id(self.config.experiment_id)
        with mlflow.start_run(run_id=latest_run_id):
            model_uri = os.path.join(mlflow.get_artifact_uri(), 'classification_model')
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