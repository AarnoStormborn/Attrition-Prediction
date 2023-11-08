import os
import sys
import joblib

from pandas import DataFrame
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class PredictionConfig:
    preprocessor_path: str
    model_path: str

class PredictionPipeline:

    def __init__(self, config:PredictionConfig):
        self.config = config
        self.preprocessor = joblib.load(self.config.preprocessor_path)
        self.model = joblib.load(self.config.model_path)

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