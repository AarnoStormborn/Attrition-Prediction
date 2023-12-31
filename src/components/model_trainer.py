import os
import sys
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import read_config
from src.constant import PARAMS_FILE, MLFLOW_SETUP_FILE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import mlflow
import mlflow.sklearn

@dataclass
class ModelTrainerConfig:
    model: str

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config = config

    def model_trainer(self) -> None:
        try:
            
            mlflow_setup = read_config(MLFLOW_SETUP_FILE).mlflow_setup

            mlflow.set_tracking_uri(mlflow_setup.mlflow_tracking_uri)
            mlflow.set_experiment(mlflow_setup.mlflow_experiment_name)

            X_train = pd.read_csv(self.config.preprocessed_train_data)
            y_train = np.reshape(pd.read_csv(self.config.labels), -1)

            params = read_config(PARAMS_FILE).param_grid
            logging.info("Loading Model Parameters Grid")
            
            mlflow.start_run()

            logging.info("Starting Model Training...")
            start = time.time()
            rfc = RandomForestClassifier(random_state=42)
            grid_search_cv = GridSearchCV(estimator=rfc,
                                          param_grid=params,
                                          cv=5,
                                          scoring='accuracy')
            
            grid_search_cv.fit(X_train, y_train)

            end = time.time() - start
            logging.info(f"Model Training Complete. Time taken: {end:.2f} seconds")

            mlflow.log_param("cv", 5)
            mlflow.log_param("scoring", "accuracy")
            mlflow.log_metric("training time", end)
            mlflow.log_metric("training - accuracy", grid_search_cv.best_score_)

            mlflow.sklearn.log_model(grid_search_cv.best_estimator_, "classification_model")

            logging.info("Best Model Saved")

        except Exception as e:
            logging.error(CustomException(e,sys))

