import os
import sys
import time
import joblib
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import read_config
from src.constant import PARAMS_FILE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    root_dir: str
    preprocessor_path: str

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config = config

    def model_trainer(self, train_set:pd.DataFrame) -> None:
        try:
            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)

            X_train, y_train = train_set
            params = read_config(PARAMS_FILE).param_grid
            logging.info("Loading Model Parameters Grid")

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
            best_model = grid_search_cv.best_estimator_
            joblib.dump(best_model, os.path.join(root_dir, "model.pkl"))
            logging.info("Best Model Saved")

        except Exception as e:
            logging.error(CustomException(e,sys))

