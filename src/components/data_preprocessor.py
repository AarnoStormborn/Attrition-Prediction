import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import read_config
from src.constant import SCHEMA_FILE

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

@dataclass
class DataPreprocessorConfig:
    root_dir: str
    train_data_path: str
    test_data_path: str

class DataPreprocessor:
    def __init__(self, config:DataPreprocessorConfig):
        self.config = config

    def get_preprocessor(self, cat_feats:List[str], num_feats:List[str]) -> ColumnTransformer:
        try:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("encoder", OrdinalEncoder())
            ])

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ("cat_pipeline",cat_pipeline,cat_feats),
                ("num_pipeline",num_pipeline,num_feats)
            ])

            logging.info("Preprocessor Pipeline Created")

            return preprocessor
        
        except Exception as e:
            logging.error(CustomException(e, sys))
    
    def preprocess_data(self) -> None:
        try:
            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)

            target = read_config(SCHEMA_FILE).target

            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            X_train, y_train = train_df.drop([target], axis=1), train_df[target]
            X_test, y_test = test_df.drop([target], axis=1), test_df[target]

            cat_feats = [col for col in X_train.columns if X_train[col].dtype == 'O']
            num_feats = [col for col in X_train.columns if col not in cat_feats]

            logging.info("Loading Preprocessor")

            preprocessor = self.get_preprocessor(cat_feats, num_feats)
            preprocessor.fit(X_train)

            label_encoder = LabelEncoder()
            label_encoder.fit(y_train)

            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            y_train_encoded = label_encoder.transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)

            logging.info("Data Succesfully Preprocessed")

            pd.DataFrame(X_train_transformed).to_csv(os.path.join(root_dir, "X_train.csv"), index=False)
            pd.DataFrame(np.reshape(y_train_encoded, -1)).to_csv(os.path.join(root_dir, "y_train.csv"), index=False)

            logging.info("Preprocessed Training Data Saved")

            pd.DataFrame(X_test_transformed).to_csv(os.path.join(root_dir, "X_test.csv"), index=False)
            pd.DataFrame(np.reshape(y_test_encoded, -1)).to_csv(os.path.join(root_dir, "y_test.csv"), index=False)

            logging.info("Preprocessed Test data saved")

            joblib.dump(preprocessor, os.path.join(root_dir, "preprocessor.pkl"))
            logging.info("Preprocessor Object Saved")
        
        except Exception as e:
            logging.error(CustomException(e, sys))
