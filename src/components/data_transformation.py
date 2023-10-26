import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataTransformationConfig:
    root_dir: str
    data_path: str
    data_split: float

class DataTransformation:
    
    def __init__(self, config:DataTransformationConfig):
        self.config = config

    def data_split(self) -> None:
        try:
            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)
            data = pd.read_csv(self.config.data_path)

            train_set, test_set = train_test_split(data, random_state=42, test_size=self.config.data_split)

            train_set.to_csv(os.path.join(root_dir, "train.csv"), index=False)
            test_set.to_csv(os.path.join(root_dir, "test.csv"), index=False)

            logging.info(f"Train and Test sets created. Test Size: {test_set.shape}")

        except Exception as e:
            logging.error(CustomException(e, sys))
    