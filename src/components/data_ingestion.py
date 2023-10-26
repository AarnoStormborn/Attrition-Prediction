import os
import sys
import zipfile
from urllib import request
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import read_config
from src.constant import CONFIG_FILE
from src.components.data_transformation import DataTransformation
from src.components.data_preprocessor import DataPreprocessor
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_URL: str
    local_file: str

class DataIngestion:

    def __init__(self, config:DataIngestionConfig):
        self.config = config

    def download_zip_file(self) -> None:

        try:
            os.makedirs(self.config.root_dir, exist_ok=True)

            if not os.path.exists(self.config.local_file):
                
                filename, _ = request.urlretrieve(
                    url = self.config.source_URL,
                    filename = self.config.local_file
                )
                logging.info(f"{filename} downloaded")

            else:
                logging.info("File already exists")
        
        except Exception as e:
            logging.error(CustomException(e,sys))    

    def extract_zip_file(self) -> None:

        try:
            with zipfile.ZipFile(self.config.local_file, 'r') as zip_ref:
                zip_ref.extractall(self.config.root_dir)
                
        except Exception as e:
            logging.error(CustomException(e, sys))

if __name__=="__main__":
    config = read_config(CONFIG_FILE)
    data = DataIngestion(config.data_ingestion)
    data.download_zip_file()
    data.extract_zip_file()

    trans = DataTransformation(config.data_transformation)
    trans.data_split()

    preprocessor = DataPreprocessor(config.data_preprocessor)
    train_set, test_set = preprocessor.preprocess_data()

    mt = ModelTrainer(config.model_trainer)
    mt.model_trainer(train_set)

    me = ModelEvaluation(config.model_evaluation)
    me.model_evaluator(test_set)


