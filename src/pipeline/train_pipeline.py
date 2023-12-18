import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import read_config
from src.constant import CONFIG_FILE

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_preprocessor import DataPreprocessor
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

config = read_config(CONFIG_FILE)

def data_ingestion():
    STAGE = "DATA INGESTION"
    try:
        logging.info(f">>>>>>>>>> {STAGE} <<<<<<<<<<")
        data_ingestion_component = DataIngestion(config=config.data_ingestion)
        data_ingestion_component.download_zip_file()
        data_ingestion_component.extract_zip_file()
        logging.info(f"{STAGE}: COMPLETE")
    except Exception as e:
        logging.error(CustomException(e, sys))


def data_transformation():
    STAGE = "DATA TRANSFORMATION"
    try:
        logging.info(f">>>>>>>>>> {STAGE} <<<<<<<<<<")
        data_transformation_component = DataTransformation(config=config.data_transformation)
        data_transformation_component.data_split()
        logging.info(f"{STAGE}: COMPLETE")
    except Exception as e:
        logging.error(CustomException(e, sys))


def data_preprocessing():
    STAGE = "DATA PREPROCESSING"
    try:
        logging.info(f">>>>>>>>>> {STAGE} <<<<<<<<<<")
        data_preprocessing_component = DataPreprocessor(config=config.data_preprocessor)
        data_preprocessing_component.preprocess_data()
        logging.info(f"{STAGE}: COMPLETE")
    except Exception as e:
        logging.error(CustomException(e, sys))


def model_training():
    STAGE = "MODEL TRAINING"
    try:
        logging.info(f">>>>>>>>>> {STAGE} <<<<<<<<<<")
        model_training_component = ModelTrainer(config=config.model_trainer)
        model_training_component.model_trainer()
        logging.info(f"{STAGE}: COMPLETE")
    except Exception as e:
        logging.error(CustomException(e, sys))


def model_evaluation():
    STAGE = "MODEL EVALUATION"
    try:
        logging.info(f">>>>>>>>>> {STAGE} <<<<<<<<<<")
        model_evaluation_component = ModelEvaluation(config=config.model_evaluation)
        model_evaluation_component.model_evaluator()
        logging.info(f"{STAGE}: COMPLETE")
    except Exception as e:
        logging.error(CustomException(e, sys))


if __name__=="__main__":
    data_ingestion()
    data_transformation()
    data_preprocessing()
    model_training()
    model_evaluation()