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

STAGE = "DATA INGESTION"
try:
    logging.info(f">>>>>>>>>> {STAGE} <<<<<<<<<<")
    data_ingestion_component = DataIngestion(config=config.data_ingestion)
    data_ingestion_component.download_zip_file()
    data_ingestion_component.extract_zip_file()
    logging.info(f"{STAGE}: COMPLETE")
except Exception as e:
    logging.error(CustomException(e, sys))


STAGE = "DATA TRANSFORMATION"
try:
    logging.info(f">>>>>>>>>> {STAGE} <<<<<<<<<<")
    data_transformation_component = DataTransformation(config=config.data_transformation)
    data_transformation_component.data_split()
    logging.info(f"{STAGE}: COMPLETE")
except Exception as e:
    logging.error(CustomException(e, sys))


STAGE = "DATA PREPROCESSING"
try:
    logging.info(f">>>>>>>>>> {STAGE} <<<<<<<<<<")
    data_preprocessing_component = DataPreprocessor(config=config.data_preprocessor)
    train_set, test_set = data_preprocessing_component.preprocess_data()
    logging.info(f"{STAGE}: COMPLETE")
except Exception as e:
    logging.error(CustomException(e, sys))


STAGE = "MODEL TRAINING"
try:
    logging.info(f">>>>>>>>>> {STAGE} <<<<<<<<<<")
    model_training_component = ModelTrainer(config=config.model_trainer)
    model_training_component.model_trainer(train_set=train_set)
    logging.info(f"{STAGE}: COMPLETE")
except Exception as e:
    logging.error(CustomException(e, sys))


STAGE = "MODEL EVALUATION"
try:
    logging.info(f">>>>>>>>>> {STAGE} <<<<<<<<<<")
    model_evaluation_component = ModelEvaluation(config=config.model_evaluation)
    model_evaluation_component.model_evaluator(test_set=test_set)
    logging.info(f"{STAGE}: COMPLETE")
except Exception as e:
    logging.error(CustomException(e, sys))