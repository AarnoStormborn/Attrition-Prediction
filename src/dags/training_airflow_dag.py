import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

from src.pipeline.train_pipeline import (
    data_ingestion, data_transformation, data_preprocessing,
    model_training, model_evaluation
)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 12, 20),
    'email': ['harsh220902@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'schedule_interval': '@weekly'
}

DESC = """
    This pipeline is responsible for retraining
    a new model for attrition prediction
"""

dag = DAG(
    dag_id='training_airflow_dag',
    default_args=default_args,
    description=DESC
)

run_data_ingestion = PythonOperator(
    task_id='data_ingestion',
    python_callable=data_ingestion,
    dag=dag
)

run_data_transformation = PythonOperator(
    task_id='data_transformation',
    python_callable=data_transformation,
    dag=dag
)

run_data_preprocessing = PythonOperator(
    task_id='data_preprocessing',
    python_callable=data_preprocessing,
    dag=dag
)

run_model_training = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    dag=dag
)

run_model_evaluation = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation,
    dag=dag
)

run_data_ingestion >> run_data_transformation >> run_data_preprocessing >> run_model_training >> run_model_evaluation

