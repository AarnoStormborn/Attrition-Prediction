# Attrition Prediction End-to-End Project

## Problem Statement
This project aims to predict employee attrition within a specified time frame using machine learning techniques. The model is designed to analyze historical employee data to foresee potential departures, allowing companies to proactively implement retention strategies. The project includes data preprocessing, model development, evaluation, and deployment stages, offering insights to reduce operational costs and enhance workforce stability.

## The Project
End-To-End Machine Learning Project for Attrition Prediction. Built as a Python Package
with an API endpoint for Prediction. The project is broken down into Components and Pipelines.
A Classification model is trained using IBM's employee attrition data, and predictions are mdae on
the FastAPI app.

### Components
Reponsible for Model Training. They are as follows:

`Data Ingestion -> Data Transformation -> Data Preprocessing -> Model Training -> Model Evaluation`

**Data Ingestion**: Data is downloaded from GitHub in a zipfile, and CSV File is extracted  \
**Data Transformation**: Data is split into Train and Test sets, new files are created  \
**Data Preprocessing**: Preprocessor Pipeline is defined, preprocessor is fit on train data and saved  \
**Model Training**: RandomForestClassifier Model is trained using GridSearchCV, Best estimator is saved  \
**Model Evaluation**: Best Model is evaluated on Test set, and results are saved

### Pipelines

1. **Training Pipeline**:
All components for model training are executed here
2. **Prediction Pipeline**:
Saved Preprocessor Object and Model are loaded here,  \
new data is preprocessed and predictions are returned

### Web API - FastAPI

API endpoint is defined for Prediction.  \
`infer/` takes new data and returns Prediction class and probability


### Extras

1. **Constants**:
Config file paths (yaml) are defined in this file
2. **Utils**:
Utility Functions
3. **Exception**:
A Custom Exception is setup to pinpoint errors within the project structure
4. **Logger**:
Custom Logging for project logs


### Future Work

1. **Mlflow**: Experiment & Model Tracking, versioning
2. **Artifacts Store**: Cloud storage integrated with Mlflow
3. **DVC**: Data Version Control for maintaining separate versions of Dataset
6. **Scheduling**: Automating training pipeline to support model retraining using Prefect

