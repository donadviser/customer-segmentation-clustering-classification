# pipeline name and root directory constant
import os
from os import environ
from datetime import datetime
from from_root import from_root



TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_YEAR =  datetime.now().year

# Configuration file
MODEL_CONFIG_FILE = "config/model.yaml"
PARAM_FILE_PATH = "params.yaml"
SCHEMA_FILE_PATH = "config/schema.yaml"


# Source Data and MongoDB configuration
MONGODB_DB_NAME: str = "marketing_campaign"
MONGODB_COLLECTION_NAME: str = "customer_segmentation"
MONGODB_URL_KEY = environ["MONGODB_MARKETING_URL"]

# GitHub DataSource
GITHUB_SOURCE_URL = "https://github.com/donadviser/datasets/raw/master/data-don/marketing_campaign.zip"


TARGET_COLUMN = "cluster"
PIPELINE_NAME: str = "marketing"
ARTIFACT_DIR: str = "artifact"
LOG_DIR = "logs"
LOG_FILE = "customer_segmentation.log"

ARTEFACTS_ROOT_DIR = os.path.join(from_root(), "artefacts")

# common file name

FILE_NAME: str = "marketing_campaign.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
PREPROCESSOR_UNFITTED_FILE_NAME = "precoessor_unfitted.pkl"
MODEL_FILE_NAME = "model.pkl"
TRAINING_BUCKET_NAME = "customer-segmentation-model"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = MONGODB_COLLECTION_NAME
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_SAVE_FORMAT = ".pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
METRICS_DIR_NAME: str = "metrics"

"""
MODEL Evauation related constant start with MODEL_EVALUATION var name
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_PUSHER_BUCKET_NAME = TRAINING_BUCKET_NAME
MODEL_TRAINED_FOR_PRODUCTION_NAME: str = "model.pkl"
MODEL_TRAINED_FOR_PRODUCTION_DIR: str = "prod_model"

# AWS ACCESS KEYS
AWS_ACCESS_KEY_ID = environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = environ["AWS_SECRET_ACCESS_KEY"]
REGION_NAME = "us-east-1"

#WebApp
APP_HOST = "0.0.0.0"
APP_PORT = 5000