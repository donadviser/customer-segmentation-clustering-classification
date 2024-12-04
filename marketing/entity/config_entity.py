import os
from marketing.constants import *
from dataclasses import dataclass


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    drift_report_file_path = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR,
                                          DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)


@dataclass
class DataTransformationConfig:
    data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path = os.path.join(data_transformation_dir,
                                                DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                TRAIN_FILE_NAME.replace("csv", "npz"))
    transformed_test_file_path = os.path.join(data_transformation_dir,
                                                DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                TEST_FILE_NAME.replace("csv", "npz"))
    transformed_object_file_path = os.path.join(data_transformation_dir,
                                                DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                PREPROCSSING_OBJECT_FILE_NAME)
    transformed_unfitted_file_path = os.path.join(data_transformation_dir,
                                                  DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                  PREPROCESSOR_UNFITTED_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    model_training_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_training_dir, MODEL_TRAINER_TRAINED_MODEL_DIR)
    expected_score: float = MODEL_TRAINER_EXPECTED_SCORE
    metrics_dir: str = os.path.join(training_pipeline_config.artifact_dir, METRICS_DIR_NAME)


@dataclass
class ModelEvaluationConfig:
    model_trained_for_production_path: str = os.path.join(training_pipeline_config.artifact_dir,MODEL_TRAINED_FOR_PRODUCTION_DIR, MODEL_TRAINED_FOR_PRODUCTION_NAME)
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_PUSHER_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME


class PCAConfig:
    def __init__(self):
        self.n_components = 2
        self.random_state = 42

    def get_pca_config(self):
        return self.__dict__

class ClusteringConfig:
    def __init__(self):
        self.n_clusters=3
        #self.affinity='euclidean'
        #self.linkage='ward'
        #self.random_state = 42

    def get_clustering_config(self):
        return self.__dict__