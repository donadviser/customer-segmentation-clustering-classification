from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    data_drift_file_path: str
    validation_status: bool
    validation_message: str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformer_pipeline_config: dict


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    expected_score: float
