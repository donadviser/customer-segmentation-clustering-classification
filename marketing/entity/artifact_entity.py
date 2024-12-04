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
    trained_model_dir: str
    metric_artefacts_dir: str
    trained_model_names: list


@dataclass
class ClassificationMetricArtifact:
    confusion_matrix: str
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    auc_macro: float
    auc_weighted: float
    log_loss: float
    false_positive_rate: float
    average_precision_macro: float
    average_precision_weighted: float


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_accuracy: float
    best_model_path: str
    s3_model_score: float
    best_model_name: float
    best_model_metric_artifact: ClassificationMetricArtifact