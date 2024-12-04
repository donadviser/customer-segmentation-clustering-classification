import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


from marketing.entity import ModelTrainerConfig
from marketing.entity import(
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from marketing import logging
from marketing import CustomException
from marketing.constants import (
    SCHEMA_FILE_PATH, MODEL_CONFIG_FILE, MODEL_SAVE_FORMAT,
    PARAM_FILE_PATH
)
from marketing.utils.main_utils import MainUtils
from utils.model_factory import ModelFactory

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss,confusion_matrix,
                             average_precision_score)




class CustomerSegmentationModel:
    def __init__(self, pipeline_model: object, preprocess_pipeline: object = None):
        self.preprocess_pipeline = preprocess_pipeline
        self.pipeline_model = pipeline_model

    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> object:
        """Train the model with the provided data."""
        try:
            if self.preprocess_pipeline:
                X_train = self.preprocess_pipeline.fit_transform(X_train)
            self.pipeline_model.fit(X_train, y_train)
            return self.pipeline_model
        except Exception as e:
            self._handle_exception(e)

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Predict the target variable for the given test data."""
        logging.info("Entered predict method of CustomerSegmentationModel class.")
        try:
            if self.preprocess_pipeline:
                X_test = self.preprocess_pipeline.transform(X_test)
                logging.info("Transformed the X_test using preprocess_pipeline to get predictions")

            y_pred = self.pipeline_model.predict(X_test)
            y_pred_proba = self.pipeline_model.predict_proba(X_test)
            logging.info("Exited predict method of CustomerSegmentationModel class.")

            return y_pred, y_pred_proba
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self, y_test, y_pred, y_pred_proba=None, n_classes=None)-> Dict[str, float]:
        """Evaluate the model's performance using given test data and predictions."""

        logging.info("Entered evaluate method of CustomerSegmentationModel class.")
        metrics = {}
        try:
            if n_classes is None:
                n_classes = len(np.unique(y_test))
            logging.info(f"No of unique classes in y_test: {n_classes}")

            # Accuracy
            metrics['accuracy'] = accuracy_score(y_test, y_pred)

            # Precision, Recall, F1 (Macro and Weighted)
            metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
            metrics['precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
            metrics['recall_weighted'] = recall_score(y_test, y_pred, average='weighted')
            metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
            metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')

            # AUC ROC (Macro and Weighted)
            metrics['auc_macro'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            metrics['auc_weighted'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

            # Log Loss
            metrics['log_loss'] = log_loss(y_test, y_pred_proba)

            # False Positive Rate
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
            fprs = []
            for i in range(n_classes):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                fprs.append(fp / (fp + tn))
            metrics['false_positive_rate'] = np.mean(fprs)

            # Average Precision Score (Macro and Weighted)
            metrics['average_precision_macro'] = average_precision_score(y_test, y_pred_proba, average='macro')
            metrics['average_precision_weighted'] = average_precision_score(y_test, y_pred_proba, average='weighted')

            logging.info("Exited evaluate method of CustomerSegmentationModel class.")
            return metrics
        except Exception as e:
            self._handle_exception(e)

    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self) -> str:
        return f"Customer Segmentation Model: {type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig,
                 ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.main_utils = MainUtils()
        self._schema_config = self.main_utils.read_yaml_file(SCHEMA_FILE_PATH)
        self.model_config = self.main_utils.read_yaml_file(MODEL_CONFIG_FILE)
        # Get the params from the params.yaml file
        self.param_constants = self.main_utils.read_yaml_file(PARAM_FILE_PATH)
        logging.info(f"self.param_constants: {self.param_constants}")


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class.")

        try:
            data_train = self.main_utils.load_object(file_path=self.data_transformation_artifact.transformed_train_file_path)
            data_test = self.main_utils.load_object(file_path=self.data_transformation_artifact.transformed_test_file_path)

            target_col = self._schema_config['target_column']
            logging.info(f"Obtained the target column: {target_col}")
            X_train, y_train = self.main_utils.separate_data(data_train, target_col)
            X_test, y_test = self.main_utils.separate_data(data_test, target_col)

            # Get the transform pipeline configure to configure the pipeline
            transformer_pipeline_config = self.data_transformation_artifact.transformer_pipeline_config
            logging.info(f"Obtained the transformer pipeline configuration: {transformer_pipeline_config}")
            # Update the transform pipeline configuration and set the imputer strategy to None for optuna to optimize
            updated_pipeline_config_values = {
                "categorical_strategy":'most_frequent',
                "numerical_strategy":'mean',
                "outlier_strategy":"power_transform"
            }
            transformer_pipeline_config.update(updated_pipeline_config_values)
            logging.info(f"Updated transformer pipeline configuration: {transformer_pipeline_config}")

            # Initialize the ModelFactory
            model_factory = ModelFactory(transformer_pipeline_config, self.model_config, self.param_constants)
            best_models = model_factory.run(X_train, y_train)

            trained_model_dir = self.model_trainer_config.trained_model_file_path
            os.makedirs(trained_model_dir, exist_ok=True)

            metric_artefacts_dir = self.model_trainer_config.metrics_dir
            os.makedirs(metric_artefacts_dir, exist_ok=True)

            # Display the results
            overall_best_model_score = 0
            #scores_dict_list = []
            trained_model_names = []
            for best_model in best_models:
                logging.info(f"Model: {best_model.model_name}")
                logging.info(f"best_model: {best_model.best_model}")
                logging.info(f"Best Parameters: {best_model.best_params}")
                logging.info(f"Best Score: {best_model.best_score:.4f}")
                trained_model_names.append(best_model.model_name)

                trained_model_pipeline_filename = f'{best_model.model_name}_pipeline{MODEL_SAVE_FORMAT}'
                trained_model_saved_path = os.path.join(trained_model_dir, trained_model_pipeline_filename)
                self.main_utils.save_object(trained_model_saved_path, best_model.best_pipeline)
                logging.info(f'Serialized {best_model.model_name} trained pipeline to {trained_model_saved_path}')
                if best_model.best_score > overall_best_model_score:
                    overall_best_model_name = best_model.model_name
                    overall_best_model_score = best_model.best_score
                    overall_best_model_pipeline = best_model.best_pipeline

                #scores_dict = best_model.scores_dict
                #scores_dict_list.append(scores_dict)

                #logging.info(f"scores_dict: {scores_dict}")


            # Plotting boxplot for the training scores of each classifier


            """sns.set(style="darkgrid")
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=[scores for scores in scores_dict_list.values()],
                        orient="v",
                        palette="Set3")
            plt.xticks(ticks=range(len(scores_dict_list)), labels=scores_dict_list.keys())
            plt.title("Comparison of Training Scores for Each Classifier")
            plt.xlabel("Classifier")
            plt.ylabel("Optuna Hyperparameter Tunning Cross-validation F1 Score ")


            # Superimposing mean scores as scatter points with higher zorder
            mean_scores = [np.mean(scores) for scores in scores_dict_list.values()]
            for i, mean_score in enumerate(mean_scores):
                plt.scatter(i, mean_score, color='red', marker='o', s=100, label='Mean Score' if i == 0 else "", zorder=10)
            plt.xticks(rotation=45, ha="right")
            plt.legend()

            file_path = os.path.join(metric_artefacts_dir, "Boxplot_training_score.png")
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            logging.info("Boxplot for training score saved")"""

            logging.info(f"(overall_best_model_score: {overall_best_model_name}")
            logging.info(f"(overall_best_model_score: {overall_best_model_score}")

            model_trainer_artifact = ModelTrainerArtifact(
                metric_artefacts_dir=metric_artefacts_dir,
                trained_model_dir=trained_model_dir,
                trained_model_names=trained_model_names
            )
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)