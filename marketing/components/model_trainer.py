import sys
from typing import List, Tuple
import os
import pandas as pd
import numpy as np

from marketing.entity.config_entity import ModelTrainerConfig
from marketing.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from marketing import logging
from marketing import CustomException
from marketing.constants import SCHEMA_FILE_PATH, MODEL_CONFIG_FILE
from marketing.utils.main_utils import MainUtils
from marketing.utils.model_factory import ModelFactory

from sklearn.metrics import (accuracy_score, f1_score, classification_report,
precision_score, recall_score, roc_auc_score, log_loss)

from imblearn.over_sampling import SMOTE



class CustomerSegmentationModel:
    def __init__(self, preprocessing_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Entered predict method of CustomerSegmentationModel class.")
        try:
            X_transformed = self.preprocessing_object.transform(X)
            preds = self.trained_model_object.predict(X_transformed)
            logging.info("Exited predict method of CustomerSegmentationModel class.")
            return preds
        except Exception as e:
            raise CustomException(e, sys)

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
        self.model_training_config = model_trainer_config
        self.main_utils = MainUtils()
        self._schema_config = self.main_utils.read_yaml_file(SCHEMA_FILE_PATH)
        self.model_config = self.main_utils.read_yaml_file(MODEL_CONFIG_FILE)


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class.")

        try:
            data_train = self.main_utils.load_object(file_path=self.data_transformation_artifact.transformed_train_file_path)
            data_test = self.main_utils.load_object(file_path=self.data_transformation_artifact.transformed_test_file_path)

            target_col = self._schema_config['target_column']
            logging.info(f"Obtained the target column: {target_col}")
            X_train, y_train = self.main_utils.separate_data(data_train, target_col)
            X_test, y_test = self.main_utils.separate_data(data_test, target_col)

            # transform the dataset
            oversample = SMOTE()
            X_train, y_train = oversample.fit_resample(X_train, y_train)

            # Initialise the ModelFactory
            model_factory = ModelFactory(MODEL_CONFIG_FILE)
            best_models = model_factory.run(X_train, y_train)

            # Display the results
            for best_model in best_models:
                logging.info(f"Model: {best_model.model_name}")
                logging.info(f"best_model: {best_model.best_model}")
                logging.info(f"Best Parameters: {best_model.best_params}")
                logging.info(f"Best Score: {best_model.best_score:.4f}")

            #Use the best models for predictions
            for best_model in best_models:
                logging.info(f"\nEvaluating Model: {best_model.model_name}")

                n_classes = 3
                metrics = {}

                # Train the best model on the training set
                trained_model = best_model.best_model.fit(X_train, y_train)
                y_pred = trained_model.predict(X_test)
                y_pred_proba = trained_model.predict_proba(X_test)[:, 1]

                # Calculate evaluation metrics
                metrics['Precision (macro)'] = precision_score(y_test, y_pred, average='macro')
                metrics['Recall (macro)'] = recall_score(y_test, y_pred, average='macro')
                metrics['F1 Score (macro)'] = f1_score(y_test, y_pred, average='macro')
                metrics['Accuracy'] = accuracy_score(y_test, y_pred)
                #metrics['Log Loss'] = log_loss(y_test, y_test)

                logging.info(f"Test Accuracy: {metrics['Accuracy']:.4f}")
                logging.info(f"Test Precision: {metrics['Precision (macro)']:.4f}")
                logging.info(f"Test Recall: {metrics['Recall (macro)']:.4f}")
                logging.info(f"Test F1 Score: {metrics['F1 Score (macro)']:.4f}")
                #logging.info(f"Test Log Loss: {metrics['Log Loss']:.4f}")
                logging.info("\nClassification Report:")
                logging.info(classification_report(y_test, y_pred))

                logging.info(f"Predictions for {best_model.model_name}: {y_pred[:5]}")
            logging.info(f"Truth value for y_test: {y_test[:5].to_list()}")
        except Exception as e:
            raise CustomException(e, sys)