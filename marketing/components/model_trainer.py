import sys
import pandas as pd


from marketing.entity import ModelTrainerConfig
from marketing.entity import(
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from marketing import logging
from marketing import CustomException
from marketing.constants import SCHEMA_FILE_PATH, MODEL_CONFIG_FILE
from marketing.utils.main_utils import MainUtils
from utils.model_factory import ModelFactory




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

            # Get the transform pipeline configure to configure the pipeline
            transformer_pipeline_config = self.data_transformation_artifact.transformer_pipeline_config
            logging.info(f"Obtained the transformer pipeline configuration: {transformer_pipeline_config}")
            # Update the transform pipeline configuration and set the imputer strategy to None for optuna to optimize
            updated_pipeline_config_values = {
                "categorical_strategy":None,
                "numerical_strategy":None,
                "outlier_strategy":None
            }
            transformer_pipeline_config.update(updated_pipeline_config_values)
            logging.info(f"Updated transformer pipeline configuration: {transformer_pipeline_config}")

            # Initialize the ModelFactory
            model_factory = ModelFactory(transformer_pipeline_config, MODEL_CONFIG_FILE)
            best_models = model_factory.run(X_train, y_train)

            # Display the results
            overall_best_model_score = 0
            for best_model in best_models:
                logging.info(f"Model: {best_model.model_name}")
                logging.info(f"best_model: {best_model.best_model}")
                logging.info(f"Best Parameters: {best_model.best_params}")
                logging.info(f"Best Score: {best_model.best_score:.4f}")
                if best_model.best_score > overall_best_model_score:
                    overall_best_model_name = best_model.model_name
                    overall_best_model_score = best_model.best_score

        except Exception as e:
            raise CustomException(e, sys)