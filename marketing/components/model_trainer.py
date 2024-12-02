import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


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
            model_factory = ModelFactory(transformer_pipeline_config, MODEL_CONFIG_FILE)
            best_models = model_factory.run(X_train, y_train)

            trained_model_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(trained_model_path, exist_ok=True)

            os.makedirs(os.path.dirname(self.model_trainer_config.model_trained_for_production_path), exist_ok=True)
            model_trained_for_production_path = self.model_trainer_config.model_trained_for_production_path

            metric_artefacts_dir = self.model_trainer_config.metrics_dir
            os.makedirs(metric_artefacts_dir, exist_ok=True)

            # Display the results
            overall_best_model_score = 0
            #scores_dict_list = []
            for best_model in best_models:
                logging.info(f"Model: {best_model.model_name}")
                logging.info(f"best_model: {best_model.best_model}")
                logging.info(f"Best Parameters: {best_model.best_params}")
                logging.info(f"Best Score: {best_model.best_score:.4f}")
                trained_model_filename = f'{best_model.model_name}_pipeline{MODEL_SAVE_FORMAT}'
                trained_model_saved_path = os.path.join(trained_model_path, trained_model_filename)
                self.main_utils.save_object(trained_model_saved_path, best_model.best_model)
                logging.info(f'Serialized {best_model.model_name} trained pipeline to {trained_model_saved_path}')
                if best_model.best_score > overall_best_model_score:
                    overall_best_model_name = best_model.model_name
                    overall_best_model_score = best_model.best_score
                    overall_best_model_pipeline = best_model.best_model

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


            trained_model_filename = f'BestModel_{overall_best_model_name}{MODEL_SAVE_FORMAT}'
            trained_model_saved_path = os.path.join(trained_model_path, trained_model_filename)
            self.main_utils.save_object(model_trained_for_production_path, overall_best_model_pipeline)

            logging.info(f"(overall_best_model_score: {overall_best_model_name}")
            logging.info(f"(overall_best_model_score: {overall_best_model_score}")



        except Exception as e:
            raise CustomException(e, sys)