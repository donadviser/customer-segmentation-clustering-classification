import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union, List
import copy


from marketing.entity.config_entity import DataTransformationConfig
from marketing.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from marketing.components.data_ingestion import DataIngestion
from marketing.components.data_clustering import CreateClusters


from marketing import logging
from marketing import CustomException
from marketing.utils.main_utils import MainUtils
from marketing.constants import SCHEMA_FILE_PATH

from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler,
    OrdinalEncoder, PowerTransformer,
    RobustScaler, MinMaxScaler,
    FunctionTransformer)
from marketing.utils.custom_transformers import (
    DropRedundantColumns,
    CreateNewFeature,
    LogTransformer,
    OutlierDetector,
    ReplaceValueTransformer,
    OutlierHandler,
)
from marketing.utils.pipeline_manager import PipelineManager


class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig
    ):
        """DataTransformation class constructor

        Args:
            data_ingestion_artifact (DataIngestionArtifact): _description_
            data_validation_artifact (DataValidationArtifact): _description_
            data_transformation_config (DataTransformationConfig): _description_
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

            self.main_utils = MainUtils()
            self._schema_config = self.main_utils.read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            self._handle_exception(e)




    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)


    def transform_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> object:
        """
        Apply data preprocessing steps, feature engineering, and handling missing values.

        Returns:
            object: Returns the preprocessed training and testing dataframes.
        """
        try:
            logging.info("Starting data preprocessing")

            preprocessor = copy.deepcopy(self.preprocessor)

            transformed_train = preprocessor.fit_transform(train_df)
            transformed_test = preprocessor.transform(test_df)


            feature_names = self.get_feature_names(preprocessor, transformed_train)
            logging.info(f"Obtained Feature Names: {feature_names}")


            # Convert transformed data to DataFrame with feature names
            X_train_transformed_df = pd.DataFrame(transformed_train, columns=feature_names)
            X_test_transformed_df = pd.DataFrame(transformed_test, columns=feature_names)


            return X_train_transformed_df, X_test_transformed_df

        except Exception as e:
            self._handle_exception(e)

    @staticmethod
    def get_feature_names(pipeline, X_train):
        """Extract feature names from a pipeline that may contain ColumnTransformer and PCA."""
        column_transformer = None
        pca_step_name = None
        pca_n_components = None

        # Identify ColumnTransformer and PCA in the pipeline
        for name, step in pipeline.named_steps.items():
            if isinstance(step, ColumnTransformer):
                column_transformer = step
            elif isinstance(step, PCA):
                pca_step_name = name
                pca_n_components = step.n_components_

        # Extract feature names from ColumnTransformer
        if column_transformer:
            feature_names = []
            for name, transformer, columns in column_transformer.transformers_:
                if transformer != 'drop':
                    if hasattr(transformer, 'get_feature_names_out'):
                        feature_names.extend(transformer.get_feature_names_out(columns))
                    else:
                        feature_names.extend(columns)
        else:
            # Fallback to input features if no ColumnTransformer is present
            feature_names = X_train.columns.tolist()

        # If PCA exists, rename features as PC components
        if pca_step_name:
            feature_names = [f"PC{i+1}" for i in range(pca_n_components)]

        return feature_names

    def get_preprocessor_pipeline(self, transformer_pipeline_config):
        pipeline_manager = PipelineManager(**transformer_pipeline_config)

        pipeline_manager.add_step('create_new_features', pipeline_manager.build(step_name='create_new_features'), position=0)
        pipeline_manager.add_step('replace_class', pipeline_manager.build(step_name='replace_class'), position=1)
        pipeline_manager.add_step('drop_cols', pipeline_manager.build(step_name='drop_cols'), position=2)
        pipeline_manager.add_step('column_transformer', pipeline_manager.build(step_name='column_transformer'), position=3)

        return pipeline_manager.get_pipeline()


    def initiate_data_transformation(self)  -> DataTransformationArtifact:
        """
        Initiate data transformation by applying data preprocessing steps, feature engineering, and handling missing values, and returning the artifact.

        Returns:
            DataTransformationArtifact: Returns an object containing the preprocessed training and testing dataframes.
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")

                # Getting neccessary column names from config file
                self.numerical_features = self._schema_config['numerical_features']
                self.categorical_features = self._schema_config['categorical_features']
                self.outlier_features = self._schema_config['outlier_features']
                self.drop_columns = self._schema_config['drop_columns']
                logging.info("Obtained the numerical_features, categorical_features, outlier_features and drop_columns from schema config")

                # Getting the bins and namess from config file
                self.education_map = self._schema_config['education_map']
                self.marital_map = self._schema_config['marital_map']
                self.target_column = self._schema_config['target_column']
                logging.info("Obtained the education_map, marital_map, target_column from schema config")


                # Populate the transformer_pipe_config
                transformer_pipeline_config = {
                    "numerical_features": self.numerical_features,
                    "categorical_features": self.categorical_features,
                    "outlier_features": self.outlier_features,
                    "drop_columns": self.drop_columns,
                    "education_map":self.education_map,
                    "marital_map":self.marital_map,
                    "pipeline_type":"ImbPipeline",
                    "target_column":self.target_column,
                    "categorical_strategy":"most_frequent",
                    "numerical_strategy":"mean",
                    "outlier_strategy":"power_transform"
                }


                # Read the train and test data from data ingestion artifact folder
                data_train = self.main_utils.read_csv_file(file_path=self.data_ingestion_artifact.train_file_path)
                data_test = self.main_utils.read_csv_file(file_path=self.data_ingestion_artifact.test_file_path)
                logging.info("Obtained the data_train and data_test from the data ingestion artifact folder")


                train_df = (data_train
                            .pipe(self.main_utils.rename_columns_to_snake_case)
                            )
                test_df = (data_test
                            .pipe(self.main_utils.rename_columns_to_snake_case)
                            )
                logging.info("Renamed the orginal dataset columns to snake_case for both data_train and data_test")


                self.preprocessor = self.get_preprocessor_pipeline(transformer_pipeline_config)
                logging.info("Built the preprocessor pipeline")

                preprocessor_obj_dir = os.path.dirname(self.data_transformation_config.transformed_object_file_path)

                os.makedirs(preprocessor_obj_dir, exist_ok=True)
                transformed_object_file_path = self.main_utils.save_object(
                    self.data_transformation_config.transformed_object_file_path,
                    self.preprocessor,
                )
                logging.info(f"Saved preprocessor object to: {transformed_object_file_path}")

                # Transforms train and test data using the preprocessor
                X_train_transformed_df, X_test_transformed_df = self.transform_data(train_df, test_df)
                logging.info("Applied the feature engineering steps on the data")
                logging.info(f"transformed_train.shape: {X_train_transformed_df.shape}")
                logging.info(f"transformed_test.shape: {X_test_transformed_df.shape}")
                logging.info(f"data_train_preprocessed.head(): {X_train_transformed_df.head()}")

                #create clusters
                create_clusters = CreateClusters(self.target_column)
                data_train_labelled, data_test_labelled = create_clusters.initialise_clustering(
                    X_train_transformed_df,
                    X_test_transformed_df,
                    train_df,
                    test_df)

                logging.info("Created clusters for target column")
                logging.info(f"data_train_labelled.head(): {data_train_labelled.head()}")

                transformed_train_file_path = self.main_utils.save_object(
                    self.data_transformation_config.transformed_train_file_path,
                    data_train_labelled,
                )
                transformed_test_file_path = self.main_utils.save_object(
                    self.data_transformation_config.transformed_test_file_path,
                    data_test_labelled,
                )

                data_trasformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=transformed_object_file_path,
                    transformed_train_file_path=transformed_train_file_path,
                    transformed_test_file_path=transformed_test_file_path,
                    transformer_pipeline_config=transformer_pipeline_config,
                )
                logging.info("Exited the initiate_data_transformation method of Data Transformation class.")

            else:
                logging.info("Data validation step pipeline. Skipping data transformation pipeline")
                data_trasformation_artifact = None
                logging.info("Exited the initiate_data_transformation method of Data Transformation class.")


            return data_trasformation_artifact
        except Exception as e:
            self._handle_exception(e)




