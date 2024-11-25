import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union


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


class PreprocessingPipeline:
    """
    A class that encapsulates the preprocessing steps for feature engineering,
    imputation, scaling, encoding, and transformations. This can be inserted into
    the overall pipeline before the model fitting step.
    """
    def __init__(self, education_map, marital_map, drop_columns, numerical_features,
                 categorical_features, outlier_features, pipeline_type="Pipeline"):
        """
        Initialize the PreprocessingPipeline with necessary parameters.

        Args:
            education_map: Parameters for mapping education categorical feature into numbers.
            marital_map: parameters for mapping marital categorical feature into numbers.
            drop_columns: Columns to be dropped from the dataset.
            numerical_features: List of numerical features for processing.
            categorical_features: List of categorical features for OneHot encoding.
            outlier_features: Features that require power transformation.
        """
        self.education_map = education_map
        self.marital_map = marital_map
        self.drop_columns = drop_columns
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.outlier_features = outlier_features
        self.pipeline_type = pipeline_type


    def instantiate_numerical_simple_imputer(self, strategy: str='mean', fill_value: int=-1) -> SimpleImputer:
        return SimpleImputer(strategy=strategy, fill_value=fill_value)

    def instantiate_categorical_simple_imputer(self, strategy: str="most_frequent", fill_value: str='missing') -> SimpleImputer:
        return SimpleImputer(strategy=strategy, fill_value=fill_value)

    def instantiate_outliers(self, strategy: str="power_transform") -> Union[PowerTransformer, FunctionTransformer, OutlierDetector]:
        """
        Instantiate outlier handling method: PowerTransformer, LogTransformer, or OutlierDetector.

        Args:
            trial (optuna.Trial, optional): The trial object for hyperparameter optimization.

        Returns:
            Union[PowerTransformer, FunctionTransformer, OutlierDetector]: The selected outlier handling method.
        """
        # Suggest from available options
        # options = ['power_transform', 'log_transform', 'iqr_clip', 'iqr_median', 'iqr_mean']

        if strategy == 'power_transform':
            return PowerTransformer(method='yeo-johnson')
        elif strategy == 'log_transform':
            return LogTransformer()
            #return FunctionTransformer(np.log1p)  # Log transformation
        elif strategy in ['iqr_clip', 'iqr_median', 'iqr_mean']:
            return OutlierHandler(strategy=strategy)  # Instantiate OutlierDetector
        else:
            raise ValueError(f"Unknown strategy for outlier handling: {strategy}")

    def add_step(self, step_name, step_object, position=None):
        """
        Add a transformation step to the pipeline.

        Args:
            step_name (str): Name of the step to add.
            step_object (object): The transformer or estimator object (e.g., scaler, classifier).
            position (int or None): Optional; the position to insert the step.
                                    If None, the step is appended at the end of the pipeline.
        """
        if position is None:
            self.pipeline.steps.append((step_name, step_object))
        else:
            self.pipeline.steps.insert(position, (step_name, step_object))

    def get_pipeline(self):
        if self.pipeline_type == 'ImbPipeline':
            self.pipeline = ImbPipeline(steps=[])
        elif self.pipeline_type == 'Pipeline':
            self.pipeline = Pipeline(steps=[])
        else:
            raise ValueError("Unsupported pipeline type. Choose 'ImbPipeline' or 'Pipeline'.")

        self.add_step('create_new_features', self.build(step_name='create_new_features'), position=0)
        self.add_step('drop_cols', self.build(step_name='drop_cols'), position=1)
        self.add_step('column_transformer', self.build(step_name='column_transformer'), position=2)
        self.add_step('scalar', StandardScaler(), position=3)

        return self.pipeline


    def build(self, step_name=None, **column_transformer_strategy):
        """
        Build the preprocessing pipeline with feature creation, transformation,
        imputation, scaling, and encoding steps.

        Returns:
            Transformer: The appropriate transformer for the given step.
        """

        if step_name == "create_new_features":
            return CreateNewFeature(education_map=self.education_map, marital_map=self.marital_map)

        if step_name == "replace_class":
            return ReplaceValueTransformer(old_value="?", new_value='missing')

        if step_name == "drop_cols":
            return DropRedundantColumns(redundant_cols=self.drop_columns)

        if step_name == 'column_transformer':

            numerical_strategy = column_transformer_strategy.get('numerical_strategy', 'mean')
            categorical_strategy = column_transformer_strategy.get('categorical_strategy','most_frequent')
            outlier_strategy = column_transformer_strategy.get('outlier_strategy', 'power_transform')

            return ColumnTransformer(
                transformers=[
                    ('categorical', Pipeline([
                        ('imputer', self.instantiate_categorical_simple_imputer(strategy=categorical_strategy)),
                        #('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
                        #('ordinal', OrdinalEncoder())
                    ]), self.categorical_features),

                    ('numerical', Pipeline([
                        ('imputer', self.instantiate_numerical_simple_imputer(strategy=numerical_strategy)),
                        #('scaler', StandardScaler())  # Add scaler if needed
                    ]), self.numerical_features),

                    ('outlier_transform', Pipeline([
                        ('imputer', self.instantiate_numerical_simple_imputer(strategy=numerical_strategy)),
                        ('outlier_transformer', self.instantiate_outliers(strategy=outlier_strategy))  # Update this line
                    ]), self.outlier_features),
                ],
                remainder='passthrough'
            )




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


    def data_preprocessing(self,) -> object:
        """
        Apply data preprocessing steps, feature engineering, and handling missing values.

        Returns:
            object: Returns the preprocessed training and testing dataframes.
        """
        try:
            logging.info("Starting data preprocessing")

            # Apply feature engineering steps
            data_train_preprocessed = self._apply_feature_engineering(self.data_train)
            data_test_preprocessed = self._apply_feature_engineering(self.data_test)

            # Handle missing values
            data_train_preprocessed = self._handle_missing_values(data_train_preprocessed)
            data_test_preprocessed = self._handle_missing_values(data_test_preprocessed)

            return data_train_preprocessed, data_test_preprocessed
        except Exception as e:
            self._handle_exception(e)


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

                preprocessing_pipeline = PreprocessingPipeline(
                    numerical_features=self.numerical_features,
                    categorical_features=self.categorical_features,
                    outlier_features=self.outlier_features,
                    drop_columns=self.drop_columns,
                    education_map=self.education_map,
                    marital_map=self.marital_map,
                    #target_column=self.target_column,
                    #categorical_strategy='most_frequent',
                    #numerical_strategy='median',
                    #outlier_strategy='iqr'  # Update this line
                )

                preprocessor = preprocessing_pipeline.get_pipeline()

                # Apply feature engineering steps
                data_train_preprocessed = preprocessor.fit_transform(train_df)
                data_test_preprocessed = preprocessor.transform(test_df)
                logging.info("Applied the feature engineering steps on the data")
                logging.info(f"data_train_preprocessed.shape: {data_train_preprocessed.shape}")
                logging.info(f"data_test_preprocessed.shape: {data_test_preprocessed.shape}")

                # Convert transformed data to DataFrame
                #if isinstance(preprocessor.named_steps['column_transformer'], ColumnTransformer):
                #    feature_names = preprocessor.named_steps['column_transformer'].get_feature_names_out()
                #else:
                column_transformer = None
                for name, step in preprocessor.named_steps.items():
                    if isinstance(step, ColumnTransformer):
                        column_transformer = step


                # Extract feature names from ColumnTransformer
                if column_transformer:
                    feature_names = []
                    for name, transformer, columns in column_transformer.transformers_:
                        if transformer != 'drop':
                            if hasattr(transformer, 'get_feature_names_out'):
                                feature_names.extend(transformer.get_feature_names_out(columns))
                            else:
                                feature_names.extend(columns)

                # Convert transformed data to DataFrame for SHAP with feature names
                X_train_transformed_df = pd.DataFrame(data_train_preprocessed, columns=feature_names)
                X_test_transformed_df = pd.DataFrame(data_test_preprocessed, columns=feature_names)
                logging.info(f"data_train_preprocessed.head(): {X_train_transformed_df.head()}")

                preprocessor_obj_dir = os.path.dirname(self.data_transformation_config.transformed_object_file_path)

                os.makedirs(preprocessor_obj_dir, exist_ok=True)
                transformed_object_file_path = self.main_utils.save_object(
                    self.data_transformation_config.transformed_object_file_path,
                    preprocessor,
                )

                #create clusters
                create_clusters = CreateClusters(self.target_column)
                data_train_labelled = create_clusters.initialise_clustering(X_train_transformed_df)
                data_test_labelled = create_clusters.initialise_clustering(X_test_transformed_df)
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
                    transformed_test_file_path=transformed_test_file_path
                )
                logging.info("Exited the initiate_data_transformation method of Data Transformation class.")

            else:
                logging.info("Data validation step pipeline. Skipping data transformation pipeline")
                data_trasformation_artifact = None
                logging.info("Exited the initiate_data_transformation method of Data Transformation class.")


            return data_trasformation_artifact
        except Exception as e:
            self._handle_exception(e)




