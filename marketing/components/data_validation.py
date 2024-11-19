import os
import sys
import json
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from marketing import logging
from marketing import CustomException
from marketing.utils.main_utils import MainUtils
from marketing.entity.config_entity import DataValidationConfig
from marketing.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from marketing.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        """DataValidation class constructor

        Args:
            data_validation_config (DataValidationConfig): _description_
            data_ingestion_artifact (DataIngestionArtifact): _description_
        """


        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.main_utils = MainUtils()
        self._schema_config = self.main_utils.read_yaml_file(SCHEMA_FILE_PATH)


    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates the number of columns in the given DataFrame against the schema.

        Args:
            dataframe (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if the number of columns matches the schema, False otherwise.
        """

        try:
            if dataframe.shape[1] != len(self._schema_config["columns"]):
                logging.error(f"Number of columns in the dataframe ({dataframe.shape[1]}) does not match the schema ({len(self._schema_config['columns'])})")
                return False
            logging.info(f"The required number of columns are present")
            return True
        except Exception as e:
            self._handle_exception(e)


    def does_column_exist(self, df: pd.DataFrame) -> bool:
        """
        Validates if all the required columns exist in the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if all required columns exist, False otherwise.
        """

        try:
            ingested_columns = df.columns
            missing_numerical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in ingested_columns:
                    missing_numerical_columns.append(column)
            if missing_numerical_columns:
                logging.error(f"The following numerical columns are missing from the dataframe: {', '.join(missing_numerical_columns)}")

            missing_categorical_columns = []
            for column in self._schema_config["categorical_columns"]:
                if column not in ingested_columns:
                    missing_categorical_columns.append(column)
            if missing_categorical_columns:
                logging.error(f"The following categorical columns are missing from the dataframe: {', '.join(missing_categorical_columns)}")

            return len(missing_numerical_columns) == 0 and len(missing_categorical_columns) == 0
        except Exception as e:
            self._handle_exception(e)


    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file into a DataFrame.

        Args:
            file_path (str): The file path of the CSV file.

        Returns:
            pd.DataFrame: The DataFrame read from the CSV file.
        """

        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)


    def detect_dataset_drift(self, reference_df: pd.DataFrame, production_df: pd.DataFrame) -> bool:
        """
        Detects dataset drift between two DataFrames.

        Args:
            reference_df (pd.DataFrame): The reference DataFrame.
            current_df (pd.DataFrame): The current DataFrame.

        Returns:
            bool: True if dataset drift is detected, False otherwise.
        """

        try:
            # Create a report object with data drift preset
            report = Report(metrics=[DataDriftPreset()])

            # Run the report calculation
            report.run(reference_data=reference_df, current_data=production_df)

            # Getting data drift report in json format
            report_json = report.json()
            json_report = json.loads(report_json)

            # Saving the json report in artefacts directory
            self.main_utils.write_json_to_yaml(
                json_file=json_report,
                yaml_file_path=self.data_validation_config.drift_report_file_path,
                )


            n_features = json_report["metrics"][0]["result"]["number_of_columns"]
            n_drifted_features = json_report["metrics"][0]["result"]["number_of_drifted_columns"]
            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            return json_report["metrics"][0]["result"]["dataset_drift"]
        except Exception as e:
            self._handle_exception(e)


    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates data validation by reading data from the CSV file, validating its structure, and detecting dataset drift.

        Returns:
            DataValidationArtifact: The artifact containing the validated DataFrame and drift status.
        """

        try:
            validation_error_message = ""
            logging.info("Initiating data validation")
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.train_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataset: {status}")
            if not status:
                validation_error_message += "Data validation failed: Number of columns in the training data does not match the schema\n"

            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataset: {status}")
            if not status:
                validation_error_message += "Data validation failed: Number of columns in the testing data does not match the schema\n"

            status = self.does_column_exist(df=train_df)
            logging.info(f"All required columns present in training dataset: {status}")
            if not status:
                validation_error_message += "Data validation failed: Some required columns are missing from the training data\n"

            status = self.does_column_exist(df=test_df)
            logging.info(f"All required columns present in testing dataset: {status}")
            if not status:
                validation_error_message += "Data validation failed: Some required columns are missing from the testing data\n"

            validation_status = len(validation_error_message) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info("Dataset drift detected")
                    validation_error_message += "Data validation failed: Dataset drift detected\n"
                else:
                    logging.info("Dataset drift not detected")
            else:
                logging.error(f"validation_error_message: {validation_error_message}")


            data_validation_artifact = DataValidationArtifact(
                data_drift_file_path=self.data_validation_config.drift_report_file_path,
                validation_status=validation_status,
                validation_message=validation_error_message,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            self._handle_exception(e)