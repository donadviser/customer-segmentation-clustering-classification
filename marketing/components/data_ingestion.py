import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from marketing.entity.config_entity import DataIngestionConfig
from marketing.entity.artifact_entity import DataIngestionArtifact
from marketing import logging
from marketing import CustomException
from marketing.cloud_storage.mongodb_data import MarketingCampaignData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

        try:
            self.data_ingestion_config = DataIngestionConfig()
        except Exception as e:
            self._handle_exception(e)

    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)


    def export_data_into_feature_store(self) -> pd.DataFrame:
        """Method exports data from mongodb to a csv file

        Returns:
            pd.DataFrame: Returns dataframe as artifact of data ingestion components
        """

        try:
            logging.info("Exporting data from mongodb to a csv file")
            marketing_campaign_data = MarketingCampaignData()
            df = marketing_campaign_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
                )
            logging.info(f"The shape of the dataframe df.shape: {df.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")

            df.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Data exported successfully to {feature_store_file_path}")

            return df
        except Exception as e:
            self._handle_exception(e)


    def split_data_as_train_test(self, df: pd.DataFrame) -> None:
        """Method splits the given dataframe into training and testing datasets

        Args:
            df (pd.DataFrame): Input dataframe
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): The seed for the random number generator. Defaults to 42.
            shuffle (bool, optional): Whether or not to shuffle the data before splitting. Defaults to True.
            stratify (Union[pd.Series, None], optional): The column to use for stratified sampling. Defaults to None.
        """

        try:
            logging.info("Entered split_data_as_train_test to split data as training and testing sets")
            train_data, test_data = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info(f"The shape of the training data train_data.shape: {train_data.shape}")
            logging.info(f"The shape of the testing data test_data.shape: {test_data.shape}")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving training data into file path: {self.data_ingestion_config.training_file_path}")

            train_data.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            logging.info(f"Training data saved successfully to {self.data_ingestion_config.training_file_path}")

            dir_path = os.path.dirname(self.data_ingestion_config.testing_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving testing data into file path: {self.data_ingestion_config.testing_file_path}")
            test_data.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info(f"Testing data saved successfully to {self.data_ingestion_config.testing_file_path}")
        except Exception as e:
            self._handle_exception(e)


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Method initiates data ingestion by exporting data from mongodb to a csv file, splitting it into training and testing datasets, and returning the artifact

        Returns:
            DataIngestionArtifact: Returns an object containing the exported dataframe and the splitted training and testing dataframes
        """

        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            logging.info("Initiating data ingestion")
            df = self.export_data_into_feature_store()
            logging.info("Obtained data from mongodb")

            self.split_data_as_train_test(df)
            logging.info("Performed Split data as training and testing sets")


            logging.info("Data ingestion completed successfully")
            data_ingestion_artifact =  DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            self._handle_exception(e)
