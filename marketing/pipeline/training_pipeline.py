import sys
from marketing import logging
from marketing import CustomException
from marketing.components.data_ingestion import DataIngestion

from marketing.entity.config_entity import (
    DataIngestionConfig,
    )
from marketing.entity.artifact_entity import (
    DataIngestionArtifact,
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)


    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Start data ingestion by exporting data from MongoDB to a CSV file, splitting it into training and testing datasets, and returning the artifact.

        Returns:
            DataIngestionArtifact: Returns an object containing the exported dataframe and the splitted training and testing dataframes.
        """
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Initializing DataIngestion component")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)

            logging.info("Initiating data ingestion")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed successfully")
            return data_ingestion_artifact
        except Exception as e:
            self._handle_exception(e)

    def run_pipeline(self,) -> None:
        """
        Run the entire data ingestion, preprocessing, and model training pipeline.
        """
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            logging.info("Starting data ingestion")
            data_ingestion_artifact = self.start_data_ingestion()

            logging.info("Data ingestion completed successfully")

        except Exception as e:
            self._handle_exception(e)