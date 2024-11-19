import sys
from marketing import logging
from marketing import CustomException
from marketing.components.data_ingestion import DataIngestion
from marketing.components.data_validation import DataValidation

from marketing.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    )
from marketing.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()


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


    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        """
        Start data validation by comparing the data drift between the training and testing datasets, and returning the artifact.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): The artifact containing the training and testing dataframes.

        Returns:
            DataValidationArtifact: Returns an object containing the data drift file path, validation status, and validation message.
        """
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config)

            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed successfully")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact
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
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

            logging.info("Data ingestion completed successfully")

        except Exception as e:
            self._handle_exception(e)