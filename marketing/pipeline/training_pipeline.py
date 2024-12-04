import sys
from marketing import logging
from marketing import CustomException
from marketing.components.data_ingestion import DataIngestion
from marketing.components.data_validation import DataValidation
from marketing.components.data_transformation import DataTransformation
from marketing.components.model_trainer import ModelTrainer
from marketing.components.model_evaluation import ModelEvaluation

from marketing.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    )
from marketing.entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ClassificationMetricArtifact,
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()


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


    def start_data_transformation(self,
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact
                                  ) -> DataTransformationArtifact:

        logging.info("Entered the start_data_transformation method of TrainPipeline class")
        try:
            logging.info("Initializing DataTransformation component")
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                      data_validation_artifact=data_validation_artifact,
                                                      data_transformation_config=self.data_transformation_config)

            logging.info("Initiating data transformation")
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            logging.info("Data transformation completed successfully")
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifact
        except Exception as e:
                self._handle_exception(e)


    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        logging.info("Entered the start_model_trainer method of TrainPipeline class")
        try:
            logging.info("Initializing ModelTrainer component")
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                            model_trainer_config=self.model_trainer_config)

            logging.info("Initiating model training")
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info("Model training completed successfully")
            logging.info("Exited the start_model_trainer method of TrainPipeline class")
            return model_trainer_artifact
        except Exception as e:
            self._handle_exception(e)


    def start_model_evaluation(self,
                               data_transformation_artifact: DataTransformationArtifact,
                               model_trainer_artifact: ModelTrainerArtifact
                               ) -> ModelEvaluationArtifact:
        logging.info("Entered the start_model_evaluation method of TrainPipeline class")
        model_evalutation = ModelEvaluation(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_artifact=model_trainer_artifact,
            model_evaluation_config=self.model_evaluation_config
        )
        try:
            model_evaluation_artifact = model_evalutation.initiate_model_evaluation()
            logging.info("Model evaluation completed successfully")
            logging.info("Exited the start_model_evaluation method of TrainPipeline class")
            return model_evaluation_artifact
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
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
                )
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
                )

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
                )

            model_evaluation_artifact = self.start_model_evaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact
                )

            if not model_evaluation_artifact.is_model_accepted:
                logging.info("The model is not accpeted. Model will NOT be pushed to Production.")
                logging.info("Data run_pipeline method of TrainPipeline class")
                return None

            logging.info("Data run_pipeline method of TrainPipeline class")

        except Exception as e:
            self._handle_exception(e)