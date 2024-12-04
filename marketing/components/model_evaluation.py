import sys
import os
import pandas as pd
from dataclasses import dataclass

from marketing.entity import(
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
    ModelEvaluationArtifact,
)

from marketing.entity import ModelEvaluationConfig
from marketing import logging
from marketing import CustomException
from marketing.utils.main_utils import MainUtils
from marketing.constants import SCHEMA_FILE_PATH, PARAM_FILE_PATH, MODEL_SAVE_FORMAT
from marketing.utils.model_factory import CostModel


@dataclass
class EvaluateModelResponse:
    trained_model_score: float
    #best_model_score: float
    is_model_accepted: bool
    changed_accuracy: float
    s3_model_score: float
    best_model_name: str
    best_model_metric_artifact: ClassificationMetricArtifact

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_transformation_artifact: DataTransformationArtifact):
        """ModelEvaluation class constructor

        Args:
            model_evaluation_config (ModelEvaluationConfig): _description_
            model_trainer_artifact (ModelTrainerArtifact): _description_
            data_transformation_artifact (DataTransformationArtifact): _description_
        """
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.main_utils = MainUtils()
            self._schema_config = self.main_utils.read_yaml_file(SCHEMA_FILE_PATH)
            # Get the params from the params.yaml file
            self.param_constants = self.main_utils.read_yaml_file(filename=PARAM_FILE_PATH)
            logging.info(f"self.param_constants: {self.param_constants}")


            logging.info("ModelEvaluation object created.")
        except Exception as e:
            self._handle_exception(e)

    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)


    def evaluate_model(self) -> EvaluateModelResponse:
        """Evaluate the trained model and compare it with the production model from S3.

        Returns:
            EvaluateModelResponse: Encapsulated response with f1 scores and acceptance status.
        """
        try:
            # Load test data
            data_train = self.main_utils.load_object(file_path=self.data_transformation_artifact.transformed_train_file_path)
            data_test = self.main_utils.load_object(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Train data loaded from transformation pipeline")
            logging.info("Test data loaded from transformation pipeline")

            target_col = self._schema_config['target_column']
            logging.info(f"Obtained the target column: {target_col}")

            X_train, y_train = self.main_utils.separate_data(data_train, target_col)
            X_test, y_test = self.main_utils.separate_data(data_test, target_col)
            logging.info("Separated the train and test data into features (X) the target label (y)")

            # Evaluate trained model
            evaluation_score_param = self.param_constants['base_model']['scoring']

            trained_model_dir_path = self.model_trainer_artifact.trained_model_dir
            metric_artefacts_dir = self.model_trainer_artifact.metric_artefacts_dir
            trained_model_names = self.model_trainer_artifact.trained_model_names

            best_model_pipeline = None
            best_model_name = None
            best_model_score = 0


            #logging.info(f"Started MLflow experiment: insurance")

            logging.info("Find the best trained model based on the scoring metric")
            for model_name in trained_model_names:
                logging.info(f"Starting evaluating model: {model_name}")

                trained_model_filename = f'{model_name}_pipeline{MODEL_SAVE_FORMAT}'
                trained_model_saved_path = os.path.join(trained_model_dir_path, trained_model_filename)
                logging.info(f"trained model path: {trained_model_saved_path}")

                trained_pipeline = self.main_utils.load_object(trained_model_saved_path)
                logging.info(f'Deserialized {model_name} trained pipeline from {trained_model_saved_path}')

                # Evaluate the trained model with test data
                logging.info(f"Start prededicting with {model_name} pipeline")

                cost_model = CostModel(pipeline_model=trained_pipeline)
                y_pred, y_pred_proba = cost_model.predict(X_test)
                evaluation_scores = cost_model.evaluate(y_test, y_pred, y_pred_proba, n_classes=3)

                # Infer the model signature
                #signature = infer_signature(self.X_test, pipeline.predict(self.X_test))

                model_score = evaluation_scores[evaluation_score_param]
                logging.info(f"Model: {model_name}, Model Score ({evaluation_score_param}): {model_score}")

                if model_score > best_model_score:
                        best_model_score = model_score
                        best_model_name = model_name
                        best_model_pipeline=cost_model.pipeline_model
                        best_evaluation_scores = evaluation_scores

            trained_model_score = best_model_score
            logging.info(f"Best Trained Model -  Name: {best_model_name} with {evaluation_score_param} score: {trained_model_score}")

            # Save the best model pipeline
            os.makedirs(os.path.dirname(self.model_evaluation_config.model_trained_for_production_path), exist_ok=True)
            self.best_trained_model_path = self.model_evaluation_config.model_trained_for_production_path

            logging.info(f"Created best model file path: {self.best_trained_model_path}")
            self.main_utils.save_object(self.best_trained_model_path, best_model_pipeline)
            logging.info("Saved the best model object path")

            # Reading model config file for getting the base model score
            base_model_score = float(self.param_constants['base_model']['model_score'])
            base_model_name = self.param_constants['base_model']['model_name']
            logging.info(f"Base model -  Name: {base_model_name} with {evaluation_score_param} score: {base_model_score}")

            # If base model score is not available, set it to 0
            if base_model_score is None:
                base_model_score = 0.0


            s3_model_score = 0.0

            # If trained model score is less than the base model score, set it to the base model score
            if trained_model_score >= base_model_score:
                best_model_info  = {
                    'best_model_score': trained_model_score,
                    'best_model_name': best_model_name,
                    'best_model_scoring': evaluation_score_param
                    }
                self.main_utils.update_model_score(best_model_info)
                logging.info("Updated the best model score to model config file")



                # Load cost model object with preprocessor and model
                #cost_model = CostModel(preprocessing_obj, best_model_pipeline)

                # Evaluate S3 model if available
                #s3_model = self.get_s3_model()
                s3_model = None # Uncommet to prevent aws S3 access
                if s3_model:
                    cost_model = CostModel(pipeline_model=s3_model)
                    y_pred, y_pred_proba = cost_model.predict(X_test)
                    evaluation_scores = cost_model.evaluate(y_test, y_pred, y_pred_proba)
                    best_evaluation_scores = evaluation_scores

                    s3_model_score = evaluation_scores[evaluation_score_param]
                    logging.info(f"S3 model score: {s3_model_score}")

                # Decision making
                is_model_accepted = trained_model_score > s3_model_score
                difference = trained_model_score - s3_model_score

            else:
                is_model_accepted = False
                difference = trained_model_score
                best_model_name

            result = EvaluateModelResponse(
                    trained_model_score=trained_model_score,
                    s3_model_score=s3_model_score,
                    is_model_accepted=is_model_accepted,
                    changed_accuracy=difference,
                    best_model_name=best_model_name,
                    best_model_metric_artifact=best_evaluation_scores
                )

            logging.info(f"Model evaluation result: {result}")
            return result
        except Exception as e:
            logging.error("Error during model evaluation.", exc_info=True)
            self._handle_exception(e)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Initiate the model evaluation and create artefacts for the result.

        Returns:
            ModelEvaluationArtefacts: Object with evaluation results to be stored.
        """
        try:
            evaluate_model_response = self.evaluate_model()


            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                best_model_path=self.best_trained_model_path,
                changed_accuracy=evaluate_model_response.changed_accuracy,
                s3_model_score=evaluate_model_response.s3_model_score,
                best_model_name=evaluate_model_response.best_model_name,
                best_model_metric_artifact=evaluate_model_response.best_model_metric_artifact
            )

            logging.info("Model evaluation artefacts created.")
            return model_evaluation_artifact
        except Exception as e:
            logging.error("Error initiating model evaluation.", exc_info=True)
            self._handle_exception(e)
