import sys
import pandas as pd
from typing import Dict
from marketing import logging
from marketing import CustomException
from marketing.constants import *
from marketing.cloud_storage.aws_storage import SimpleStorageService
import joblib
from marketing.utils.main_utils import MainUtils


class MarketingData:
    def __init__(self):
        main_utils = MainUtils()
        self.prediction_schema = main_utils.read_yaml_file(SCHEMA_FILE_PATH)


    def get_input_dataset(self, column_schema:dict, input_data):
        columns = column_schema.keys()

        input_dataset = pd.DataFrame([input_data], columns = columns)
        for key, value in column_schema.items():
            input_dataset[key] = input_dataset[key].astype(value)

        logging.info(f"input_dataset.iloc[:,:5]: {input_dataset.iloc[:,:5]} ")

        return input_dataset


    def get_input_data_frame(self) -> pd.DataFrame:
        """Get the input data as a pandas DataFrame

        Returns:
            pd.DataFrame: The data
        """

        logging.info("Entered the get_input_data_frame method of the InsuranceData class")

        try:
            data_frame = pd.DataFrame(self.prediction_schema['webapp_columns'])
            logging.info("Obtained the input data in Python dictionary format")
            logging.info("Exited the get_input_data_frame method of the MarketingData class")
            return data_frame
        except Exception as e:
            raise CustomException(e, sys)


    def form_input_dataframe(self, data):
        logging.info("Entering the form_input_dataframe method of MarketingData class")
        prediction_config = self.prediction_schema
        logging.info(f"prediction_config: {prediction_config}")
        prediction_schema = prediction_config
        column_schema = prediction_schema['webapp_columns']
        logging.info(f"************** column_schema: {column_schema} ****************")

        logging.info(f"++++++++++ data: {data} ++++++++++++")


        marketing_data = MarketingData()
        input_dataset = marketing_data.get_input_dataset(
            column_schema=column_schema,
            input_data=data
        )
        logging.info(f"========== input_dataset: {input_dataset} ==========")
        return input_dataset



class PredictionPipeline:
    def __init__(self):
        pass
        #self.s3_operations = S3Operations()
        #self.bucket_name = MODEL_BUCKET_NAME
        #self.model_path = "/Users/donadviser/github-projects/auto-insurance-claim-fraud-detection/artefacts/BestModelArtefacts/best_model_insurance_claim_fraud.pkl"


    def prepare_input_data(self, input_data:list) -> pd.DataFrame:
        """
        method: prepare_input_data

        objective: This method creates a dataframe taking the column names from prediction schema file
                       with the input values for prediction and returns it

        Args:
            input_data (list): input data

        Raises:
            CustomerException

        Returns:
            customerDataframe: pd.DataFrame: a dataframe containing the input values
        """
        try:
            logging.info(f"input_data: {input_data}")

            marketing_dataframe = MarketingData()
            data_frame = marketing_dataframe.form_input_dataframe(data = input_data)
            logging.info(f"markting datafram has been created: {data_frame}")
            return data_frame
        except Exception as e:
            raise CustomException(e,sys)


    def predict(self, X) -> float:
        """Predict the cost using the trained model

        Args:
            X (pd.DataFrame): The input data

        Returns:
            float: The predicted cost
        """

        logging.info("Entered the predict method of the CostPredictor class")

        try:
            # Load the trained model from S3 Bucket
            best_model = self.s3_operations.load_model(S3_MODEL_NAME, self.bucket_name)
            #best_model = joblib.load(self.model_path)

            # Check if the model is loaded successfully
            if not best_model:
                raise ValueError("Failed to load the best model")
            logging.info(f"Loaded best mode from s3 bucket")

            #logging.info(f"X.columns: {X.columns}")

            # Make predictions using the loaded model
            prediction = best_model.predict(X)

            logging.info("Exited the predict method of the CostPredictor class")
            return prediction
        except Exception as e:
            raise CustomException(e, sys)

