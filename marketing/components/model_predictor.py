import sys
import pandas as pd
from typing import Dict, Optional, Any
from marketing import logging
from marketing import CustomException
from marketing.constants import *
from marketing.cloud_storage.aws_storage import SimpleStorageService
import joblib
from marketing.utils.main_utils import MainUtils
from dataclasses import dataclass, asdict, fields

@dataclass
class MarketingData:
    i_d: Optional[int] = 0
    year_birth: Optional[int] = 0
    education: Optional[str] = None
    marital_status: Optional[str] = None
    income: Optional[int] = 0
    kidhome: Optional[int] = 0
    teenhome: Optional[int] = 0
    dt_customer: Optional[object] = None
    recency: Optional[int] = 0
    mnt_wines: Optional[int] = 0
    mnt_fruits: Optional[int] = 0
    mnt_meat_products: Optional[int] = 0
    mnt_fish_products: Optional[int] = 0
    mnt_sweet_products: Optional[int] = 0
    mnt_gold_prods: Optional[int] = 0
    num_deals_purchases: Optional[int] = 0
    num_web_purchases: Optional[int] = 0
    num_catalog_purchases: Optional[int] = 0
    num_store_purchases: Optional[int] = 0
    num_web_visits_month: Optional[int] = 0
    accepted_cmp1: Optional[int] = 0
    accepted_cmp2: Optional[int] = 0
    accepted_cmp3: Optional[int] = 0
    accepted_cmp4: Optional[int] = 0
    accepted_cmp5: Optional[int] = 0
    complain: Optional[int] = 0
    z_cost_contact: Optional[int] = 0
    z_revenue: Optional[int] = 0
    response: Optional[int] = 0

    def get_data(self) -> Dict:
        """
        Get the data from the form in the frontend.

        Returns:
            Dict: A dictionary with correctly typed data.
        """
        logging.info("Entered the get_data method of the InsuranceData class")

        try:
            input_data = {
                "i_d":[int(self.i_d)],
                "year_birth":[int(self.year_birth)],
                "education":[str(self.education)],
                "marital_status":[str(self.marital_status)],
                "income":[int(self.income)],
                "kidhome":[int(self.kidhome)],
                "teenhome":[int(self.teenhome)],
                "dt_customer":[str(self.dt_customer)],
                "recency":[int(self.recency)],
                "mnt_wines":[int(self.mnt_wines)],
                "mnt_fruits":[int(self.mnt_fruits)],
                "mnt_meat_products":[int(self.mnt_meat_products)],
                "mnt_fish_products":[int(self.mnt_fish_products)],
                "mnt_sweet_products":[int(self.mnt_sweet_products)],
                "mnt_gold_prods":[int(self.mnt_gold_prods)],
                "num_deals_purchases":[int(self.num_deals_purchases)],
                "num_web_purchases":[int(self.num_web_purchases)],
                "num_catalog_purchases":[int(self.num_catalog_purchases)],
                "num_store_purchases":[int(self.num_store_purchases)],
                "num_web_visits_month":[int(self.num_web_visits_month)],
                "accepted_cmp1":[int(self.accepted_cmp1)],
                "accepted_cmp2":[int(self.accepted_cmp2)],
                "accepted_cmp3":[int(self.accepted_cmp3)],
                "accepted_cmp4":[int(self.accepted_cmp4)],
                "accepted_cmp5":[int(self.accepted_cmp5)],
                "complain":[int(self.complain)],
                "z_cost_contact":[int(self.z_cost_contact)],
                "z_revenue":[int(self.z_revenue)],
                "response":[int(self.response)],
            }
            logging.info("Exited the get_data method of the InsuranceData class")
            return input_data
        except Exception as e:
            raise CustomException(e, sys)

    def get_input_data_frame(self) -> pd.DataFrame:
        """Get the input data as a pandas DataFrame"""
        logging.info("Entered the get_input_data_frame method of the InsuranceData class")
        try:
            input_data = self.get_data()
            data_frame = pd.DataFrame(input_data)
            logging.info("Obtained the input data in Python dictionary format")
            logging.info("Exited the get_input_data_frame method of the InsuranceData class")
            return data_frame
        except Exception as e:
            raise CustomException(e, sys)


    @classmethod
    def from_form(cls, form: Any) -> "MarketingData":
        """Create a MarketingData object from a form data

        Args:
            form (Any): The form data

        Returns:
            MarketingData: A MarketingData object
        """
        logging.info("Entered the from_form method of the MarketingData class")

        try:
            valid_fields = {f.name for f in fields(cls)}
            form_data = {key: getattr(form, key) for key in valid_fields if hasattr(form, key)}
            return cls(**form_data)
        except Exception as e:
            raise CustomException(e, sys)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the MarketingData instance into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the instance data.
        """
        try:
            return pd.DataFrame([asdict(self)])
        except Exception as e:
            raise CustomException(e, sys)



class PredictionPipeline:
    def __init__(self):
        pass
        #self.s3_operations = S3Operations()
        #self.bucket_name = MODEL_BUCKET_NAME
        self.model_path = "/Users/donadviser/github-projects/customer-segmentation-clustering-classification/artifact/20241204_192642/prod_model/model.pkl"


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
            #best_model = self.s3_operations.load_model(S3_MODEL_NAME, self.bucket_name)
            best_model = joblib.load(self.model_path)

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

