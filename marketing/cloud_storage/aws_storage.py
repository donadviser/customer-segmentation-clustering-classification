from marketing.configuration.aws_connection import S3Client
from io import StringIO
from typing import Union, List
import os
import sys
from marketing import logging
from marketing import CustomException
from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket
import pandas as pd
import pickle



class SimpleStorageService:
    """
    Simple Storage Service class for interacting with Amazon S3.
    """
    def __init__(self):
        s3_client = S3Client()
        self.s3_client = s3_client.s3_client
        self.s3_resource = s3_client.s3_resourse


    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)


    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        """
        Check if S3 key path is available.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.fiter(Prefix=s3_key)]
            return len(file_objects) > 0
        except ClientError as e:
            raise self._handle_exception(e)


    @staticmethod
    def read_object(object_name: str, decode: bool =True, make_readable: bool = False) -> Union[StringIO, str]:
        """
        Read an object from S3.

        Args:
            object_name (str): Name of the S3 object.
            decode (bool, optional): Whether to decode the object content from bytes to string. Defaults to True.
            make_readable (bool, optional): Whether to make the object content readable. Defaults to False.

        Returns:
            Union[StringIO, str]: The object content as a string or a StringIO object.
        """
        logging.info("Entered the read_object method of SimpleStorageService class")
        try:
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode is True
                else object_name.get()["Body"].read()
            )

            conv_func = lambda: StringIO(func) if make_readable is True else func()
            logging.info("Exited the read_object method of SimpleStorageService class")
            return conv_func()

        except ClientError as e:
            raise CustomException(e, sys)


    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Get an S3 bucket by name.

        Args:
            bucket_name (str): Name of the S3 bucket.

        Returns:
            Bucket: The S3 bucket object.
        """
        logging.info("Entered the get_bucket method of SimpleStorageService class")
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info("Exited the get_bucket method of SimpleStorageService class")
            return bucket

        except ClientError as e:
            raise self._handle_exception(e)


    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """
        Get a file object from an S3 bucket.

        Args:
            filename (str): Name of the file.
            bucket_name (str): Name of the S3 bucket.

        Returns:
            Union[List[object], object]: A list of file objects or a single file object.
        """
        logging.info("Entered the get_file_object method of SimpleStorageService class")
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]
            func = lambda x: x[0] if len(x) == 1 else x
            file_objs = func(file_objects)
            logging.info("Exited the get_file_object method of SimpleStorageService class")
            return file_objs

        except ClientError as e:
            raise self._handle_exception(e)


    def load_model(self, model_name: str, bucket_name: str, model_dir: str=None) -> object:
        """
        Load a model from an S3 bucket.

        Args:
            model_name (str): Name of the model.
            bucket_name (str): Name of the S3 bucket.
            model_dir (str, optional): Directory where the model file is located. Defaults to None.

        Returns:
            object: The loaded model object.
        """
        logging.info("Entered the load_model method of SimpleStorageService class")

        try:
            func = (
                lambda: model_name
                if model_dir is None
                else f"{model_dir}/{model_name}"
            )

            model_file = func()
            file_object = self.get_file_object(model_file, bucket_name)
            model_obj = self.read_object(file_object, decode=False)
            model=pickle.loads(model_obj)
            logging.info("Exited the load_model method of SimpleStorageService class")
            return model
        except ClientError as e:
            raise self._handle_exception(e)


    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Create a folder in an S3 bucket.

        Args:
            folder_name (str): Name of the folder.
            bucket_name (str): Name of the S3 bucket.
        """
        logging.info("Entered the create_folder method of SimpleStorageService class")
        try:
            self.s3_resource.Object(bucket_name, folder_name).load()

        except ClientError as e:
            if e.response['Error']['Code']!= '404':
                folder_obj = folder_name + "/"
                self.s3_client.put_object(Bucket=bucket_name, key=folder_obj)
            else:
                logging.info(f"Folder {folder_name} already exists")

            logging.info("Exited the create_folder method of SimpleStorageService class")
        except ClientError as e:
            raise self._handle_exception(e)

    def upload_file(self, from_filename: str, to_filename: str,  bucket_name: str,  remove: bool = True):

        """
        This method uploads a file from the local system to an S3 bucket.

        Args:
            from_filename (str): Name of the file to be uploaded.
            to_filename (str): Name of the file in the S3 bucket.
            bucket_name (str): Name of the S3 bucket.
            remove (bool, optional): Whether to remove the uploaded file from the local system. Defaults to True.

        Returns:
            None

        """


        logging.info("Entered the upload_file method of S3Operations class")

        try:
            logging.info(
                f"Uploading {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            self.s3_resource.meta.client.upload_file(
                from_filename, bucket_name, to_filename
            )

            logging.info(
                f"Uploaded {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            if remove is True:
                os.remove(from_filename)

                logging.info(f"Remove is set to {remove}, deleted the file")

            else:
                logging.info(f"Remove is set to {remove}, not deleted the file")

            logging.info("Exited the upload_file method of S3Operations class")

        except Exception as e:
            raise self._handle_exception(e)

    def upload_df_as_csv(self,data_frame: pd.DataFrame,local_filename: str, bucket_filename: str,bucket_name: str,) -> None:
        """
        Method Name :   upload_df_as_csv
        Description :   This method uploads the dataframe to bucket_filename csv file in bucket_name bucket

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the upload_df_as_csv method of S3Operations class")

        try:
            data_frame.to_csv(local_filename, index=None, header=True)

            self.upload_file(local_filename, bucket_filename, bucket_name)

            logging.info("Exited the upload_df_as_csv method of S3Operations class")

        except Exception as e:
            raise self._handle_exception(e)


    def get_df_from_object(self, object_: object) -> pd.DataFrame:
        """
        Method Name :   get_df_from_object
        Description :   This method reads the object content as csv and returns a dataframe

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """

        logging.info("Entered the get_df_from_object method of S3Operations class")

        try:
            content = self.read_object(object_, make_readable=True)
            df = pd.read_csv(content, na_values="na")
            logging.info("Exited the get_df_from_object method of S3Operations class")
            return df
        except Exception as e:
            raise self._handle_exception(e)

    def read_csv(self, filename: str, bucket_name: str) -> pd.DataFrame:
        """
        Method Name :   read_csv
        Description :   This method reads a csv file from bucket_name bucket and returns a dataframe

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """

        logging.info("Entered the read_csv method of S3Operations class")

        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            df = self.get_df_from_object(csv_obj)
            logging.info("Exited the read_csv method of S3Operations class")
            return df
        except Exception as e:
            raise self._handle_exception(e)
