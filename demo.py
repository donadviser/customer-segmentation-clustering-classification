from marketing.constants import *
from marketing import logging
from marketing.import CustomException
import urllib.request as request
import os

SOURCE_URL



logging.info("Download started...")
filename, headers = request.urlretrieve(
url=SOURCE_URL,
filename=os.path.join("artifacts", "download")
)
logging.info(f"{filename} download! with following info: \n{headers}")


    # This method will fetch data from mongoDB
    def get_data_from_local_data_file(self) -> pd.DataFrame:
        """
        Get the data from csv file.

        Returns:
            pd.DataFrame: The data from csv file.
        """
        logging.info("Entered the get_data_from_file method of DataIngestion class")
        try:

            filename = self.data_ingestion_config.DOWNLOADED_DATA_FILE_PATH
            df = pd.read_csv(filename)
            logging.info(f"The length of the column in the dataset is: {len(df.columns)}\n The columns are: {df.columns}")
            logging.info(f"Obtaied the dataframe from local data file:  {filename}")
            data = (
                df
                .pipe(self.UTILS.rename_columns_to_snake_case)
                )
            logging.info("Renamed the columns to snake case")
            logging.info("Exited the get_data_from_local_data_file method of DataIngestion class")
            return data
        except Exception as e:
            raise CustomException(e, sys)