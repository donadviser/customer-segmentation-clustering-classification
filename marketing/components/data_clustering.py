import sys
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from marketing.constants import *
from marketing.entity.config_entity import PCAConfig, ClusteringConfig
from marketing import logging
from marketing import CustomException

class CreateClusters:
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.pca_config = PCAConfig()
        self.clustering_config = ClusteringConfig()


    def get_dataset_using_pca(self, preprocessed_data: pd.DataFrame) -> object:
        """
        Apply PCA to the preprocessed data.

        Args:
            preprocessed_data (pd.DataFrame): Preprocessed data.
        """
        logging.info("Entered the get_dataset_using_pca method of CreateClusters class")
        try:
            logging.info(f"Preprocessed data shape: {preprocessed_data.shape}")
            pca = PCA(**self.pca_config.__dict__)
            pca_data = pca.fit_transform(preprocessed_data)
            logging.info(f"PCA transformed data shape: {pca_data.shape}")
            return pca_data
        except Exception as e:
            raise CustomException(e, sys)


    def initialise_clustering(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Initialise clustering using KMeans.

        Args:
            preprocessed_data (pd.DataFrame): Preprocessed data.
        """
        logging.info("Entered the initialise_clustering method of CreateClusters class")
        try:
            pca_data = self.get_dataset_using_pca(preprocessed_data)
            kmeans = KMeans(**self.clustering_config.__dict__)
            kmeans_data = kmeans.fit_predict(pca_data)
            logging.info(f"KMeans clustering completed for {self.clustering_config.n_clusters} clusters.")
            preprocessed_data[self.target_column] = kmeans_data
            return preprocessed_data
        except Exception as e:
            raise CustomException(e, sys)
