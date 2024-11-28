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


    def get_pca_object(self) -> object:
        """
        Apply PCA to the preprocessed data.

        Args:
            preprocessed_data (pd.DataFrame): Preprocessed data.
        """
        logging.info("Entered the get__pca_object method of CreateClusters class")
        try:
            pca = PCA(**self.pca_config.__dict__)
            return pca
        except Exception as e:
            raise CustomException(e, sys)


    def initialise_clustering(self,
                            transformed_train: pd.DataFrame,
                            transformed_test: pd.DataFrame,
                            train_df: pd.DataFrame,
                            test_df: pd.DataFrame
                            ) -> pd.DataFrame:
        """
        Initialise clustering using KMeans.

        Args:
            preprocessed_data (pd.DataFrame): Preprocessed data.
        """
        logging.info("Entered the initialise_clustering method of CreateClusters class")
        try:






            pca = self.get_pca_object()

            reduced_train = pca.fit_transform(transformed_train)
            reduced_test = pca.transform(transformed_test)

            kmeans = KMeans(**self.clustering_config.__dict__)
            logging.info(f"KMeans clustering completed for {self.clustering_config.n_clusters} clusters.")

            train_clusters = kmeans.fit_predict(reduced_train)
            test_clusters = kmeans.predict(reduced_test)

            train_df[self.target_column] = train_clusters
            test_df[self.target_column] = test_clusters
            logging.info("Clustering completed and target column added to train and test datasets.")

            return train_df, test_df
        except Exception as e:
            raise CustomException(e, sys)
