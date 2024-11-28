import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import optuna
from typing import Union, Dict, Tuple
from typing_extensions import Annotated
from optuna.samplers import TPESampler
from dataclasses import dataclass

from marketing import logging
from marketing import CustomException
from marketing.constants import MODEL_CONFIG_FILE, MODEL_SAVE_FORMAT, PARAM_FILE_PATH
from marketing.entity import ModelTrainerConfig
from marketing.entity import (
    DataIngestionArtefacts,
    DataTransformationArtefacts,
    ModelTrainerArtefacts,
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler,
    OrdinalEncoder, PowerTransformer,
    RobustScaler, MinMaxScaler,
    FunctionTransformer)
from sklearn.model_selection import   cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from sklearn.decomposition import PCA
from imblearn.over_sampling import (
    RandomOverSampler,
    ADASYN,
)
from imblearn.under_sampling import (
    RandomUnderSampler,
    NearMiss,
)
from imblearn.combine import (
    SMOTEENN,
    SMOTETomek
)

from marketing.utils.custom_transformers import (
    DropRedundantColumns,
    CreateNewFeature,
    LogTransformer,
    OutlierDetector,
    ReplaceValueTransformer,
    OutlierHandler,
)


class PipelineManager:
    """
    A class that handles both building and modifying pipelines dynamically.
    This class supports both scikit-learn's Pipeline and imbalanced-learn's Pipeline.

    It allows the construction of the initial pipeline and the insertion of steps
    at any position within the pipeline.
    """

    def __init__(self, education_map, marital_map, drop_columns, numerical_features,
                 categorical_features, outlier_features, pipeline_type="Pipeline",
                 trial: optuna.Trial = None, **kwargs):
        """
        Initialize the PreprocessingPipeline with necessary parameters.

        Args:
            education_map (dict): Parameters for mapping education categorical feature into numbers.
            marital_map (dict): Parameters for mapping marital categorical feature into numbers.
            drop_columns (list): Columns to be dropped from the dataset.
            numerical_features (list): List of numerical features for processing.
            categorical_features (list): List of categorical features for OneHot encoding.
            outlier_features (list): Features that require power transformation.
            pipeline_type (str): Type of pipeline, default is "Pipeline".
            trial (optuna.Trial, optional): Optuna trial object for hyperparameter optimization.
            kwargs (dict): Additional keyword arguments.
        """
        self.education_map = education_map
        self.marital_map = marital_map
        self.drop_columns = drop_columns
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.outlier_features = outlier_features
        self.pipeline_type = pipeline_type
        self.trial = trial
        self.kwargs = kwargs  # Assign kwargs directly without unpacking


        #Initialize the PipelineManager with a specified pipeline type.
        if self.pipeline_type == 'ImbPipeline':
            self.pipeline = ImbPipeline(steps=[])
        elif self.pipeline_type == 'Pipeline':
            self.pipeline = Pipeline(steps=[])
        else:
            raise ValueError("Unsupported pipeline type. Choose 'ImbPipeline' or 'Pipeline'.")


        self.numerical_strategy = kwargs.get('numerical_strategy', None)
        self.categorical_strategy = kwargs.get('categorical_strategy',None)
        self.outlier_strategy = kwargs.get('outlier_strategy', None)

    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)


    def add_step(self, step_name, step_object, position=None):
        """
        Add a transformation step to the pipeline.

        Args:
            step_name (str): Name of the step to add.
            step_object (object): The transformer or estimator object (e.g., scaler, classifier).
            position (int or None): Optional; the position to insert the step.
                                    If None, the step is appended at the end of the pipeline.
        """
        if position is None:
            self.pipeline.steps.append((step_name, step_object))
        else:
            self.pipeline.steps.insert(position, (step_name, step_object))

    def remove_step(self, step_name):
        """
        Remove a step from the pipeline by its name.

        Args:
            step_name (str): The name of the step to remove.
        """
        self.pipeline.steps = [(name, step) for name, step in self.pipeline.steps if name != step_name]

    def replace_step(self, step_name, new_step_object):
        """
        Replace an existing step in the pipeline with a new step.

        Args:
            step_name (str): The name of the step to replace.
            new_step_object (object): The new transformer or estimator object.
        """
        for i, (name, step) in enumerate(self.pipeline.steps):
            if name == step_name:
                self.pipeline.steps[i] = (step_name, new_step_object)
                break

    def get_pipeline(self):
        """
        Get the constructed or modified pipeline.

        Returns:
            Pipeline: The constructed or modified pipeline object.
        """
        return self.pipeline

    def instantiate_numerical_simple_imputer(self, strategy: str=None, fill_value: int=-1) -> SimpleImputer:
        if strategy is None and self.trial:
            strategy = self.trial.suggest_categorical('numerical_imputer', ['mean', 'median', 'most_frequent'])
        return SimpleImputer(strategy=strategy, fill_value=fill_value)

    def instantiate_categorical_simple_imputer(self, strategy: str=None, fill_value: str='missing') -> SimpleImputer:
        if strategy is None and self.trial:
            strategy = self.trial.suggest_categorical('categorical_imputer', ['most_frequent', 'constant'])
        return SimpleImputer(strategy=strategy, fill_value=fill_value)

    def instantiate_outliers(self, strategy: str=None) -> Union[PowerTransformer, FunctionTransformer, OutlierDetector]:
        """
        Instantiate outlier handling method: PowerTransformer, LogTransformer, or OutlierDetector.

        Args:
            trial (optuna.Trial, optional): The trial object for hyperparameter optimization.

        Returns:
            Union[PowerTransformer, FunctionTransformer, OutlierDetector]: The selected outlier handling method.
        """
        # Suggest from available options
        options = ['power_transform', 'log_transform', 'iqr_clip', 'iqr_median', 'iqr_mean']
        if self.trial and strategy is None:
            strategy = self.trial.suggest_categorical('outlier_strategy', options)
        else:
            strategy = strategy  # Default to first option if no trial is provided

        if strategy == 'power_transform':
            return PowerTransformer(method='yeo-johnson')
        elif strategy == 'log_transform':
            return LogTransformer()
            #return FunctionTransformer(np.log1p)  # Log transformation
        elif strategy in ['iqr_clip', 'iqr_median', 'iqr_mean']:
            return OutlierHandler(strategy=strategy)  # Instantiate OutlierDetector
        else:
            raise ValueError(f"Unknown strategy for outlier handling: {strategy}")


    def build(self, step_name=None, **column_transformer_strategy):
        """
        Build the preprocessing pipeline with feature creation, transformation,
        imputation, scaling, and encoding steps.

        Returns:
            Transformer: The appropriate transformer for the given step.
        """

        if step_name == "create_new_features":
            return CreateNewFeature(education_map=self.education_map, marital_map=self.marital_map)

        if step_name == "replace_class":
            return ReplaceValueTransformer(old_value="?", new_value='missing')

        if step_name == "drop_cols":
            return DropRedundantColumns(redundant_cols=self.drop_columns)

        if step_name == 'column_transformer':
            return ColumnTransformer(
                transformers=[
                    ('cat', Pipeline([
                        ('imputer', self.instantiate_categorical_simple_imputer(strategy=self.categorical_strategy)),
                        #('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
                    ]), self.categorical_features),

                    ('num', Pipeline([
                        ('imputer', self.instantiate_numerical_simple_imputer(strategy=self.numerical_strategy)),
                        #('scaler', StandardScaler())  # Add scaler if needed
                    ]), self.numerical_features),



                    ('ord', Pipeline([
                        ('imputer', self.instantiate_categorical_simple_imputer(strategy=self.categorical_strategy)),
                        ('ordinal', OrdinalEncoder())
                    ]), self.categorical_features),

                    ('out', Pipeline([
                        ('imputer', self.instantiate_numerical_simple_imputer(strategy=self.numerical_strategy)),
                        ('outlier', self.instantiate_outliers(strategy=self.outlier_strategy))
                    ]), self.outlier_features_features),
                ],
                remainder='passthrough'
            )



class ResamplerSelector:
    """
    A class to select and return a resampling algorithm based on a given parameter or
    from a trial suggestion if available.

    Attributes:
        trial (optuna.trial, optional): The trial object for hyperparameter optimization.
    """

    def __init__(self, trial=None, random_state=42):
        """
        Initialize the ResamplerSelector with an optional trial for hyperparameter optimization.

        Args:
            trial (optuna.trial, optional): An optional trial object for suggesting resampling strategies.
            random_state (int): Random seed for reproducibility. Default is 42.
        """
        self.trial = trial
        self.random_state = random_state

    def get_resampler(self, resampler=None):
        """
        Return the resampling algorithm based on the provided `resampler` parameter.
        If `resampler` is not given, it is suggested from the trial.

        Args:
            resampler (str, optional): The resampling method ('RandomOverSampler', 'ADASYN', etc.).
                                       If not provided, it will be suggested from the trial (if available).

        Returns:
            resampler_obj (object): The resampling instance based on the selected method.
        """
        if resampler is None and self.trial:
            resampler = self.trial.suggest_categorical(
                'resampler', ['RandomOverSampler', 'ADASYN', 'RandomUnderSampler', 'NearMiss',
                              'SMOTEENN', 'SMOTETomek']
            )

            """resampler = self.trial.suggest_categorical(
                'resampler', ['SMOTEENN',]
            )"""

        if resampler == 'RandomOverSampler':
            return RandomOverSampler(random_state=self.random_state)
        elif resampler == 'ADASYN':
            return ADASYN(random_state=self.random_state)
        elif resampler == 'RandomUnderSampler':
            return RandomUnderSampler(random_state=self.random_state)
        elif resampler == 'NearMiss':
            return NearMiss()
        elif resampler == 'SMOTEENN':
            return SMOTEENN(random_state=self.random_state, sampling_strategy='minority' )
        elif resampler == 'SMOTETomek':
            return SMOTETomek(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown resampler: {resampler}")


class ScalerSelector:
    """
    A class to select and return a scaling algorithm based on a given parameter or
    from a trial suggestion if available.

    Attributes:
        trial (optuna.trial, optional): The trial object for hyperparameter optimization.
    """

    def __init__(self, trial=None):
        """
        Initialize the ScalerSelector with an optional trial for hyperparameter optimization.

        Args:
            trial (optuna.trial, optional): An optional trial object for suggesting resampling strategies.
        """
        self.trial = trial

    def get_scaler(self, scaler_name=None):
        """
        Return the scaling algorithm based on the provided `scaler_name` parameter.
        If `scaler_name` is not given, it is suggested from the trial.

        Args:
            scaler_name (str, optional): The scalring method ('MinMaxScaler', 'StandardScaler', etc.).
                                       If not provided, it will be suggested from the trial (if available).

        Returns:
            rscaler_obj (object): The scaling instance based on the selected method.
        """

        # -- Instantiate scaler (skip scaler for CatBoostClassifier as it handles categorical features internally)
        if scaler_name is None and self.trial:
            scaler_name = self.trial.suggest_categorical("scaler", ['minmax', 'standard', 'robust'])
            #scaler_name = self.trial.suggest_categorical("scaler", ['robust'])

        if scaler_name == "minmax":
            return MinMaxScaler()
        elif scaler_name == "standard":
            return StandardScaler()
        elif scaler_name == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler_name}")


class DimensionalityReductionSelector:
    """
    A class to select and return a dimensionality reduction algorithm based on a given parameter
    or from a trial suggestion if available.

    Attributes:
        trial (optuna.trial, optional): The trial object for hyperparameter optimization.
    """

    def __init__(self, trial=None):
        """
        Initialize the DimensionalityReductionSelector with an optional trial for hyperparameter optimization.

        Args:
            trial (optuna.trial, optional): An optional trial object for suggesting dimensionality reduction strategies.
        """
        self.trial = trial

    def get_dimensionality_reduction(self, dim_red=None, pca_n_components=5):
        """
        Return the dimensionality reduction algorithm based on the provided `dim_red` parameter.
        If `dim_red` is not given, it is suggested from the trial.

        Args:
            dim_red (str, optional): The dimensionality reduction method ('PCA' or None). If not provided,
                                     it will be suggested from the trial (if available).

        Returns:
            dimen_red_algorithm (object or str): PCA algorithm or 'passthrough'.
        """
        if dim_red is None and self.trial:
            dim_red = self.trial.suggest_categorical("dim_red", ["PCA", None])

        if dim_red == "PCA":
            if self.trial:
                pca_n_components = self.trial.suggest_int("pca_n_components", 2, 30)
            else:
                pca_n_components = pca_n_components  # Default value if trial is not provided
            dimen_red_algorithm = PCA(n_components=pca_n_components)
        else:
            dimen_red_algorithm = 'passthrough'

        return dimen_red_algorithm





"""def get_pipeline(
    numerical_features,
    categorical_features,
    outlier_features,
    drop_columns,
    education_map,
    marital_map,
    pipeline_type,
    target_column,
    trial=None
):

    # Got the Preprocessed Pipeline containting Data Cleaning and Column Transformation
    try:
        pipeline_manager = PipelineManager(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            outlier_features=outlier_features,
            drop_columns=drop_columns,
            education_map=education_map,
            marital_map=marital_map,
            pipeline_type=pipeline_type,
            target_column=target_column,
            categorical_strategy='most_frequent',
            numerical_strategy='median',
            outlier_strategy='iqr',
        )


        # Initialize the manager with the preferred pipeline type ('ImbPipeline' or 'Pipeline')
        pipeline_manager = PipelineManager(pipeline_type='ImbPipeline')

        pipeline_manager.add_step('create_new_features', pipeline_manager.build(step_name='create_new_features'), position=0)
        pipeline_manager.add_step('replace_class', pipeline_manager.build(step_name='replace_class'), position=1)
        pipeline_manager.add_step('drop_cols', pipeline_manager.build(step_name='drop_cols'), position=2)
        pipeline_manager.add_step('column_transformer', pipeline_manager.build(step_name='column_transformer'), position=3)

        # Add the resampler step based on the provided resample name or trial suggestion
        resample_selector = ResamplerSelector(trial=trial)
        resampler_obj = resample_selector.get_resampler()
        pipeline_manager.add_step('resampler', resampler_obj, position=4)


        # Add the scaler step based on the provided resample name or trial suggestion
        scaler_selector = ScalerSelector(trial=trial)
        scaler_obj = scaler_selector.get_scaler()
        pipeline_manager.add_step('scaler', scaler_obj, position=5)


        # Add the Dimensional Reduction step based on the provided parameter or trial suggestion
        #dim_red_selector = DimensionalityReductionSelector(trial=trial)
        #dim_red_obj = dim_red_selector.get_dimensionality_reduction()
        #pipeline_manager.add_step('dim_reduction', dim_red_obj, position=6)

        # Create an instance of the ModelFactory class with best_model and best_params
        model_factory = ModelFactory(model_name, model_hyperparams)
        model_obj = model_factory.get_model_instance()
        pipeline_manager.add_step('model', model_obj, position=7)

        pipeline = pipeline_manager.get_pipeline()

        return pipeline

    except Exception as e:
        self._handle_exception(e)"""