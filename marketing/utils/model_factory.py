import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import optuna
from typing import Dict, Tuple
from typing_extensions import Annotated
from optuna.samplers import TPESampler


from marketing import logging
from marketing import CustomException
from marketing.constants import MODEL_CONFIG_FILE, MODEL_SAVE_FORMAT, PARAM_FILE_PATH

from sklearn.model_selection import   cross_val_score, StratifiedKFold


from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss,confusion_matrix,
                             average_precision_score)

from marketing.utils.pipeline_manager import (
    PipelineManager,
    ResamplerSelector,
    ScalerSelector,
    DimensionalityReductionSelector
)
from importlib import import_module
from collections import namedtuple
from sklearn.metrics import make_scorer
from datetime import datetime
from marketing.utils.main_utils import MainUtils


BestModel = namedtuple("BestModel", ["model_name", "best_pipeline", "best_model", "best_params", "best_score", "trial_summary", "scores_list"])


class CostModel:
    def __init__(self,
                 pipeline_model: object,
                 X_train: pd.DataFrame = None,
                 y_train: pd.DataFrame = None,
                 preprocess_pipeline: object=None,
                ):
        """
        Initialize the CostModel class

        Args:
            pipeline_model (object): model or model in the pipeline
            X_train (pd.DataFrame, optional): data (features). Needed if training is required
            y_train (pd.DataFrame, optional): labels. Needed if training is required
            preprocess_pipeline (object, optional): _description_. Defaults to None.
        """
        self.preprocess_pipeline = preprocess_pipeline
        self.pipeline_model = pipeline_model
        self.X_train = X_train
        self.y_train = y_train

    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)


    def train(self) -> object:
        """Train the model with the provided data."""
        try:
            if self.preprocess_pipeline:
                self.X_train = self.preprocess_pipeline.fit_transform(self.X_train)
            self.pipeline_model.fit(self.X_train, self.y_train)
            return self.pipeline_model
        except Exception as e:
            self._handle_exception(e)


    def predict(self, X_test) -> Tuple[
        Annotated[float, "y_pred"],
        Annotated[float, "y_pred_proba"]
        ]:
        """
        This method predicts the data

        Args:
            X (pd.DataFrame): The data to be predicted.

        Returns:
            float: The predicted data.
        """
        try:
            if self.preprocess_pipeline:
                X_test = self.preprocess_pipeline.transform(X_test)
                logging.info("Transformed the X_test using preprocess_pipeline to get predictions")

            y_pred = self.pipeline_model.predict(X_test)
            y_pred_proba = self.pipeline_model.predict_proba(X_test)

            return y_pred, y_pred_proba
        except Exception as e:
            self._handle_exception(e)


    def evaluate(self, y_true, y_pred, y_pred_proba=None, n_classes=None)-> Dict[str, float]:
        metrics = {}
        try:
            if n_classes is None:
                n_classes = len(np.unique(y_true))
            logging.info(f"No of unique classes in y_true: {n_classes}")

            # Accuracy
            metrics['accuracy'] = accuracy_score(y_true, y_pred)

            # Precision, Recall, F1 (Macro and Weighted)
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')

            # AUC ROC (Macro and Weighted)
            metrics['auc_macro'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            metrics['auc_weighted'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')

            # Log Loss
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)

            # False Positive Rate
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
            fprs = []
            for i in range(n_classes):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                fprs.append(fp / (fp + tn))
            metrics['false_positive_rate'] = np.mean(fprs)

            # Average Precision Score (Macro and Weighted)
            metrics['average_precision_macro'] = average_precision_score(y_true, y_pred_proba, average='macro')
            metrics['average_precision_weighted'] = average_precision_score(y_true, y_pred_proba, average='weighted')


            return metrics
        except Exception as e:
            self._handle_exception(e)


    def __repr__(self) -> str:
        return f"{type(self.pipeline_model).__name__}()"

    def __str__(self) -> str:
        return f"{type(self.pipeline_model).__name__}()"


class HyperparameterTuner:
    """
    HyperparameterTuner to return hyperparameters for each classifier.
    """

    def __init__(self):
        pass

    def get_params(self, trial: optuna.trial.Trial, model_name: str, classifier_params):
        """
        Get hyperparameters for a specified classifier.

        Args:
            trial (optuna.Trial): Optuna trial instance for hyperparameter suggestions.
            model_name (str): Name of the classifier.

        Returns:
            dict: A dictionary of hyperparameters.
        """


        # Fetch classifier-specific parameters
        model_params = classifier_params.get(model_name, {})
        params = {}

        if model_name == "LogisticRegression":
            # Suggest penalty from a unified set

            # Basic hyperparameters
            params = {
                "solver": trial.suggest_categorical('solver', ['newton-cholesky', 'lbfgs', 'liblinear', 'sag', 'saga']),
                "max_iter": trial.suggest_int('max_iter', 10000, 50000), # Increased max_iter to allow for better convergence
                }

            all_penalties = ['l1', 'l2', 'elasticnet', None] # Unified penalties
            params['penalty'] = trial.suggest_categorical('penalty', all_penalties)
            # Only suggest C if penalty is not None
            if params['penalty'] is not None:
                params["C"] = trial.suggest_float('C', 1e-10, 1000, log=True)

            # Only suggest l1_ratio if penalty is 'elasticnet'

            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)


            # Prune invalid combinations:
            if (
                (params['solver'] == 'lbfgs' and params['penalty'] not in ['l2', None]) or
                (params['solver'] == 'liblinear' and params['penalty'] not in ['l1', 'l2']) or
                (params['solver'] == 'sag' and params['penalty'] not in ['l2', None]) or
                (params['solver'] == 'newton-cholesky' and params['penalty'] not in ['l2', None]) or
                (params['solver'] == 'saga' and params['penalty'] not in ['elasticnet', 'l1', 'l2', None])
                ):
                raise optuna.TrialPruned() # Invalid combination of solver and penalty


            return params

        # Fetch classifier-specific parameters
        for param, settings in model_params.items():
            if isinstance(settings, list):  # Categorical parameter
                    params[param] = trial.suggest_categorical(param, settings)
            elif isinstance(settings, dict):
                param_type = settings.get("type")
                min_val = settings.get("min")
                max_val = settings.get("max")

                # Generate parameter suggestions based on type
                if param_type == "int":
                    params[param] = trial.suggest_int(param, min_val, max_val)
                elif param_type == "float":
                    params[param] = trial.suggest_float(param, min_val, max_val, log=settings.get("log", False))
            else:

                params[param] = settings  # Fixed parameters

        return params


class ModelFactory:
    """
    A class to create model instances with additional parameters for specific classifiers.

    Attributes:
        model_name (str): The name of the model to be instantiated.
        model_hyperparams (dict): The best hyperparameters for the model.
    """

    def __init__(self, transformer_pipeline_config, model_config, param_constants):
        """
        Initialize ModelFactory with a YAML config.

        Args:
            config_path (str): Path to the configuration file in YAML format.
        """
        main_utils = MainUtils()
        self.model_config = model_config
        self.param_constants = param_constants
        self.transformer_pipeline_config = transformer_pipeline_config

        self.study_config = self.param_constants.get("study", {})
        self.model_config = self.model_config.get("models", {})
        self.cross_val_config = self.param_constants.get("cross_val", {})
        self.model_list_config = self.param_constants.get("model_list", {})
        self.base_model_config = self.param_constants.get("base_model", {})
        self.best_models = []


    @staticmethod
    def class_for_name(module_name, class_name):
        """
        Dynamically import and return a class reference.

        Args:
            module_name (str): Module name (e.g., 'sklearn.ensemble').
            class_name (str): Class name (e.g., 'RandomForestClassifier').

        Returns:
            type: The imported class.
        """
        try:
            module = import_module(module_name)
            return getattr(module, class_name)
        except Exception as e:
            raise ImportError(f"Error loading class {class_name} from module {module_name}: {e}")


    def _create_model(self, trial, model_config):
        """
        Dynamically create a model instance using trial and configuration.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            model_config (dict): Model configuration with parameter details.

        Returns:
            model: An instance of the specified model class.
        """
        params = {}

        if model_config["class"] == "LogisticRegression":
            # Basic hyperparameters
            params = {
                "solver": trial.suggest_categorical('solver', ['newton-cholesky',"newton-cg", 'lbfgs', 'liblinear', 'sag', 'saga']),
                "max_iter": trial.suggest_int('max_iter', 10000, 50000),
                }

            params['penalty'] = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])

            # Only suggest C if penalty is not None
            if params['penalty'] is not None:
                params["C"] = trial.suggest_float('C', 1e-10, 1000, log=True)

            # Only suggest l1_ratio if penalty is 'elasticnet'
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)

            # Prune invalid combinations:
            if (
                (params['solver'] == 'lbfgs' and params['penalty'] not in ['l2', None]) or
                (params['solver'] == 'liblinear' and params['penalty'] not in ['l1', 'l2']) or
                (params['solver'] == 'sag' and params['penalty'] not in ['l2', None]) or
                (params['solver'] == 'newton-cg' and params['penalty'] not in ['l2', None]) or
                (params['solver'] == 'newton-cholesky' and params['penalty'] not in ['l2', None]) or
                (params['solver'] == 'saga' and params['penalty'] not in ['elasticnet', 'l1', 'l2', None])
                ):
                raise optuna.TrialPruned() # Invalid combination of solver and penalty

        else:
            for param, details in model_config["parameters"].items():
                if isinstance(details, list):  # Categorical parameter
                    params[param] = trial.suggest_categorical(param, details)
                elif isinstance(details, dict):  # Ranges for int or float
                    param_type = details["type"]
                    if param_type == "int":
                        params[param] = trial.suggest_int(param, details["min"], details["max"])
                    elif param_type == "float":
                        params[param] = trial.suggest_float(param, details["min"], details["max"], log=details.get("log", False))
                    else:
                        raise ValueError(f"Unsupported parameter type: {param_type}")
                else:  # Fixed parameters
                    params[param] = details

        # Dynamically import and initialise the model
        model_class = self.class_for_name(model_config["module"], model_config["class"])
        return model_class(**params)

    def get_preprocessor_pipeline(self, trial, model: object):
        pipeline_manager = PipelineManager(trial=trial, **self.transformer_pipeline_config)

        pipeline_manager.add_step('create_new_features', pipeline_manager.build(step_name='create_new_features'), position=0)
        pipeline_manager.add_step('replace_class', pipeline_manager.build(step_name='replace_class'), position=1)
        pipeline_manager.add_step('drop_cols', pipeline_manager.build(step_name='drop_cols'), position=2)
        pipeline_manager.add_step('column_transformer', pipeline_manager.build(step_name='column_transformer'), position=3)

        # Add the resampler step based on the provided resample name or trial suggestion
        pipeline_manager.add_step('resampler', ResamplerSelector(trial=trial).build(), position=4)

        # Add the scaler step based on the provided resample name or trial suggestion
        pipeline_manager.add_step('scaler', ScalerSelector(trial=trial).build(), position=5)

        # Add the Dimensional Reduction step based on the provided parameter or trial suggestion
        #pipeline_manager.add_step('dim_reduction', DimensionalityReductionSelector(trial=trial).build(), position=6)

        # Add created model step based on the provided
        pipeline_manager.add_step('model', model, position=7)

        return pipeline_manager.get_pipeline()



    def optimize_model(self, model_name, model_config, X_train, y_train):
        """
        Optimize a single model using Optuna.

        Args:
            model_name (str): Name of the model.
            model_config (dict): Configuration of the model.
            X (array-like): Training features.
            y (array-like): Training targets.

        Returns:
            BestModel: Namedtuple containing the best model and its details.
        """

        scoring = self.cross_val_config["scoring"]
        cv = StratifiedKFold(n_splits=self.cross_val_config["cv"])
        if scoring == "f1":
            scoring = make_scorer(f1_score, average='weighted')


        scores_list = []
        def _objective(trial):
            model = self._create_model(trial, model_config)
            pipeline_model = self.get_preprocessor_pipeline(trial, model)

            scores = cross_val_score(
                pipeline_model, X_train, y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=self.cross_val_config["n_jobs"],
                error_score='raise',
                verbose=self.cross_val_config["verbose"]
            )

            # Save the scores for each model
            mean_score = scores.mean()
            scores_list.extend(scores)
            return mean_score


        study = optuna.create_study(direction=self.study_config["direction"])
        study.optimize(_objective,  n_trials=self.study_config["n_trials"])

        logging.info(f'completed cross-validation for the model_name: {model_name}')


        best_params = study.best_params
        best_model = self._create_model(optuna.trial.FixedTrial(best_params), model_config)
        pipeline_model = self.get_preprocessor_pipeline(study.best_trial, best_model)
        logging.info("patched model_pipeline with the best model and pipeline parameters")

        # Train the model with the best model and pipeline parameters
        trainer = CostModel(pipeline_model, X_train, y_train)
        trained_pipeline = trainer.train()
        y_pred, y_pred_proba = trainer.predict(X_train)
        evaluation_metrics = trainer.evaluate(y_train, y_pred, y_pred_proba)
        model_score = evaluation_metrics[scoring]
        trial_summary = {
            "trial_number": study.best_trial.number,
            "best_score": study.best_value,
            "start_time": study.best_trial.datetime_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        logging.info(f"Completed model training, evaluation scores ({scoring})for X_train: {model_score}")
        return BestModel(
            model_name=model_name,
            best_pipeline=trained_pipeline,
            best_model=best_model,
            best_params=best_params,
            best_score=study.best_value,
            trial_summary=trial_summary,
            scores_list=scores_list
        )


    def run(self, X, y):
        """
        Run optimisation for all models defined in the config.

        Args:
            X (array-like): Training features.
            y (array-like): Training targets.

        Returns:
            list[BestModel]: List of best models for each configuration.
        """
        scores_dict = {}
        for model_name, model_config in self.model_config.items():
            logging.info(f"Optimising {model_name}...")
            best_model_detail = self.optimize_model(model_name, model_config, X, y)
            self.best_models.append(best_model_detail)

            logging.info(f"Best {model_name} Params: {best_model_detail.best_params}")
            logging.info(f"Best {model_name} Score: {best_model_detail.best_score:.4f}")
            model_short_name = model_config['short_name']
            scores_dict[model_short_name] = best_model_detail.scores_list
        logging.info(f"Length of scores_dict {len(scores_dict)}")

        # Plotting boxplot for the training scores of each classifier
        sns.set(style="darkgrid")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=[scores for scores in scores_dict.values()],
                    orient="v",
                    palette="Set3")
        plt.xticks(ticks=range(len(scores_dict)), labels=scores_dict.keys())
        plt.title("Comparison of Training Scores for Each Classifier")
        plt.xlabel("Classifier")
        plt.ylabel("Optuna Hyperparameter Tunning Cross-validation F1 Score ")


        # Superimposing mean scores as scatter points with higher zorder
        mean_scores = [np.mean(scores) for scores in scores_dict.values()]
        for i, mean_score in enumerate(mean_scores):
            plt.scatter(i, mean_score, color='red', marker='o', s=100, label='Mean Score' if i == 0 else "", zorder=10)
        plt.xticks(rotation=45, ha="right")
        plt.legend()

        file_path =  "Boxplot_training_score.png"
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        logging.info("Boxplot for training score saved")

        return self.best_models