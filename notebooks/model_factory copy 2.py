import optuna
import yaml
from sklearn.model_selection import cross_val_score
from importlib import import_module
from collections import namedtuple
from datetime import datetime

# Namedtuples for structured return values
InitializedModelDetail = namedtuple("InitializedModelDetail", ["model_name", "model", "param_space"])
BestModel = namedtuple("BestModel", ["model_name", "best_model", "best_params", "best_score", "trial_summary"])


class ModelFactory:
    def __init__(self, config_path):
        """
        Initialize ModelFactory with a YAML config.

        Args:
            config_path (str): Path to the configuration file in YAML format.
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.study_config = self.config.get("study", {})
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

    def initialize_model(self, model_name, model_config):
        """
        Initialize a model and its hyperparameter search space.

        Args:
            model_name (str): Name of the model.
            model_config (dict): Model configuration containing module, class, and parameter space.

        Returns:
            InitializedModelDetail: Namedtuple with model details.
        """
        try:
            model_class = self.class_for_name(model_config["module"], model_config["class"])
            model = model_class()
            param_space = model_config.get("parameters", {})
            return InitializedModelDetail(model_name=model_name, model=model, param_space=param_space)
        except Exception as e:
            raise ValueError(f"Error initializing model {model_name}: {e}")

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

    def _objective(self, trial, model_config, cross_val_params, X, y):
        """
        Objective function for Optuna trials.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            model_config (dict): Model configuration.
            cross_val_params (dict): Cross-validation settings.
            X (array-like): Training features.
            y (array-like): Training targets.

        Returns:
            float: The mean cross-validation score.
        """
        model = self._create_model(trial, model_config)
        score = cross_val_score(
            model, X, y,
            cv=cross_val_params["CV"],
            scoring=cross_val_params["scoring"]
        ).mean()
        return score

    def optimize_model(self, model_name, model_config, X, y):
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
        cross_val_params = self.config["cross_validation"]

        study = optuna.create_study(direction=self.study_config["direction"])
        study.optimize(
            lambda trial: self._objective(trial, model_config, cross_val_params, X, y),
            n_trials=self.study_config["n_trials"]
        )

        best_params = study.best_params
        best_model = self._create_model(optuna.trial.FixedTrial(best_params), model_config)
        trial_summary = {
            "trial_number": study.best_trial.number,
            "best_score": study.best_value,
            "start_time": study.best_trial.datetime_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return BestModel(
            model_name=model_name,
            best_model=best_model,
            best_params=best_params,
            best_score=study.best_value,
            trial_summary=trial_summary
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
        for model_name, model_config in self.config["models"].items():
            print(f"Optimising {model_name}...")
            best_model_detail = self.optimize_model(model_name, model_config, X, y)
            self.best_models.append(best_model_detail)

            print(f"Best {model_name} Params: {best_model_detail.best_params}")
            print(f"Best {model_name} Score: {best_model_detail.best_score:.4f}")
        return self.best_models
