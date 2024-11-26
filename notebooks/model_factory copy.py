
import sys
import logging
import importlib
import yaml
import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from collections import namedtuple

logger = logging.getLogger(__name__)

__version__ = "0.0.6"

InitializedModelDetail = namedtuple("InitializedModelDetail", ["model_name", "model", "param_space"])
BestModel = namedtuple("BestModel", ["model_name", "best_model", "best_params", "best_score"])

class ModelFactory:
    def __init__(self, config_path: str):
        """Initialize ModelFactory with YAML config"""
        self.config = self.read_yaml_config(config_path)
        self.models_config = self.config["models"]
        self.study_config = self.config["study"]
        self.best_models = []

    @staticmethod
    def read_yaml_config(config_path: str) -> dict:
        """Read YAML configuration file"""
        try:
            with open(config_path, "r") as yaml_file:
                return yaml.safe_load(yaml_file)
        except Exception as e:
            raise Exception(f"Error reading config file: {e}")

    @staticmethod
    def class_for_name(module_name, class_name):
        """Dynamically import and return class reference"""
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as e:
            raise Exception(f"Error loading class {class_name} from module {module_name}: {e}")

    def initialize_model(self, model_name: str, model_config: dict):
        """Initialize a model and its hyperparameter search space"""
        try:
            model_class = self.class_for_name(model_config["module"], model_config["class"])
            model = model_class()
            param_space = model_config.get("param_space", {})
            return InitializedModelDetail(model_name=model_name, model=model, param_space=param_space)
        except Exception as e:
            raise Exception(f"Error initializing model {model_name}: {e}")

    def objective(self, trial, model_detail: InitializedModelDetail, X, y):
        """Objective function for Optuna"""
        try:
            params = {
                param: trial.suggest_categorical(param, values)
                for param, values in model_detail.param_space.items()
            }
            model = model_detail.model.set_params(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            return np.mean(scores)
        except Exception as e:
            raise Exception(f"Error during Optuna trial for {model_detail.model_name}: {e}")

    def optimize_model(self, model_detail: InitializedModelDetail, X, y):
        """Optimize a single model using Optuna"""
        try:
            study = optuna.create_study(direction=self.study_config["direction"])
            study.optimize(lambda trial: self.objective(trial, model_detail, X, y), 
                           n_trials=self.study_config["n_trials"])
            best_params = study.best_params
            best_score = study.best_value
            best_model = model_detail.model.set_params(**best_params)
            return BestModel(model_name=model_detail.model_name, best_model=best_model, 
                             best_params=best_params, best_score=best_score)
        except Exception as e:
            raise Exception(f"Error optimizing model {model_detail.model_name}: {e}")

    def run(self, X, y):
        """Run optimization for all models in the config"""
        try:
            for model_name, model_config in self.models_config.items():
                model_detail = self.initialize_model(model_name, model_config)
                best_model = self.optimize_model(model_detail, X, y)
                self.best_models.append(best_model)
            return self.best_models
        except Exception as e:
            raise Exception(f"Error running model factory: {e}")
