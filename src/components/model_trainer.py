# Basic Import
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import save_object
from src.utils.utils import optimal_cluster
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
import sys
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import rand_score, adjusted_rand_score


@dataclass  # Used when we want to create class without init function
class ModelTrainerConfig:
    # we are defining path for final model pickle file
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        # to create path object
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            # split the train test array
            X_train, y_train, X_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[
                                                                                                            :, -1]
            logging.info('Split Dependent and Independent variables from train and test data')

            number_of_cluster = optimal_cluster(X_train)
            print(number_of_cluster)

            # Create and fit GMM model
            gmm = GaussianMixture(n_components=number_of_cluster,
                                  random_state=42,
                                  covariance_type="full",
                                  init_params="random_from_data")
            gmm.fit(X_train)
            logging.info("Model Training completed")

            # Get cluster labels
            train_labels = gmm.predict(X_train)
            logging.info(f"Training Accuracy: {adjusted_rand_score(y_train, train_labels)}")
            test_labels = gmm.predict(X_test)
            logging.info(f"Testing Accuracy: {adjusted_rand_score(y_test, test_labels)}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=gmm
            )

            return gmm

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)
