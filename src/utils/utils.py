import os
import sys
import pickle
from src.exception.exception import CustomException
from src.logger.logging import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score


def save_object(file_path, obj):
    try:
        # If directory is not available it will create the folder
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        # To save the pickle file, "wb" string stands for "write binary" mode
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def optimal_cluster(X):
    wcss = []
    for k in range(2, 15):
        kmean = KMeans(n_clusters=k, init="k-means++", random_state=42)
        kmean.fit(X)
        wcss.append(kmean.inertia_)

    k = KneeLocator(range(2, 15), wcss, curve='convex', direction='decreasing')
    optimal_clusters = k.elbow
    return optimal_clusters


def load_object(file_path):
    try:
        # read the final model pickle file
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function utils')
        raise CustomException(e, sys)
