import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, LabelEncoder
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import save_object
from dataclasses import dataclass
import category_encoders as ce


@dataclass  # Used when we want to create class without init function
class DataTransformationConfig:
    # we are defining path for final preprocessor pickel file
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    product_id_preprocessor_obj_file_path = os.path.join('artifacts', 'product_id_preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        # to create path object
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            num_cols = ['tempo', 'beats', 'chroma_stft',
                        'spectral_centroid', 'spectral_bandwidth',
                        'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
                        'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
                        'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',
                        'mfcc20']
            target_col = ["label"]

            categories = ['pop', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'blues', 'reggae', 'rock']
            logging.info('Pipeline Initiated')

            # Categorical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                ]
            )

            target_pipeline = Pipeline(
                steps=[
                    ("encoder", OrdinalEncoder(categories=[categories]))
                ]
            )

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_cols),
            ])

            target_preprocessor = ColumnTransformer([
                ('target', target_pipeline, target_col),
            ])

            logging.info('Pipeline Completed')

            return preprocessor, target_preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)

    def initaite_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            # defining preprocessor object
            preprocessing_obj, target_preprocessor_obj = self.get_data_transformation_object()
            logging.info('Obtained preprocessing object')

            target_column_name = 'label'
            drop_columns = [target_column_name, 'filename', "rmse", "rolloff"]

            # defining X_train, y_train
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]
            # defining X_test, y_test
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Transforming input features using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(X=input_feature_train_df,
                                                                      y=target_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            target_feature_train_df = target_preprocessor_obj.fit_transform(pd.DataFrame(target_feature_train_df))
            print(target_feature_train_df)
            print(target_feature_test_df)

            logging.info("Transformed training and testing datasets.")

            # concatenate input and target features by converting target features into array
            # input feature is numpy array, only target feature is pandas dataframe so we are converting
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # preprocessor obj pickle file is saved in artifacts folder
            # It contains only column transformer details, it has no transformed array
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info("Exception occurred in the initiate_data-transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    dt_obj = DataTransformation()
    transform_data = dt_obj.initaite_data_transformation("artifacts/train.csv", "artifacts/test.csv")
    print(transform_data)
