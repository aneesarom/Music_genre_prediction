import sys
import os
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # set the preprocessor and model pickle file path
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            # get the preprocessor and model pickle file
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            # transform the data
            data_scaled = preprocessor.transform(features)
            # prediction
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)


# to get the new data for prediction
class CustomData:
    def __init__(self, tempo, beats, chroma_stft, rmse, spectral_centroid, spectral_bandwidth,
                 rolloff, zero_crossing_rate, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8,
                 mfcc9, mfcc10, mfcc11, mfcc12, mfcc13, mfcc14, mfcc15, mfcc16, mfcc17, mfcc18, mfcc19,
                 mfcc20):
        self.tempo = tempo
        self.beats = beats
        self.chroma_stft = chroma_stft
        self.rmse = rmse
        self.spectral_centroid = spectral_centroid
        self.spectral_bandwidth = spectral_bandwidth
        self.rolloff = rolloff
        self.zero_crossing_rate = zero_crossing_rate
        self.mfcc1 = mfcc1
        self.mfcc2 = mfcc2
        self.mfcc3 = mfcc3
        self.mfcc4 = mfcc4
        self.mfcc5 = mfcc5
        self.mfcc6 = mfcc6
        self.mfcc7 = mfcc7
        self.mfcc8 = mfcc8
        self.mfcc9 = mfcc9
        self.mfcc10 = mfcc10
        self.mfcc11 = mfcc11
        self.mfcc12 = mfcc12
        self.mfcc13 = mfcc13
        self.mfcc14 = mfcc14
        self.mfcc15 = mfcc15
        self.mfcc16 = mfcc16
        self.mfcc17 = mfcc17
        self.mfcc18 = mfcc18
        self.mfcc19 = mfcc19
        self.mfcc20 = mfcc20

    # to convert new data to dataframe
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'tempo': [self.tempo],
                'beats': [self.beats],
                'chroma_stft': [self.chroma_stft],
                'rmse': [self.rmse],
                'spectral_centroid': [self.spectral_centroid],
                'spectral_bandwidth': [self.spectral_bandwidth],
                'rolloff': [self.rolloff],
                'zero_crossing_rate': [self.zero_crossing_rate],
                'mfcc1': [self.mfcc1],
                'mfcc2': [self.mfcc2],
                'mfcc3': [self.mfcc3],
                'mfcc4': [self.mfcc4],
                'mfcc5': [self.mfcc5],
                'mfcc6': [self.mfcc6],
                'mfcc7': [self.mfcc7],
                'mfcc8': [self.mfcc8],
                'mfcc9': [self.mfcc9],
                'mfcc10': [self.mfcc10],
                'mfcc11': [self.mfcc11],
                'mfcc12': [self.mfcc12],
                'mfcc13': [self.mfcc13],
                'mfcc14': [self.mfcc14],
                'mfcc15': [self.mfcc15],
                'mfcc16': [self.mfcc16],
                'mfcc17': [self.mfcc17],
                'mfcc18': [self.mfcc18],
                'mfcc19': [self.mfcc19],
                'mfcc20': [self.mfcc20]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)
