import numpy as np
import pandas as pd


class DefaultTraining:

    def __init__(self):
        self.stopwords = None
        self._preprocessing_data = None
        self.model = None
        self.to_vector = None
        self.data = None
        self.epochs = None
        self.batch_size = None
        self.history = None

    def read_data(self, path):
        self.data = pd.read_json(path)

    def define_stopwords(self, stopwords):
        self.stopwords = stopwords

    def __preprocess__(self):
        if self.data is None:
            raise ValueError("Intent must not be None")
        pass

    def get_preprocessing_data(self):
        pass

    def __build_model__(self):
        if self._preprocessing_data is None:
            raise ValueError("Must execute first __preprocess__")
        pass

    def __train__(self):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        pass

    def __compile_model__(self, ephocs, batch_size):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        self.epochs = ephocs
        self.batch_size = batch_size
        pass

    def execute(self):
        pass

    def save_model(self, path):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        pass

    def load_model(self, path):
        pass

    def predict(self, docs: np.ndarray, return_name=True):
        pass
