import math

import pandas as pd
from keras.models import load_model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import keras
import tensorflow as tf
import logging
import numpy as np
import json
from Dialogflow.get_dialogflow_intents import get_dialogflow_intents_as_json
from Metrics.metrics import get_confusion_matrix, true_positives, false_positives, false_negative, true_negative
from sklearn.model_selection import train_test_split

from Training.default_training import DefaultTraining
from util import Util


class TensorflowTrainingIntent(DefaultTraining):

    def __init__(self, data: pd.DataFrame, EMBEDDING_DIM=100, PADDING=15):
        super(TensorflowTrainingIntent, self).__init__()
        self.data = data
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.PADDING = PADDING

    def read_data(self, path):
        self.data = pd.read_json(path)

    def define_stopwords(self, stopwords):
        self.stopwords = stopwords

    def get_preprocessing_data(self):
        super(TensorflowTrainingIntent, self).get_preprocessing_data()
        return self._preprocessing_data

    def __preprocess__(self):
        if self.data is None:
            raise ValueError("data must not be None")
        phrases = []
        for i in range(self.data.shape[0]):
            for phrase in self.data.iloc[i]["phrases"]:
                phrase = Util.remove_punctuation(phrase)
                phrase = Util.remove_stopwords(phrase, self.stopwords)
                phrases.append((self.data.iloc[i].name, phrase))
        self._preprocessing_data = pd.DataFrame(phrases, columns=["classes", "phrases"])

    def __vectorize__(self):
        should_adapt = False
        if self.to_vector is None:
            self.to_vector = TextVectorization(output_mode="int", output_sequence_length=self.PADDING)
            should_adapt = True
        if self._preprocessing_data is None:
            raise ValueError("preprocessing_data must not be None. Call first __preprocess__")
        if should_adapt:
            self.to_vector.adapt(self._preprocessing_data["phrases"].to_list())
        tensor = self.to_vector(self._preprocessing_data["phrases"].to_numpy())
        self._preprocessing_data["tokenized"] = tensor.numpy().tolist()

    def __build_model__(self):
        super(TensorflowTrainingIntent, self).__build_model__()
        self.model = keras.Sequential([
            keras.layers.Embedding(len(self.to_vector.get_vocabulary()), self.EMBEDDING_DIM, input_length=self.PADDING),
            keras.layers.LSTM(self.data.shape[0]),
            keras.layers.Dense(self.data.shape[0], activation="softmax")
        ])

    def __compile_model__(self, epochs=100, batch_size=2):
        super(TensorflowTrainingIntent, self).__compile_model__(ephocs=epochs, batch_size=batch_size)
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def execute(self, epochs=100):
        self.__preprocess__()
        self.__vectorize__()
        if self.model is None:
            self.__build_model__()
        self.__compile_model__(epochs=epochs)
        x = np.array([np.array(phrase_vec) for phrase_vec in self._preprocessing_data["tokenized"]])
        y = pd.get_dummies(self._preprocessing_data["classes"]).to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.75)
        compute_steps_per_epoch = lambda size: int(math.ceil(1. * size / self.batch_size))
        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.epochs,
            steps_per_epoch=compute_steps_per_epoch(x_train.shape[0]),
            validation_steps=compute_steps_per_epoch(x_test.shape[0]),
            validation_data=(x_test, y_test),
            verbose=True,
        ).history

    def save_model(self, filename):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        self.model.save("../temp/nlp/tensorflow/" + filename + ".h5")
        vocabulary = self.to_vector.get_vocabulary()
        with open("../temp/nlp/tensorflow/vocabulary.json", "w", encoding='utf-8') as json_file:
            json_file.write(json.dumps(vocabulary, ensure_ascii=False))

    def load_model(self, model_path="../temp/nlp/tensorflow/intent_detector.h5",
                   vocabulary_path="../temp/nlp/tensorflow/vocabulary.json"):
        super(TensorflowTrainingIntent, self).load_model(model_path)
        if model_path:
            self.model = load_model(model_path)
            with open(vocabulary_path, "r", encoding="utf-8") as vocabulary:
                self.to_vector = TextVectorization(output_mode="int", vocabulary=json.load(vocabulary),
                                                   output_sequence_length=self.model.get_layer(
                                                       index=0).input_length)

    def predict(self, docs: np.ndarray, return_name=True):
        vectors = self.to_vector(docs)
        prediction = self.model.predict(vectors)
        data = self.data.name
        labels = [data.iloc[i] for i in range(self.data.shape[0])]
        prediction = np.argmax(prediction, axis=1)
        return [labels[intent_index] if return_name else intent_index for intent_index in prediction]


def get_training_model(path, epochs=100) -> TensorflowTrainingIntent:
    intents_json = get_dialogflow_intents_as_json(path)
    intents_json = pd.DataFrame(intents_json)
    training = TensorflowTrainingIntent(intents_json, PADDING=20)
    training.execute(epochs=epochs)
    return training


def execute():
    training = get_training_model("../temp/")
    docs = np.array([
        "Oi",
        "obrigado",
        "Qual o endereço ? ",
        "Que horas funcioname ?",
    ])
    prediction = training.predict(docs)
    cm = get_confusion_matrix(training)
    print(cm)
    tp = true_positives(cm, training)
    print(tp)
    fp = false_positives(cm, training)
    print(fp)
    fn = false_negative(cm, training)
    print(fn)
    tn = true_negative(cm, training)
    print(tn)
    print(tn + fn + tp + fp, training.get_preprocessing_data().shape)
    # training.save_model("intent_detector")


if __name__ == '__main__':
    execute()
