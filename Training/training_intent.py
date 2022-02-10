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
import spacy

from Training.default_training import DefaultTraining
from util import Util


class TrainingIntent(DefaultTraining):

    def __init__(self, raw_data, data: pd.DataFrame, EMBEDDING_DIM=100, PADDING=15):
        super(TrainingIntent, self).__init__()
        self.data = data
        self.raw_data = raw_data
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.PADDING = PADDING

    def read_data(self, path):
        self.data = pd.read_json(path)

    def define_stopwords(self, stopwords):
        self.stopwords = stopwords

    def get_preprocessing_data(self):
        super(TrainingIntent, self).get_preprocessing_data()
        return self._preprocessing_data

    def __preprocess__(self):
        if self.data is None:
            raise ValueError("data must not be None")
        phrases = []
        classes_dict = {}
        index = 0
        for intent in self.raw_data:
            classes_dict[intent["name"]] = index
            index += 1
        self.classes_dict = classes_dict
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
        super(TrainingIntent, self).__build_model__()
        self.model = keras.Sequential([
            keras.layers.Embedding(len(self.to_vector.get_vocabulary()), self.EMBEDDING_DIM, input_length=self.PADDING),
            keras.layers.LSTM(self.data.shape[0]),
            keras.layers.Dense(self.data.shape[0], activation="softmax")
        ])

    def __compile_model__(self, epochs=100, batch_size=2):
        super(TrainingIntent, self).__compile_model__(ephocs=epochs, batch_size=batch_size)
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
        super(TrainingIntent, self).load_model(model_path)
        if model_path:
            self.model = load_model(model_path)
            with open(vocabulary_path, "r", encoding="utf-8") as vocabulary:
                self.to_vector = TextVectorization(output_mode="int", vocabulary=json.load(vocabulary),
                                                   output_sequence_length=self.model.get_layer(
                                                       index=0).input_length)

    def predict(self, docs: np.ndarray, return_name=True, use_entity_heuristic=None):
        vectors = self.to_vector(docs)
        prediction = self.model.predict(vectors)
        if use_entity_heuristic:
            nlp = None
            try:
                nlp = spacy.load(use_entity_heuristic)
                nlp.from_disk(use_entity_heuristic)
            except:
                print("Erro ao carregar modelo existente. Usando modelo padrão...")
                nlp = spacy.load("pt_core_news_sm")
            ef = entity_frequency(nlp, self.raw_data, self.classes_dict)
            prediction = heuristic(nlp, prediction, docs.tolist(), ef)
        data = self.data.name
        labels = [data.iloc[i] for i in range(self.data.shape[0])]
        prediction = np.argmax(prediction, axis=1)
        return [labels[intent_index] if return_name else intent_index for intent_index in prediction]


def heuristic(nlp, predictions, phrases, entity_frequency_by_intent, verbose=False):
    for i in range(len(phrases)):
        phrase = phrases[i]
        prediction = predictions[i, :]
        total_classes = predictions.shape[1]
        freq = entity_frequency_by_phrase(nlp, phrase)
        for ent_type in freq.keys():
            if verbose:
                print(f"Entidade: {ent_type}")
            for j in range(total_classes):
                efbi = entity_frequency_by_intent[j]
                if ent_type in efbi.keys():
                    if verbose:
                        print(f"val = {prediction[j]} * {entity_frequency_by_intent[j][ent_type]} * {freq[ent_type]}")
                    val = prediction[j] * entity_frequency_by_intent[j][ent_type] * freq[ent_type]
                    if verbose:
                        print(f"val = {val}")
                        print(f"prediction[j] = {prediction[j]}")
                    prediction[j] += val
                    underflow = 0
                    for k in range(total_classes):
                        if k != j:
                            if verbose:
                                print(f"prediction[{k}] = {prediction[k]}")
                            prediction[k] -= val / (total_classes - 1)
                            if (prediction[k] < 0):
                                underflow += (-1) * prediction[k]
                                prediction[k] = 0
                            if verbose:
                                print(f"prediction[{k}] = {prediction[k]}")
                    prediction[j] -= underflow
                    if verbose:
                        print(f"prediction[j] = {prediction[j]}")
                    predictions[i, :] = prediction
                else:
                    continue
    return predictions


def entity_frequency(nlp, intents, classes_dict):
    entity_frequency_by_intent = {}
    for intent in intents:
        intent_index = classes_dict[intent["name"]]
        entity_frequency_by_intent[intent_index] = {}
        total_phrases = len(intent["phrases"])
        for phrase in intent["phrases"]:
            frequency = entity_frequency_by_phrase(nlp, phrase, total_phrases=total_phrases)
            for ent_type in frequency.keys():
                if ent_type in entity_frequency_by_intent[intent_index].keys():
                    entity_frequency_by_intent[intent_index][ent_type] += frequency[ent_type]
                else:
                    entity_frequency_by_intent[intent_index][ent_type] = frequency[ent_type]
    return entity_frequency_by_intent


def entity_frequency_by_phrase(nlp, phrase, total_phrases=1):
    result = {}
    doc = nlp(phrase)
    for word in doc:
        ent_type = word.ent_type_
        if ent_type is '':
            continue
        if ent_type in result.keys():
            result[ent_type] += 1 / total_phrases
        else:
            result[ent_type] = 1 / total_phrases
    return result


def get_training_model(raw_intents_json=None, path="../temp/", epochs=100) -> TrainingIntent:
    if raw_intents_json is None:
        raw_intents_json = get_dialogflow_intents_as_json(path)
    intents_json = pd.DataFrame(raw_intents_json)
    training = TrainingIntent(raw_intents_json, intents_json, PADDING=15)
    training.execute(epochs=epochs)
    return training


def execute():
    training = get_training_model(path="../temp/")
    docs = np.array([
        "Oi",
        "obrigado",
        "Qual o endereço ? ",
        "Que horas funcioname ?",
    ])
    prediction = training.predict(docs)
    data = training.data.name
    classes_names = [data.iloc[i] for i in range(training.data.shape[0])]
    cm = get_confusion_matrix(training, classes_names, use_entity_heuristic=False)
    print(cm)
    tp = true_positives(cm, classes_names)
    print(tp, sum(tp))
    fp = false_positives(cm, classes_names)
    print(fp, sum(fp))
    fn = false_negative(cm, classes_names)
    print(fn, sum(fn))
    tn = true_negative(cm, classes_names)
    print(tn, sum(tn))
    print(tn + fn + tp + fp, training.get_preprocessing_data().shape)
    cm = get_confusion_matrix(training, classes_names, use_entity_heuristic="../temp/spacy")
    print(cm)
    tp = true_positives(cm, classes_names)
    print(tp, sum(tp))
    fp = false_positives(cm, classes_names)
    print(fp, sum(fp))
    fn = false_negative(cm, classes_names)
    print(fn, sum(fn))
    tn = true_negative(cm, classes_names)
    print(tn, sum(tn))
    print(tn + fn + tp + fp, training.get_preprocessing_data().shape)
    # training.save_model("intent_detector")


if __name__ == '__main__':
    execute()
