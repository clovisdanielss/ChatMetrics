import pandas as pd
import random

import spacy

from Metrics.metrics import get_confusion_matrix, true_positives, false_positives, false_negative, true_negative
from util import Util
from Training.default_training import DefaultTraining
from spacy.util import minibatch
from spacy.training import Example
from Dialogflow.get_dialogflow_intents import get_dialogflow_entities_as_json
import numpy as np
import re


class SpacyTrainingNER(DefaultTraining):

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.entities = None

    def __preprocess__(self):
        super(SpacyTrainingNER, self).__preprocess__()
        entities = list(set(self.data["entities"]))
        self.entities = dict([(entities[i], i) for i in range(len(entities))])
        self.entities[''] = len(self.entities.keys())
        phrases = []
        j = -1
        for i in range(self.data.shape[0]):
            if i < j:
                continue
            j = i
            phrase = self.data["phrase"].iloc[j]
            phrase = Util.preprocess(phrase)
            entities = {'entities': []}
            phrase_ = phrase[:]
            end = 0
            while self.data.iloc[j].id == self.data.iloc[i].id:
                start = end + phrase_.index(Util.preprocess(self.data.iloc[j].word))
                end = start + len(self.data.iloc[j].word)
                phrase_ = phrase[end:]
                entity = (start, end, self.data.entities.iloc[j])
                entities['entities'].append(entity)
                j = j + 1
                if j >= self.data.shape[0]:
                    break
            phrases.append((phrase, entities))
        self._preprocessing_data = phrases

    def __build_model__(self, language="pt", model_name="ner"):
        super(SpacyTrainingNER, self).__build_model__()
        labels = self.entities
        if self.model is None:
            self.model = spacy.load("pt_core_news_lg")
        if not self.model.has_pipe(model_name):
            ner = self.model.add_pipe(model_name)
            for label in labels:
                ner.add_label(label)

    def __compile_model__(self, epochs=20, batch_size=1):
        super(SpacyTrainingNER, self).__compile_model__(epochs, batch_size)

    def __train__(self):
        super(SpacyTrainingNER, self).__train__()
        other_pipes = [pipe for pipe in self.model.pipe_names if pipe != 'ner']
        with self.model.disable_pipes(other_pipes):
            optimizer = self.model.create_optimizer()
            for epoch in range(self.epochs):
                random.shuffle(self._preprocessing_data)
                batches = minibatch(self._preprocessing_data, size=self.batch_size)
                for batch in batches:
                    examples = []
                    for text, ent in batch:
                        examples.append(Example.from_dict(self.model.make_doc(text), ent))
                    loss = self.model.update(examples, sgd=optimizer)
                    print(loss)

    def execute(self, epochs=100, batch_size=10):
        super(SpacyTrainingNER, self).execute()
        self.__preprocess__()
        self.__build_model__()
        self.__compile_model__(epochs=epochs, batch_size=batch_size)
        self.__train__()

    def save_model(self, path):
        super(SpacyTrainingNER, self).save_model(path)
        self.model.to_disk(path)

    def load_model(self, path="../nlp/spacy"):
        super(SpacyTrainingNER, self).load_model(path)
        if path:
            self.model = spacy.load(path)
            self.model.from_disk(path)

    def get_preprocessing_data(self):
        result = {"classes":[], "phrases":[]}
        for i in range(self.data.shape[0]):
            phrase = self.data.phrase.iloc[i]
            result["phrases"].append(phrase)
            words = [word for word in self.model(phrase)]
            result["classes"] += [self.entities[word.ent_type_] if word.ent_type_ in self.entities.keys() else self.entities[''] for word in words]
        result["phrases"] = pd.Series(result["phrases"])
        result["classes"] = np.array(result["classes"])
        return result

    @staticmethod
    def reload_data():
        data_ = pd.read_csv("../dataset/dataset.ptbr.twitter.train.ner", sep="\t")
        data = pd.read_csv("../dataset/twitter.train.csv")
        for i in range(data.shape[0]):
            index = data.iloc[i].old_id
            print("****************", index, "\n", data_.iloc[index], "\n")
            data.named_entity_type.iloc[i] = data_.named_entity_type.iloc[index]
            print(data.iloc[i])
        data.to_csv("../dataset/twitter.train.csv", index=False)

    def predict(self, docs, return_name=True):
        result = []
        for phrase in docs:
            prediction = self.model(str(phrase))
            entities_indexes = [self.entities[word.ent_type_] if word.ent_type_ in self.entities.keys() else self.entities[''] for word in prediction]
            entities_names = [word.ent_type_ if word.ent_type_ else '' for word in prediction]
            result += entities_names if return_name else entities_indexes
        return result


def get_training_model(entities_json=None, path="../temp/", epochs=100) -> SpacyTrainingNER:
    if entities_json is None:
        entities_json = get_dialogflow_entities_as_json(path)
    print(entities_json)
    entities_json = pd.DataFrame(entities_json)
    print(entities_json.head())
    training = SpacyTrainingNER(entities_json)
    training.execute(epochs=epochs)
    return training


def execute():
    training = get_training_model(epochs=10)
    docs = np.array(["Já que eu não consigo fazer o que eu queria. Vou perguntar o endereço dessa forma."])
    prediction = training.predict(docs)
    print(training.entities)
    print(training.get_preprocessing_data())
    print(prediction)
    classes_names = list(training.entities.keys())
    cm = get_confusion_matrix(training, classes_names)
    print(cm)
    tp = true_positives(cm, classes_names)
    print(tp)
    fp = false_positives(cm, classes_names)
    print(fp)
    fn = false_negative(cm, classes_names)
    print(fn)
    tn = true_negative(cm, classes_names)
    print(tn)
    print(tn + fn + tp + fp, len(training.get_preprocessing_data()["phrases"]))
    training.save_model('../temp/spacy')


if __name__ == '__main__':
    execute()
