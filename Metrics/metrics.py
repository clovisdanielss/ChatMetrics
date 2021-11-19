from Training.default_training import DefaultTraining
import numpy as np
import tensorflow as tf
import pandas as pd


def get_confusion_matrix(training: DefaultTraining):
    phrases = training.get_preprocessing_data()["phrases"].to_numpy()
    true_class = training.get_preprocessing_data()["classes"]
    prediction = np.array(training.predict(phrases, return_name=False))
    data = training.data.name
    intent_names = [data.iloc[i] for i in range(training.data.shape[0])]
    cm = tf.math.confusion_matrix(labels=true_class, predictions=prediction).numpy()
    return pd.DataFrame(cm, columns=intent_names, index=intent_names)


def true_positives(confusion_matrix: pd.DataFrame, training: DefaultTraining):
    data = training.data.name
    intent_names = [data.iloc[i] for i in range(training.data.shape[0])]
    return [confusion_matrix[intent][intent] for intent in intent_names]


def false_positives(confusion_matrix: pd.DataFrame, training: DefaultTraining):
    tp = true_positives(confusion_matrix, training)
    tp = np.array(tp)
    sum_cols = confusion_matrix.sum(axis=0).to_numpy()
    fp = sum_cols - tp
    return fp


def false_negative(confusion_matrix: pd.DataFrame, training: DefaultTraining):
    tp = true_positives(confusion_matrix, training)
    tp = np.array(tp)
    sum_rows = confusion_matrix.sum(axis=1).to_numpy()
    fn = sum_rows - tp
    return fn


def true_negative(confusion_matrix: pd.DataFrame, training: DefaultTraining):
    tp = true_positives(confusion_matrix, training)
    fn = false_negative(confusion_matrix, training)
    fp = false_positives(confusion_matrix, training)
    sum_all = confusion_matrix.values.sum()
    tn = sum_all - (tp + fn + fp)
    return tn
