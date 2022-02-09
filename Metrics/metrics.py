from Training.default_training import DefaultTraining
import numpy as np
import tensorflow as tf
import pandas as pd


def get_confusion_matrix(training: DefaultTraining, classes_names, use_entity_heuristic = False):
    phrases = training.get_preprocessing_data()["phrases"].to_numpy()
    true_class = training.get_preprocessing_data()["classes"]
    prediction = np.array(training.predict(phrases, return_name=False, use_entity_heuristic=use_entity_heuristic))
    cm = tf.math.confusion_matrix(labels=true_class, predictions=prediction).numpy()
    return pd.DataFrame(cm, columns=classes_names, index=classes_names)


def true_positives(confusion_matrix: pd.DataFrame, classes_names):
    return [confusion_matrix[_][_] for _ in classes_names]


def false_positives(confusion_matrix: pd.DataFrame, classes_names):
    tp = true_positives(confusion_matrix, classes_names)
    tp = np.array(tp)
    sum_cols = confusion_matrix.sum(axis=0).to_numpy()
    fp = sum_cols - tp
    return fp


def false_negative(confusion_matrix: pd.DataFrame, classes_names):
    tp = true_positives(confusion_matrix, classes_names)
    tp = np.array(tp)
    sum_rows = confusion_matrix.sum(axis=1).to_numpy()
    fn = sum_rows - tp
    return fn


def true_negative(confusion_matrix: pd.DataFrame, classes_names):
    tp = true_positives(confusion_matrix, classes_names)
    fn = false_negative(confusion_matrix, classes_names)
    fp = false_positives(confusion_matrix, classes_names)
    sum_all = confusion_matrix.values.sum()
    tn = sum_all - (tp + fn + fp)
    return tn
