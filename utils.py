import json
import os

import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier import (ClassificationReport, ClassPredictionError,
                                    ConfusionMatrix)


class DatasetLoader(object):
    TRAIN_CSV = "datasets/Lyrics-Genre-Train.csv"
    TEST_CSV = "datasets/Lyrics-Genre-Test-GroundTruth.csv"

    def __init__(self):
        self._label_encoder = LabelEncoder()


    def load_train(self):
        X, y = self.__load_file(self.TRAIN_CSV)
        y = self._label_encoder.fit_transform(y)

        return X, y

    def load_test(self):
        X, y = self.__load_file(self.TEST_CSV)
        y = self._label_encoder.transform(y)

        return X, y

    def __load_file(self, file):
        df = pd.read_csv(file)
        X = df["Lyrics"]
        y = df["Genre"]
        return X, y


class ResultStorage(object):
    def __init__(self, experiment, model):
        self.ex = experiment
        self.model = model
        os.makedirs("metrics", exist_ok=True)

    def store_experiment_data(self, X_test, y_test):
        class_report = ClassificationReport(self.model)
        score = class_report.score(X_test, y_test)
        class_report.poof(
            'metrics/classification_report.png', clear_figure=True)
        self.ex.add_artifact('metrics/classification_report.png')

        confustion_matrix = ConfusionMatrix(self.model)
        confustion_matrix.score(X_test, y_test)
        confustion_matrix.poof(
            'metrics/confusion_matrix.png', clear_figure=True)
        self.ex.add_artifact('metrics/confusion_matrix.png')

        cpd = ClassPredictionError(self.model)
        cpd.score(X_test, y_test)
        cpd.poof('metrics/class_prediction_error.png', clear_figure=True)
        self.ex.add_artifact('metrics/class_prediction_error.png')

        print('score=', score)
        self.ex.log_scalar('score', score)
