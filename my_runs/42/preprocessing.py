import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DatasetLoader(object):
    TRAIN_CSV = "Lyrics-Genre-Train.csv"
    TEST_CSV = "Lyrics-Genre-Test-GroundTruth.csv"

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
