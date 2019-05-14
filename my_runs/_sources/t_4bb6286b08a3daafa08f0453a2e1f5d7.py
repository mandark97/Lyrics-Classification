import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from preprocessing import DatasetLoader
from sklearn.pipeline import Pipeline
from utils import ResultLogger
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("tfid_svc")
ex.observers.append(FileStorageObserver.create("my_runs"))

@ex.config
def tf_id_config():
    max_df = 0.1


@ex.config
def svc_config():
    kernel = 'linear'
    C = 0.5


@ex.automain
def main(max_df, kernel, C):
    dataset_loader = DatasetLoader()
    X, y = dataset_loader.load_train()

    pipe1 = Pipeline([
        ("tfid", TfidfVectorizer(strip_accents="ascii", max_df=max_df)),
        ("svc", SVC(verbose=True, kernel=kernel, C=C, max_iter=10))
    ])

    pipe1.fit(X, y)

    X_test, y_test = dataset_loader.load_test()
    print(pipe1.score(X_test, y_test))
