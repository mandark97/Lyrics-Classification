import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from preprocessing import DatasetLoader

ex = Experiment("tfid_naive_bayes")
ex.observers.append(FileStorageObserver.create("my_runs"))


@ex.config
def tf_id_config():
    max_df = 0.4


@ex.config
def svc_config():
    kernel = 'linear'
    C = 0.5


@ex.automain
def main(max_df, kernel, C, _run):
    dataset_loader = DatasetLoader()
    X, y = dataset_loader.load_train()

    pipe1 = Pipeline([
        ("tfid", TfidfVectorizer(strip_accents="ascii", max_df=max_df)),
        ("naibe_bayes", MultinomialNB())
    ])

    pipe1.fit(X, y)

    X_test, y_test = dataset_loader.load_test()
    score = pipe1.score(X_test, y_test)
    print(score)
    _run.log_scalar("pipeline_score", score)
