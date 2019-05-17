import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from preprocessing import DatasetLoader

ex = Experiment("bow")
ex.observers.append(FileStorageObserver.create("my_runs"))


@ex.config
def count_vectorizer_config():
    max_features = 50000
    lowercase = True


@ex.config
def svc_config():
    kernel = 'linear'
    C = 0.5


@ex.automain
def main(max_features, lowercase, kernel, C, _run):
    dataset_loader = DatasetLoader()
    X, y = dataset_loader.load_train()

    pipe = Pipeline([
        ('bow', CountVectorizer(strip_accents='ascii',
                                lowercase=lowercase,
                                max_features=max_features)),
        ('svc', SVC(C=C, kernel=kernel, verbose=True))
    ])

    pipe.fit(X, y)

    X_test, y_test = dataset_loader.load_test()
    score = pipe.score(X_test, y_test)
    print(score)
    _run.log_scalar("pipeline_score", score)
