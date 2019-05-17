import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from preprocessing import DatasetLoader

ex = Experiment("tfid")
ex.observers.append(FileStorageObserver.create("my_runs"))


@ex.config
def count_vectorizer_config():
    count_vect = {
        "max_df": 0.6,
        "lowercase": True,
        "ngram_range": (2,2),
        "analyzer": "char"
    }


@ex.config
def svc_config():
    svc_conf = {
        "kernel": "linear",
        "C": 0.5
    }


@ex.automain
def main(count_vect, svc_conf, _run):
    dataset_loader = DatasetLoader()
    X, y = dataset_loader.load_train()

    pipe = Pipeline([
        ('bow', TfidfVectorizer(strip_accents='ascii',
                                **count_vect)),
        ('svc', SVC(verbose=True, **svc_conf))
    ])

    pipe.fit(X, y)

    X_test, y_test = dataset_loader.load_test()
    score = pipe.score(X_test, y_test)
    print(score)
    _run.log_scalar("pipeline_score", score)
