import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from utils import DatasetLoader, ResultStorage

ex = Experiment("tfid_clf")
ex.observers.append(MongoObserver.create(
    url="mongodb://localhost:27017",
    db_name="lyrics_classification"
))

ex.add_config("configs/preprocess_config.json")
@ex.config
def model_config():
    classifier_params = {
        "C": 0.5,
        "kernel": "linear"
    }


@ex.automain
def main(preprocess_params, classifier_params):
    dataset_loader = DatasetLoader()
    X_train, y_train = dataset_loader.load_train()
    X_test, y_test = dataset_loader.load_test()

    clf = Pipeline([
        ("preprocessing", TfidfVectorizer(**preprocess_params)),
        ("classifier", SVC(verbose=2, **classifier_params))
    ])

    clf.fit(X_train, y_train)
    result_storage = ResultStorage(ex, clf)
    result_storage.store_experiment_data(X_test, y_test)
