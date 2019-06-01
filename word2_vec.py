import gensim
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from utils import DatasetLoader, ResultStorage

ex = Experiment("word2vec_clf")
ex.observers.append(MongoObserver.create(
    url="mongodb://localhost:27017",
    db_name="lyrics_classification"
))


@ex.config
def model_config():
    clf_params = {}


def get_sentence_embedding(w2v_model, sentence):
    embedding = np.zeros(3000)

    for word in sentence.split():
        try:
            vector = w2v_model.wv.get_vector(word)
        except KeyError:
            vector = np.zeros(3000)
        embedding += vector

    return embedding / len(sentence.split())


def preprocess_x(X, w2v_model):
    return np.array([get_sentence_embedding(w2v_model, sentence)
                     for sentence in X])


@ex.automain
def main(clf_params):
    dataset_loader = DatasetLoader()
    w2v_model = Word2Vec.load("word2vec.model")

    X_train, y_train = dataset_loader.load_train()
    X_train = preprocess_x(X_train, w2v_model)
    X_test, y_test = dataset_loader.load_test()
    X_test = preprocess_x(X_test, w2v_model)

    grid = GridSearchCV(SVC(),
                        clf_params,
                        n_jobs=-1,
                        cv=5,
                        verbose=2,
                        return_train_score=True,
                        refit=True)
    grid.fit(X_train, y_train)

    result_storage = ResultStorage(ex, grid)
    result_storage.store_experiment_data(X_test, y_test)
