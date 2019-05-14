import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from tensorflow import keras
import gensim
from gensim.models import Word2Vec

from preprocessing import DatasetLoader

ex = Experiment("tfid_neural_network")
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

    score = model.evaluate(X_test, y_test)
    print(score)
    _run.log_scalar("pipeline_score", score)
