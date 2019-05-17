import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from preprocessing import DatasetLoader
import gensim
from gensim.models import Word2Vec


ex = Experiment("word 2 vec")
ex.observers.append(FileStorageObserver.create("my_runs"))


@ex.config
def model_config():
    model_conf =  {
        "alpha": 0.5
    }


def get_sentence_embedding(w2v_model, sentence):
    embedding = np.zeros(3000)

    for word in sentence.split():
        try:
            vector = w2v_model.wv.get_vector(word)
        except KeyError:
            vector = np.zeros(3000)
        embedding += vector

    return embedding / len(sentence.split())


@ex.automain
def main(model_conf, _run):
    dataset_loader = DatasetLoader()
    X, y = dataset_loader.load_train()

    w2v_model = Word2Vec.load("word2vec.model")
    x_w2v = np.array([get_sentence_embedding(w2v_model, sentence)
                      for sentence in X])

    svc = MultinomialNB(**model_conf)
    svc.fit(x_w2v, y)

    X_test, y_test = dataset_loader.load_test()

    w_test = np.array([get_sentence_embedding(w2v_model, sentence)
                       for sentence in X_test])
    score = svc.score(w_test, y_test)

    print(score)
    _run.log_scalar("pipeline_score", score)
