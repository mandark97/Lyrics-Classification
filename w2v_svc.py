import numpy as np
from gensim.models import Word2Vec
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.svm import SVC

from utils import DatasetLoader, ResultStorage

ex = Experiment("w2v_clf")
ex.observers.append(MongoObserver.create(
    url="mongodb://localhost:27017",
    db_name="lyrics_classification"
))

@ex.config
def model_config():
    classifier_params = {
        "C": 0.5,
        "kernel": "linear"
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
def main(classifier_params):
    dataset_loader = DatasetLoader()
    x_train, y_train = dataset_loader.load_train()
    w2v_model = Word2Vec.load("word2vec_models/word2vec.model")
    X_train = np.array([get_sentence_embedding(w2v_model, sentence)
                      for sentence in x_train])

    x_test, y_test = dataset_loader.load_test()
    X_test = np.array([get_sentence_embedding(w2v_model, sentence)
                      for sentence in x_test])

    clf = SVC(verbose=2, max_iter=10000 ,**classifier_params)

    clf.fit(X_train, y_train)
    result_storage = ResultStorage(ex, clf)
    result_storage.store_experiment_data(X_test, y_test)
