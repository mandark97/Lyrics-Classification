import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from tensorflow import keras

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
    vectorizer = TfidfVectorizer(
        strip_accents='ascii', max_df=max_df, max_features=50000)
    X_train = vectorizer.fit_transform(X).todense()

    model = keras.models.Sequential([
        keras.layers.Dense(64, input_shape=(50000,), activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(len(np.unique(y)), activation='softmax'),
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y)

    X_test, y_test = dataset_loader.load_test()
    X_test = vectorizer.transform(X_test)

    score = model.score(X_test, y_test)
    print(score)
    _run.log_scalar("pipeline_score", score)
