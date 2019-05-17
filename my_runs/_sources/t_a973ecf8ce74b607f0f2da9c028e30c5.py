import gensim
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from preprocessing import DatasetLoader

ex = Experiment("word 2 vec")
ex.observers.append(FileStorageObserver.create("my_runs"))


@ex.config
def model_config():
    model_conf = {
        "n_jobs": -1,
        "verbose": 2,
        "n_estimators": 100,
        "max_features": 100
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

    X_train, X_val, y_train, y_val = train_test_split(x_w2v, y)
    model = keras.Sequential([
        Dense(8, activation='relu', input_shape=(3000,)),
        Dense(8, activation='relu'),
        Dense(8, activation='relu'),
        Dropout(0.5),
        Dense(6, activation='relu'),
        Dense(6, activation='relu'),
        Dense(6, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=500, batch_size=1024,
                        validation_data=(X_val, y_val))

    X_test, y_test = dataset_loader.load_test()
    w_test = np.array([get_sentence_embedding(w2v_model, sentence)
                       for sentence in X_test])

    score = model.evaluate(w_test, y_test)

    print(score)
    _run.log_scalar("pipeline_score", score)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
