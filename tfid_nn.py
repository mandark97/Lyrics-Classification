import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras

from utils import DatasetLoader

ex = Experiment("tfid_neural_network")
ex.observers.append(FileStorageObserver.create("my_runs"))


@ex.config
def tf_id_config():
    max_df = 0.5


@ex.automain
def main(max_df):
    dataset_loader = DatasetLoader()
    X, y = dataset_loader.load_train()
    vectorizer = TfidfVectorizer(
        strip_accents='ascii', max_df=max_df, max_features=50000)
    X_train = vectorizer.fit_transform(X).todense()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y)

    model = keras.models.Sequential([
        keras.layers.Dense(128, input_shape=(50000,), activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(len(np.unique(y)), activation='softmax'),
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=15,
                       validation_data=(X_val, y_val))

    X_test, y_test = dataset_loader.load_test()
    X_test = vectorizer.transform(X_test)

    score = model.evaluate(X_test, y_test)
    print(score)

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
