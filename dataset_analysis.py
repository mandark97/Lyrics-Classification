import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.target import ClassBalance
from yellowbrick.text import FreqDistVisualizer, TSNEVisualizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

TRAIN_CSV = "datasets/Lyrics-Genre-Train.csv"
TEST_CSV = "datasets/Lyrics-Genre-Test-GroundTruth.csv"

df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

# cb = ClassBalance()
# cb.fit(df_train['Genre'])
# cb.poof("images/train_balance.png")

# cb = ClassBalance()
# cb.fit(df_test['Genre'])
# cb.poof("images/test_balance.png")


# def freq_dist_viz(vectorizer, data, file_name):
#     docs = vectorizer.fit_transform(data)
#     features = vectorizer.get_feature_names()
#     visualizer = FreqDistVisualizer(features=features)
#     visualizer.fit(docs)
#     visualizer.poof(file_name, clear_figure=True)


# vectorizer = CountVectorizer()
# freq_dist_viz(vectorizer, df_train['Lyrics'], "images/count_train.png")
# freq_dist_viz(vectorizer, df_test['Lyrics'], "images/count_test.png")


# vectorizer = CountVectorizer(stop_words="english")
# freq_dist_viz(vectorizer, df_train['Lyrics'],
#               "images/count_stopwords_train.png")
# freq_dist_viz(vectorizer, df_test['Lyrics'], "images/count_stopwords_test.png")


# vectorizer = TfidfVectorizer()
# freq_dist_viz(vectorizer, df_train['Lyrics'], "images/tfid_train.png")
# freq_dist_viz(vectorizer, df_test['Lyrics'], "images/tfid_test.png")


# vectorizer = TfidfVectorizer(stop_words="english")
# freq_dist_viz(vectorizer, df_train['Lyrics'],
#               "images/tfid_stopwords_train.png")
# freq_dist_viz(vectorizer, df_test['Lyrics'], "images/tfid_stopwords_test.png")

def get_sentence_embedding(w2v_model, sentence):
    embedding = np.zeros(3000)

    for word in sentence.split():
        try:
            vector = w2v_model.wv.get_vector(word)
        except KeyError:
            vector = np.zeros(3000)
        embedding += vector

    return embedding / len(sentence.split())

w2v_model = Word2Vec.load("word2vec_models/word2vec.model")
docs = np.array([get_sentence_embedding(w2v_model, sentence)
                    for sentence in df_train['Lyrics']])
# tfidf = TfidfVectorizer()
# docs = tfidf.fit_transform(X)
labels = df_train['Genre']

tsne = TSNEVisualizer()
tsne.fit(docs, labels)
tsne.poof("images/w2v_tsne.png")
