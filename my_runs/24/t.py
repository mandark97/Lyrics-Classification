import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from preprocessing import DatasetLoader
import gensim
from gensim.models import Word2Vec



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
def get_sentence_embedding(w2v_model,sentence):
  
  embedding = np.zeros(300)
  
  for word in sentence.split():
    try:
      vector = w2v_model.wv.get_vector(word)
    except KeyError as e:
      vector = np.zeros(300)
    embedding += vector
    
  return embedding / len(sentence.split())

@ex.automain
def main(count_vect, svc_conf, _run):
    dataset_loader = DatasetLoader()
    X, y = dataset_loader.load_train()
    w2v_model = Word2Vec(min_count=20,
                        window=2,
                        size=3000,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20)

    cleaned_text = [x.split() for x in X]
    w2v_model.build_vocab(cleaned_text)

    w2v_model.train(cleaned_text, total_examples=w2v_model.corpus_count, epochs=50)
    w2v_model.init_sims(replace=True)
    x_w2v = np.array([get_sentence_embedding(w2v_model,sentence) for sentence in X])
    
    svc = SVC(verbose=True, **svc_conf)
    svc.fit(x_w2v, y)

    X_test, y_test = dataset_loader.load_test()

    w_test = np.array([get_sentence_embedding(w2v_model,sentence) for sentence in X_test])
    score = svc.score(w_test, y_test)

    print(score)
    _run.log_scalar("pipeline_score", score)
