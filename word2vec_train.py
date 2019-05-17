from preprocessing import DatasetLoader
import gensim
from gensim.models import Word2Vec

dataset_loader = DatasetLoader()
X, y = dataset_loader.load_train()
print("loaded data")

w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=3000,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20)

cleaned_text = [x.split() for x in X]
w2v_model.build_vocab(cleaned_text)

print("start training")
w2v_model.train(cleaned_text,
                total_examples=w2v_model.corpus_count,
                epochs=100)
w2v_model.save("word2vec.model")
