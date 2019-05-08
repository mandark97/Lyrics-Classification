import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from preprocessing import DatasetLoader
from sklearn.pipeline import Pipeline
from utils import ResultLogger

dataset_loader = DatasetLoader()
X, y = dataset_loader.load_train()

pipe1 = Pipeline([
    ("tfid", TfidfVectorizer(strip_accents="ascii", max_df=0.1)),
    ("svc", SVC(verbose=True, kernel='linear', C=0.5, max_iter=10))
])
pipe1.fit(X, y)

result_logger = ResultLogger(ResultLogger.PIPELINE, pipe1)
X_test, y_test = dataset_loader.load_test()
result_logger.score(X_test, y_test)
