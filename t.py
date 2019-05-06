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
    ("tfid", TfidfVectorizer(strip_accents="ascii", max_df=0.4)),
    ("svc", SVC(verbose=True))
])

X_train, X_val, y_train, y_val = train_test_split(X, y)
pipe1.fit(X_train, X_train)

result_logger = ResultLogger(ResultLogger.PIPELINE, pipe1)

result_logger.score(X_val, y_val)
X_test, y_test = dataset_loader.load_test()
result_logger.score(X_val, y_val)
