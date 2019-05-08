from tpot import TPOTClassifier
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
tfid = TfidfVectorizer(strip_accents='ascii', max_df=0.5)
X = tfid.fit_transform(X)
X_test, y_test = dataset_loader.load_test()
X_test = tfid.transform(X_test)

pipeline_optimizer = TPOTClassifier(verbosity=2, use_dask=True, n_jobs=-1,generations=5, population_size=20, cv=5, config_dict='TPOT sparse')
pipeline_optimizer.fit(X, y)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')

