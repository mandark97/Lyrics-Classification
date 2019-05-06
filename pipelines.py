from sklearn.base import BaseEstimator, TransformerMixin


class CustomLabelEncode(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

