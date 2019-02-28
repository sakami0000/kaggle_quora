import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


class NBFeaturer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + self.alpha) / ((y == y_i).sum() + self.alpha)

    def fit(self, x, y):
        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) / self.pr(x, 0, y)))
        return self

    def preprocess_x(self, x, r):
        return x.multiply(r)

    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb


def tokenize(s):
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split()
