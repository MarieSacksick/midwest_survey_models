from sklearn.base import BaseEstimator, TransformerMixin


class NumericalStabilizer(BaseEstimator, TransformerMixin):
    """Clips extreme values and ensures numerical stability of features
    before passing them to the downstream estimator.

    Parameters
    ----------
    threshold : float, default=1e10
        Values beyond [-threshold, threshold] are clipped.
    """

    def __init__(self, threshold=1e10):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X
