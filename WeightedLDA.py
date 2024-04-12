import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class WeightedLDA(ClassifierMixin, TransformerMixin, BaseEstimator):
    def __init__(
        self, *, solver="svd", n_components=None, tol=0.0001, store_covariance=False
    ):
        self.solver = solver
        if solver != "svd":
            raise NotImplementedError("WeightedLDA only allows for 'svd' solver.")
        self.n_components = n_components
        self.tol = tol
        self.store_covariance = store_covariance
        return

    def fit(self, X, y, wgts=None, requires_y=True):
        """
        For some reason the within cluster scatter matrix is standardized in the sklearn
        implementation.  That is not done here.

        Note that the normalization in the sample variance for the within cluster scatter is
        (n_samples - n_classes) for the case that the data is unweighted.  When it is weighted,
        that equation does not hold any more. (1 - n_classes / n_samples) * wgt_sum is used.

        Some features not supported as compared to sklearn:
            self.feature_names_in_ is not defined
            self.priors_ cannot be set manually
        """
        n_samples = len(X)
        n_features = len(X[0])

        self.n_features_in_ = n_features
        self.classes_ = np.array(list(set(y)))
        n_classes = len(self.classes_)

        max_components = min(n_classes - 1, n_features)
        if not self.n_components:
            self.n_components = max_components

        if wgts is None:
            wgts = np.ones_like(y)
            wgt_sum = n_samples
            var_norm = (n_samples - n_classes) ** 0.5
        else:
            wgt_sum = np.sum(wgts)
            var_norm = ((1 - n_classes / n_samples) * wgt_sum) ** 0.5

        self.xbar_ = np.average(X, weights=wgts, axis=0)
        hw = np.zeros_like(X)
        hb = np.zeros((n_classes, n_features))
        self.means_ = np.zeros_like(hb)
        self.priors_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            self.priors_[i] = np.sum(wgts[y == c, None]) / wgt_sum
            self.means_[i] = np.average(X[y == c], weights=wgts[y == c], axis=0)
            hb[i] = self.priors_[i] ** 0.5 * (self.means_[i] - self.xbar_)
            hw[y == c] = np.sqrt(wgts[y == c, None]) * (X[y == c] - self.means_[i])

        u, s, v = np.linalg.svd(hw, full_matrices=False)
        rank = np.sum((s/wgt_sum**0.5 > self.tol).astype(np.int32))
        v = var_norm * v[:rank] / s[:rank, None]

        x = np.dot(v, hb.T)
        up, sp, vp = np.linalg.svd(x, full_matrices=False)
        rank = np.sum((sp > sp[0] * self.tol).astype(np.int32))
        self.scalings_ = np.dot(v.T, up[:, :rank])

        coef = np.dot((self.means_ - self.xbar_), v.T)

        self.coef_ = np.dot(coef, v)

        self.intercept_ = (
            -0.5 * np.sum(coef**2, axis=1)
            - np.dot(self.xbar_, self.coef_.T)
            + np.log(self.priors_)
        )

        self.explained_variance_ratio_ = sp**2
        self.explained_variance_ratio_ /= np.sum(self.explained_variance_ratio_)
        self.explained_variance_ratio_ = self.explained_variance_ratio_[
            : self.n_components
        ]

        if self.store_covariance:
            self.covariance_ = np.dot(hw.T, hw) / wgt_sum
        else:
            self.covariance_ = None

        # following code for two cluster case from sklearn - unnecessary but to be consistent
        if n_classes == 2:  # treat binary case as a special case
            coef_ = np.asarray(self.coef_[1, :] - self.coef_[0, :], dtype=X.dtype)
            self.coef_ = np.reshape(coef_, (1, -1))
            intercept_ = np.asarray(
                self.intercept_[1] - self.intercept_[0], dtype=X.dtype
            )
            self.intercept_ = np.reshape(intercept_, (1,))

        return self

    def predict(self, X):
        scores = np.dot(X, self.coef_.T) + self.intercept_
        if len(self.classes_) == 2:
            return np.take(self.classes_, scores.flatten() > 0)
        else:
            return np.take(self.classes_, np.argmax(scores, axis=1))

    def transform(self, X):
        return np.dot(X - self.xbar_, self.scalings_[:, : self.n_components])
