[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# WeightedLDA

This code provides a `scikit-learn` class that extends the `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` analysis to allow for weighting of each sample.  For general use of the analysis, please refer to the `scikit-learn` webpage:

[https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

`WeightedLDA` provides the same basic utilities of the `scikit-learn` class except for the following limitations:
- Only `solve="svd"` is supported.
- The priors cannot be defined a priori.
- The `feature_names_in_` attribute is not included.

The following code shows standard usage given sample data `X`, cluster identification `y`, and weights `wgt`:

```python
from WeightedLDA import WeightedLDA

lda = WeightedLDA()

lda.fit(X, y, wgts=wgts)
```

Calling `fit` without the `wgts` variable produces a fit the same as the `scikit-learn` class.

I tried to replicate the results numerically with the `scikit-learn` class, and was able to do so with all examples except when the sample data has a nullspace.  Even then, the resulting `predict` method produces consistent results and the `transform` method is numerically consistent when the data does not extend into the nullspace of the sampled data.

### TODO

Write the `fit` method using `PyTorch`.
