import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN

print(__doc__)

# Generate the dataset
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=100, random_state=10)

print(X)
print(X.shape)
print(y)
print(y.shape)

sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_sample(X, y)
print(y_resampled)
print(y_resampled.shape)

