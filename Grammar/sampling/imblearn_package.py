
import numpy as np
from scipy import sparse

from sklearn.datasets import make_classification
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.under_sampling import ClusterCentroids

from imblearn.under_sampling import RandomUnderSampler

from imblearn.under_sampling import NearMiss

# 1. Compressed Sparse Rows(CSR)
print("> 1. Compressed Sparse Rows(CSR)")

data = np.array([1, 2, 3, 4, 5, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
indptr = np.array([0, 2, 3, 6])


mtx = sparse.csr_matrix((data,indices,indptr),shape=(3,3))
mtx.todense()

print(mtx.todense())
print()



# 2. Over sampling
print("> 2. Over Sampling")

print(">> 2.1 Random over sampling")
X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)
print(Counter(y))
print(X.shape)
print(y.shape)

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)

print(sorted(Counter(y_resampled).items()))

print(">> 2.2 SMOTE Sampling")
X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(X, y)
print(sorted(Counter(y_resampled_smote).items()))

print(">> 2.3 ADASYN Sampling")
X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(X, y)
print(sorted(Counter(y_resampled_adasyn).items()))

print(">> 2.4 SMOTE-kind Sampling")
# kind option 'borderline1', 'borderline2', 'svm'
X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))
print()



# 3. Under sampling
print("> 3. Over Sampling")

print(">> 3.1 Prototype Generation")
cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_sample(X, y)

print(sorted(Counter(y_resampled).items()))

print(">> 3.2 Prototype Selection")
print(">>> 3.2.1 Controlled under-sampling techniques")
print(">>>> 3.2.1.1 Controlled under-sampling techniques: Random Under Sampler")
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_sample(X, y)

print(sorted(Counter(y_resampled).items()))

print(">>>> 3.2.1.2 Controlled under-sampling techniques: Bootstrap")
np.vstack({tuple(row) for row in X_resampled}).shape

rus = RandomUnderSampler(random_state=0, replacement=True)
X_resampled, y_resampled = rus.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))

print(">>>> 3.2.1.3 Controlled under-sampling techniques: Bootstrap")
nm1 = NearMiss(random_state=0, version=1)
X_resampled_nm1, y_resampled = nm1.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))


