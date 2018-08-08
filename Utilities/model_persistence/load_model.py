
from sklearn.externals import joblib

from os import path
root_dir = path.dirname(path.dirname(path.dirname(__file__)))
mode_path = path.join(root_dir, "model_save", "train_model.m")
# print(mode_path)

clf = joblib.load(mode_path)

X = [[0, 0], [1, 1]]

print(clf.predict(X))


