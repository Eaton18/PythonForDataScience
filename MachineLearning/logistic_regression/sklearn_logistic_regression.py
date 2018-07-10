
#-*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


root_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
file_path = os.path.abspath(os.path.join(root_path, "python", "datasets", "datasets", "SMSSpamCollection.csv"))

df = pd.read_csv(file_path, delimiter='\t', header=None)
print(df.head())
print('Spam messages counts:', df[df[0] == 'spam'][0].count())
print('Ham messages counts:', df[df[0] == 'ham'][0].count())

lb = preprocessing.LabelBinarizer()
df[0] = np.array([number[0] for number in lb.fit_transform(df[0])])
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0], train_size=0.75)

# y_train = np.array([number[0] for number in lb.fit_transform(y_train)])
# y_test = np.array([number[0] for number in lb.fit_transform(y_test)])


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

print(type(X_train), "\tShape:", X_train.shape)
print(type(X_test), "\tShape:", X_test.shape)

# for i in range(10):
#     print('X_train:\t %s.' % (X_train.iloc[i]))
#     print('X_test:\t %s.' % (X_test.iloc[i]))

# classifier = LogisticRegression()
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

for i, prediction in enumerate(predictions[-5:]):
    print('Type: %s. \tMessages: %s' % (prediction, X_test_raw.iloc[i]))


# Model Evaluation
# confusion_matrix
y_test_cm = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred_cm = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
confusion_matrix = confusion_matrix(y_test_cm, y_pred_cm)
# print(confusion_matrix)
# plt.matshow(confusion_matrix)
# plt.title('Confusing Matrix')
# plt.colorbar()
# plt.ylabel('Actual Type')
# plt.xlabel('Predict Type')
# plt.show()

# Accuracy
y_pred, y_true = [0, 1, 1, 0], [1, 1, 1, 1]
print(accuracy_score(y_true, y_pred))

scores = cross_val_score(classifier, X_train, y_train, cv=5)
print('Accuracy:\t',np.mean(scores), scores)

# Precisions
precisions = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision')
print('Precisions:', np.mean(precisions), precisions)

# Recalls
recalls = cross_val_score(classifier, X_train, y_train, cv=5, scoring='recall')
print('Recalls:', np.mean(recalls), recalls)

# F1-Score
recalls = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
print('F1-Score:', np.mean(recalls), recalls)

predictions = classifier.predict_proba(X_test)
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:, 1])
roc_auc = auc(false_positive_rate, recall)
print('AUC:', np.mean(recalls), roc_auc)

# Plot ROC_Curve
if False:
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.show()


print(">> Grid Search:")
pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])
parameters = {
    'vect__max_df': (0.25, 0.5, 0.75),
    'vect__stop_words': ('english', None),
    'vect__max_features': (2500, 5000, 10000, None),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'vect__norm': ('l1', 'l2'),
    'clf__penalty': ('l1', 'l2'),
    'clf__C': (0.01, 0.1, 1, 10),
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=3)

grid_search.fit(X_train_raw, y_train)
print('Optimum: %0.3f' % grid_search.best_score_)
print('Optimized params combines: ')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions))
print('Precision: ', precision_score(y_test, predictions))
print('Recall: ', recall_score(y_test, predictions))




root_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
print(root_path)
file_path = os.path.abspath(os.path.join(root_path, "python", "datasets", "datasets", "bankloan.xls"))
print(file_path)

data = pd.read_excel(file_path)

x = data.iloc[:,:8].as_matrix()
y = data.iloc[:,8].as_matrix()

features_columns = data.columns[:len(data.columns) - 1]

rlr = RandomizedLogisticRegression()  # random logistics regression model
rlr.fit(x, y)  # training
rlr.get_support()  # get feature selection results

print(features_columns)
print(rlr.get_support())  # get feature selection result.
print(rlr.scores_)  # get each feature score

print(u'RandomizedLogisticRegression feature selection finished.')
print(u'The effective features are:\n\t %s' % ', '.join(features_columns[rlr.get_support()]))

x = data[features_columns[rlr.get_support()]].as_matrix() # selected features

lr = LogisticRegression() # create logistic regression model
lr.fit(x, y) # using effective features training model

print(u'Accuracy：%s' % lr.score(x, y))
