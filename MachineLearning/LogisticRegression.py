#-*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

# init parameters
# os.getcwd()
filename = '../Datasets/bankloan.csv'
data = pd.read_csv(filename)
x = data.iloc[:,:8].as_matrix()   # regressors
y = data.iloc[:,8].as_matrix()    # labels

features_columns = data.columns[:len(data.columns)-1]

rlr = RLR()  # create a randomized LogisticRegression, feature selection
rlr.fit(x, y)  # training model
print(features_columns)
print(rlr.get_support())  # get feature selection result.
print(rlr.scores_)  # get each feature score

print(u'RandomizedLogisticRegression feature selection finished.')
print(u'The effective features are:\n\t %s' % ', '.join(features_columns[rlr.get_support()]))

x = data[features_columns[rlr.get_support()]].as_matrix() # selected features

lr = LR() # create logistic regression model
lr.fit(x, y) # using effective features training model
print(u'Accuracy：%s' % lr.score(x, y))
