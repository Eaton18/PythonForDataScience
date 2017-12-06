#-*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

# init parameters
# os.getcwd()
filename = 'Datasets/bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:,:8].as_matrix()   # regressors
y = data.iloc[:,8].as_matrix()    # labels

rlr = RLR()  # create a randomized LogisticRegression, feature selection
rlr.fit(x, y)  # training model
print(rlr.get_support())  # get feature selection result.
print(rlr.scores_)  # get each feature score

print(u'RandomizedLogisticRegression feature selection finished.')
print(u'The effective features are: %s' % ','.join(data.columns[rlr.get_support()]))

features = data.columns[:len(data.columns-1)]
x = data[data.columns[rlr.get_support()]].as_matrix() # selected features

lr = LR() # create logistic regression model
lr.fit(x, y) # using effective features training model
print(u'Accuracy：%s' % lr.score(x, y))
