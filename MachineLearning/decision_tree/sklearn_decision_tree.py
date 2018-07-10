#-*- coding: utf-8 -*-

import os
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

root_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
file_path = os.path.abspath(os.path.join(root_path, "python", "datasets", "datasets", "sales_data.csv"))

data = pd.read_csv(file_path, index_col = u'index')

data[data == u'good'] = 1
data[data == u'high'] = 1
data[data == u'yes'] = 1
data[data != 1] = -1
x = data.iloc[:,:3].as_matrix().astype(int)
y = data.iloc[:,3].as_matrix().astype(int)

dt_classifier = DecisionTreeClassifier(criterion='entropy')
dt_classifier.fit(x, y)
print('Accuracy All：' + str(dt_classifier.score(x, y)))

from sklearn.tree import export_graphviz
x = pd.DataFrame(x)
from sklearn.externals.six import StringIO
x = pd.DataFrame(x)
with open("tree.dot", 'w') as f:
    f = export_graphviz(dt_classifier, feature_names = x.columns, out_file = f)
