
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'class_label']

print(df)
#    color size  price class_label
# 0  green    M   10.1     class1
# 1    red    L   13.5     class2
# 2   blue   XL   15.3     class1

# color: disordered discrete features
# size: ordered discrete features
# price: discrete features
# class_label: label
print()



# Mapping order feature
print("> Mapping order feature")
size_mapping = {'XL':3, 'L':2, 'M':1}

df['size'] = df['size'].map(size_mapping)
print(df)
#    color  size  price class_label
# 0  green     1   10.1     class1
# 1    red     2   13.5     class2
# 2   blue     3   15.3     class1

# inv_size_mapping = {v:k for k, v in size_mapping.items()}
# df['size'] = df['size'].map(inv_size_mapping)
# print(df)
#    color size  price class_label
# 0  green    M   10.1     class1
# 1    red    L   13.5     class2
# 2   blue   XL   15.3     class1

print()

# Encode the class_label
print("> Code the class_label")
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['class_label']))}
print(class_mapping)
df['class_label'] = df['class_label'].map(class_mapping)
print(df)
#    color  size  price  class_label
# 0  green     1   10.1            0
# 1    red     2   13.5            1
# 2   blue     3   15.3            0

inv_class_mapping = {v:k for k, v in class_mapping.items()}
df['class_label'] = df['class_label'].map(inv_class_mapping)
print(df)
#    color  size  price class_label
# 0  green     1   10.1      class1
# 1    red     2   13.5      class2
# 2   blue     3   15.3      class1
print()

# Encode by LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['class_label'].values)
print(y)
# [0 1 0]
print(class_le.inverse_transform(y))
# ['class1' 'class2' 'class1']


# One-hot encoding for the discrete features
print("> Hot encode the discrete features")
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
print(X)
# [[1 1 10.1]
#  [2 2 13.5]
#  [0 3 15.3]]

ohe = OneHotEncoder(categorical_features=[0], sparse=False)
# print(ohe.fit_transform(X).toarray()) # convert sparse matrix into norm matrix, only useful when sparse=True

print(X)

print("ohe.fit_transform(X):")
print(ohe.fit_transform(X))

print("pd.get_dummies:")
print(pd.get_dummies(df[['color', 'size', 'price']]))

