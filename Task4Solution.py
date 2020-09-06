# Importing libraries in Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the iris dataset
df=pd.read_csv('iris.csv')

print(df.head(5))

x = df.iloc[:, 1: -1].values
y = df.iloc[: , -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

from sklearn import tree
plt.figure(figsize = (10,8))
tree.plot_tree(classifier, filled = True , rounded = True, node_ids = True , proportion = True, feature_names = ['SepalLength(cm)', 'SepalWidth(cm)', 'PetalLength(cm)', 'PetalWidth(cm)'])
plt.show()