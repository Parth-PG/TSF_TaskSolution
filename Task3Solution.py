# Task 3

### To Explore Unsupervised Machine Learning 

# From the 'Iris' dataset, predict the optinum number of clusters and represent it visually.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
%matplotlib inline

df = pd.read_csv("D:\Internships\The Sparks Foundation\Task 3\Iris.csv")

df.head() 

a = df.iloc[:, [0, 1, 2, 3]].values

from sklearn import datasets
from sklearn.cluster import KMeans

sos = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(a)
    sos.append(kmeans.inertia_)

from sklearn import datasets

plt.plot(range(1, 11), sos)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squares')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(a)

plt.scatter(a[y_kmeans == 0, 0], a[y_kmeans == 0, 1], s = 50, c = 'orange', label = 'Iris-setosa')
plt.scatter(a[y_kmeans == 1, 0], a[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Iris-versicolour')
plt.scatter(a[y_kmeans == 2, 0], a[y_kmeans == 2, 1], s = 50, c = 'black', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.legend()

