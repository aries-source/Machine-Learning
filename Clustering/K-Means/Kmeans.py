# K-Means Cluster

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')

Features = dataset.iloc[:,1:].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
obj = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
Features = np.array(obj.fit_transform(Features))

print(Features)

# KMeans
# Visualizing the Number of Clusters Needed
from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters=i, init = 'k-means++', random_state = 42)
    Kmeans.fit(Features)
    WCSS.append(Kmeans.inertia_)
plt.plot(range(1,11),WCSS)
plt.title('Within Clusters Sum of Squares')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Actual Model 
Kmeans = KMeans(n_clusters=5, init = 'k-means++', random_state = 42)
Clusters = Kmeans.fit_predict(Features)
print(Clusters)

# Visualizing the Clusters
plt.scatter(Features[Clusters==0,3],Features[Clusters==0,4], s = 30, c='red', label = 'Cluster 1')
plt.scatter(Features[Clusters==1,3],Features[Clusters==1,4], s = 30, c='blue', label = 'Cluster 2')
plt.scatter(Features[Clusters==2,3],Features[Clusters==2,4], s = 30, c='green', label = 'Cluster 3')
plt.scatter(Features[Clusters==3,3],Features[Clusters==3,4], s = 30, c='cyan', label = 'Cluster 4')
plt.scatter(Features[Clusters==4,3],Features[Clusters==4,4], s = 30, c='magenta', label = 'Cluster 5')
plt.scatter(Kmeans.cluster_centers_[:,3],Kmeans.cluster_centers_[:,4], s= 50, c='yellow',label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (K $)')
plt.ylabel('Spending Score (%)')
plt.legend()
plt.show()