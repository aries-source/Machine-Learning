# Hierarchical Clustering
#Agglomerative Method

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset

dataset = pd.read_csv('Mall_Customers.csv')
Features = dataset.iloc[:,1:].values

# Encoding Categorical Variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
obj = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
Features = np.array(obj.fit_transform(Features))

# Finding Optimal Clusters Using a Dendrogram

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(Features,method = 'ward'))
plt.title('Dendrogram of Clusters')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Training the Hierarchical Clustering Model on the Dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity= 'euclidean', linkage= 'ward')

# Fitting and Predicting the Clusters

Clusters = hc.fit_predict(Features)
print(Clusters)

# Visualizing the Clusters

plt.scatter(Features[Clusters==0,3],Features[Clusters==0,4], s = 30, c='red', label = 'Cluster 1')
plt.scatter(Features[Clusters==1,3],Features[Clusters==1,4], s = 30, c='blue', label = 'Cluster 2')
plt.scatter(Features[Clusters==2,3],Features[Clusters==2,4], s = 30, c='green', label = 'Cluster 3')
plt.scatter(Features[Clusters==3,3],Features[Clusters==3,4], s = 30, c='cyan', label = 'Cluster 4')
plt.scatter(Features[Clusters==4,3],Features[Clusters==4,4], s = 30, c='magenta', label = 'Cluster 5')
# plt.scatter(hc.cluster_centers_[:,3],hc.cluster_centers_[:,4], s= 50, c='yellow',label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (K $)')
plt.ylabel('Spending Score (%)')
plt.legend()
plt.show()