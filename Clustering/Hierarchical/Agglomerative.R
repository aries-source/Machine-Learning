#Hierarchical Clustering
#Agglomerative Method

#Importing the Dataset
Mall = read.csv('Mall_Customers.csv')

#Data Preprocessing
Mall = as.data.frame(Mall)
Mall = Mall[,2:5]

#Encoding Categorical Data
Mall$Gender = factor(Mall$Gender,
                        levels = c('Female','Male'),
                        labels = c(0,1))
#Dendrogram
Dendrogram = hclust(dist(Mall, method = 'euclidean'),
                    method = 'ward.D')
plot(Dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distances')

#Fitting the Hierarchical Clustering algorithms on the dataset
hc = hclust(dist(Mall, method = 'euclidean'),
                    method = 'ward.D')

Clusters = cutree(hc,3)

#Visualizing the Clusters
library(cluster)
clusplot(Mall[,3:4],
         Clusters,
         lines = 0,
         span = TRUE,
         plotchar = FALSE,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         main = paste('Clusters of Clients'),
         xlab = 'Annual Income (K$)',
         ylab = 'Spending Score (%)')
