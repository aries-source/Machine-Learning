#K-Means Clustering

#Importing the Dataset
Mall = read.csv('Mall_Customers.csv')
#Data Preprocessing
Mall = as.data.frame(Mall)
Mall = Mall[,2:5]

#Encoding Categorical Data
Mall$Gender = factor(Mall$Gender,
                        levels = c('Female','Male'),
                        labels = c(0,1))

#Finding the number of clusters to be used
set.seed(6)
WCSS = vector()
for (i in 1:10) {
  WCSS[i] = sum(kmeans(Mall,i)$withinss)
}

#Visualizing the WCSS
plot(1:10,WCSS, type = 'b',
     main = paste('Clusters of Clients'),
     xlab = 'Number of Clusters',
     ylab = 'WCSS')

#Kmeans

KMeans = kmeans(Mall,5,iter.max = 300,nstart = 10)

library(cluster)
clusplot(Mall[,3:4],
         KMeans$cluster,
         lines = 0,
         span = TRUE,
         plotchar = FALSE,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         main = paste('Clusters of Clients'),
         xlab = 'Annual Income (K$)',
         ylab = 'Spending Score (%)')
