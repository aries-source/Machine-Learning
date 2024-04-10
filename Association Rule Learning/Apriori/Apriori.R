# Apriori Algorithm

# Importing Dataset
dataset = read.csv('Market_Basket_Optimisation.csv', header= FALSE)

library(arules)
#Creating a Sparse Matrix of transactions
sparseMatrix = read.transactions('Market_Basket_Optimisation.csv', 
                                 sep = ',', rm.duplicates = TRUE)
summary(sparseMatrix)

#Visualizing the Sparse Matrix
itemFrequencyPlot(sparseMatrix, topN = 50)

#Training the Apriori Algorithm on the Dataset
Rules = apriori(data = sparseMatrix, parameter = list(support = 0.003,
                                                      confidence = 0.4))
# Creating a Table to Visualize the Rules 
inspect(sort(Rules, by = 'lift')[1:10])
