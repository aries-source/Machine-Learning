# Eclat Algorithm

# Importing Dataset
dataset = read.csv('Market_Basket_Optimisation.csv', header= FALSE)

library(arules)
#Creating a Sparse Matrix of transactions
sparseMatrix = read.transactions('Market_Basket_Optimisation.csv', 
                                 sep = ',', rm.duplicates = TRUE)
summary(sparseMatrix)

#Visualizing the Sparse Matrix
itemFrequencyPlot(sparseMatrix, topN = 50)

#Training the Eclat Algorithm on the Dataset
Rules = eclat(data = sparseMatrix, parameter = list(support = 0.004,
                                                      minlen = 2))
# Creating a Table to Visualize the Rules 
inspect(sort(Rules, by = 'support')[1:10])
