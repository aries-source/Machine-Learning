# Association Rule Learning
# Apriori

# Import packages
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as pyplot

# Importing Dataset
Dataset = pd.read_csv('Market_Basket_Optimisation.csv', header= None)

# Creating a List for the Apriori Algorithm
Transactions = []
for i in range(0,len(Dataset)):
    Transactions.append([str(Dataset.values[i,j]) for j in range(0,20)])

# Training the Apriori Model on the Dataset
from apyori import apriori
Rules = apriori(transactions= Transactions, min_support = 0.003, min_confidence=0.2,min_lift = 3, min_length =2,max_length = 2)

Results = list(Rules)

# Organizing the Results into a Pandas Dataframe
def inspect (Results):
    lhs = [tuple(result[2][0][0])[0] for result in Results]
    rhs = [tuple(result[2][0][1])[0] for result in Results]
    supports = [result[1] for result in Results]
    confidences = [result[2][0][2] for result in Results]
    lifts = [result[2][0][3] for result in Results]
    return list(zip(lhs,rhs,supports,confidences,lifts))

ResultsinDataframe = pd.DataFrame(inspect(Results), columns= ['Left Hand Side','Right Hand Side','Support','Confidence','Lift'])

# Sorting the Created Dataframe by Lift
finalResults = ResultsinDataframe.nlargest(n=10, keep= 'all',columns='Lift')
print(finalResults)
