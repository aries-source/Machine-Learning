# Reinforcement Learning
# Upper Confidence Bound

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math

# Importing Dataset

Ads = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing the Upper Confidence Bound
# Setting the variables

numberOfRounds = 550
numberOfAds = 10
selectedAds = [] # To all the selected Ads the whole rounds
frequencyOfAds = [0]*numberOfAds # Number of times each Ad is seleected
sumOfRewards = [0]*numberOfAds # For each Ad 
totalReward = 0

# Actual implementation
for n in range(0,numberOfRounds):
    ad = 0
    maxUpperBound = 0
    for i in range(0,numberOfAds):
        if (frequencyOfAds[i] > 0):
            avegrageReward = sumOfRewards[i] / frequencyOfAds[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/frequencyOfAds[i])
            upperBound = avegrageReward + delta_i
        else:
            upperBound = 1e400

        if (upperBound > maxUpperBound):
            maxUpperBound = upperBound
            ad = i
    selectedAds.append(ad)
    frequencyOfAds[ad] += 1
    reward = Ads.values[n,ad]
    sumOfRewards[ad] += reward
    totalReward += reward

# Visualising the results
plt.hist(selectedAds)
plt.title('Histogram of Ads Selected')
plt.xlabel('Ads')
plt.ylabel('Selections')
plt.show()


