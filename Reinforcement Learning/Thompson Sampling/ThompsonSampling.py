<<<<<<< HEAD
# Thompson Sampling Algorithm

# Import Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import random

# Import Dataset
Ads = pd.read_csv('Ads_CTR_Optimisation.csv')

numberOfRounds = 375
numberOfAds = 10
selectedAds = [] # To all the selected Ads the whole rounds
numbersOfReward1 = [0]*numberOfAds
numbersOfReward0 = [0]*numberOfAds
totalReward = 0

for n in range(0,numberOfRounds):
    ad = 0
    maxRandom = 0
    for i in range(0,numberOfAds):
        randomDraw = random.betavariate(numbersOfReward1[i] +1,numbersOfReward0[i] +1)
        if (randomDraw > maxRandom):
            maxRandom = randomDraw
            ad =i
    selectedAds.append(ad)
    Reward = Ads.values[n,ad]
    if (Reward == 1):
        numbersOfReward1[ad] += 1
    else:
        numbersOfReward0[ad] += 1
    
    totalReward += Reward


# Visualising the results
plt.hist(selectedAds)
plt.title('Histogram of Ads Selected')
plt.xlabel('Ads')
plt.ylabel('Selections')
plt.show()

=======
# Thompson Sampling Algorithm

# Import Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import random

# Import Dataset
Ads = pd.read_csv('Ads_CTR_Optimisation.csv')

numberOfRounds = 375
numberOfAds = 10
selectedAds = [] # To all the selected Ads the whole rounds
numbersOfReward1 = [0]*numberOfAds
numbersOfReward0 = [0]*numberOfAds
totalReward = 0

for n in range(0,numberOfRounds):
    ad = 0
    maxRandom = 0
    for i in range(0,numberOfAds):
        randomDraw = random.betavariate(numbersOfReward1[i] +1,numbersOfReward0[i] +1)
        if (randomDraw > maxRandom):
            maxRandom = randomDraw
            ad =i
    selectedAds.append(ad)
    Reward = Ads.values[n,ad]
    if (Reward == 1):
        numbersOfReward1[ad] += 1
    else:
        numbersOfReward0[ad] += 1
    
    totalReward += Reward


# Visualising the results
plt.hist(selectedAds)
plt.title('Histogram of Ads Selected')
plt.xlabel('Ads')
plt.ylabel('Selections')
plt.show()

>>>>>>> 21cec8927b342ab0897e1485d6013fda26630e11
