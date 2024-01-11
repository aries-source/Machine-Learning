#Reinforcement Learning 
#Upper Confidence Bound

#Importing Dataset
Ads = read.csv('Ads_CTR_Optimisation.csv')

#Implementing Random Selection (Generic)
N = 10000
d = 10
adsSelected = integer(0)
totalReward = 0
for (n in 1:N){
  ad = sample(1:10,1)
  adsSelected = append(adsSelected,ad)
  Reward = Ads[n,ad]
  totalReward = totalReward + Reward
}

hist(adsSelected,
     main = 'Click Through Rate',
     xlab = 'Ads',
     ylab = 'Frequency')


#Implementing The Upper Confidence Bound Algorithm
numberOfSelection = integer(d)
sumOfReward = integer(d)
selectedAds = integer()
TotalReward = 0
for (i in 1:N){
  maxUpperBound = 0
  ad = 0
  for (j in 1:d){
    if (numberOfSelection[j] > 0){
      averageReward = sumOfReward[j]/numberOfSelection[j]
      delta_i = sqrt(3/2 * log(i)/numberOfSelection[j])
      upperBound = averageReward + delta_i
    } else {
      upperBound = 1e400
    }
    if (upperBound > maxUpperBound){
      maxUpperBound = upperBound
      ad = j
    }
  }
  selectedAds = append(selectedAds,ad)
  numberOfSelection[ad] = numberOfSelection[ad] +1
  reward = Ads[i,ad]
  sumOfReward[ad] = sumOfReward[ad] + reward
  TotalReward = TotalReward + reward
}

# Visualizing The Click Through Reward
hist(selectedAds,
     col = 'blue',
     main = 'Click Through Rate',
     xlab = 'Ads',
     ylab = 'Frequency')
