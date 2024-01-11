#Reinforcement Learning 
#Thompson Sampling

#Importing Dataset
Ads = read.csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling
N = 10000
d = 10
adSelected = integer(0)
numberOfReward1 = integer(d)
numberOfReward0 = integer(d)
totalReward = 0

for (n in 1:N){
  ad = 0
  maxRandom = 0
  for (i in 1:d){
    randomDraw = rbeta(n=1,shape1 = numberOfReward1[i]+1,
                       shape2 = numberOfReward0[i]+1)
    if (randomDraw > maxRandom){
      maxRandom = randomDraw
      ad = i
    }
    
  }
  adSelected = append(adSelected,ad)
  Reward = Ads[n,ad]
  if(Reward == 1){
    numberOfReward1[ad] = numberOfReward1[ad] +1
  }else{
    numberOfReward0[ad] = numberOfReward0[ad] +1
  }
  totalReward = totalReward + Reward
}

# Visualizing The Click Through Reward
hist(adSelected,
     col = 'blue',
     main = 'Click Through Rate',
     xlab = 'Ads',
     ylab = 'Frequency')