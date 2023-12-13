<<<<<<< HEAD
dataset = read.csv('50_Startups.csv')

#Encoding the Categorical data

dataset$State = factor(dataset$State,
                       levels = c('New York','California','Florida'),
                       labels = c(1,2,3))

#Splitting Into Train and Test
library( caTools)
set.seed(123)
split = sample.split(dataset$Profit,SplitRatio = 0.8)
TrainSet = subset(dataset,split == TRUE)
TestSet = subset(dataset,split == FALSE)

#Model Fitting
regressor = lm(formula = Profit ~ ., data = TrainSet)
summary(regressor)

depPred = predict(regressor,newdata = TestSet)

#Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend+Administration+Marketing.Spend+State,
               data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend+Administration+Marketing.Spend,
               data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend+Marketing.Spend,
               data = dataset)
summary(regressor)

#Alternatively
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
backwardElimination(TrainSet, SL)

#Alternatively
regressor = lm(formula = Profit ~ ., data = TrainSet)
finalModel = step(regressor,direction = 'backward')
summary(finalModel)
=======
dataset = read.csv('50_Startups.csv')

#Encoding the Categorical data

dataset$State = factor(dataset$State,
                       levels = c('New York','California','Florida'),
                       labels = c(1,2,3))

#Splitting Into Train and Test
library( caTools)
set.seed(123)
split = sample.split(dataset$Profit,SplitRatio = 0.8)
TrainSet = subset(dataset,split == TRUE)
TestSet = subset(dataset,split == FALSE)

#Model Fitting
regressor = lm(formula = Profit ~ ., data = TrainSet)
summary(regressor)

depPred = predict(regressor,newdata = TestSet)

#Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend+Administration+Marketing.Spend+State,
               data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend+Administration+Marketing.Spend,
               data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend+Marketing.Spend,
               data = dataset)
summary(regressor)

#Alternatively
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
backwardElimination(TrainSet, SL)

#Alternatively
regressor = lm(formula = Profit ~ ., data = TrainSet)
finalModel = step(regressor,direction = 'backward')
summary(finalModel)
>>>>>>> 18e1876480b15ecc8ba58530f63fb9d0263e8f26
