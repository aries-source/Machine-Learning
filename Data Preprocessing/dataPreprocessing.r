<<<<<<< HEAD
#Import data
data = read.csv("Data.csv")

#Dealing with missing values
data$Age = ifelse(is.na(data$Age),
                  ave(data$Age,FUN = function(x) mean(x,na.rm = TRUE)),
                  data$Age)

data$Salary = ifelse(is.na(data$Salary),
                  ave(data$Salary,FUN = function(x) mean(x,na.rm = TRUE)),
                  data$Salary)

#Encoding Categorical Data
data$Country = factor(data$Country,
                      levels = c('France','Spain','Germany'),
                      labels = c(1,2,3))
data$Purchased = factor(data$Purchased,
                      levels = c('Yes','No'),
                      labels = c(1,0))

#Splitting into Test and Train
library(caTools)
set.seed(123)
split = sample.split(data$Purchased,SplitRatio = 0.8)
trainSet = subset(data, split==TRUE)
testSet = subset(data, split==FALSE)

#Feature Scaling
scaledTrain = scale(trainSet[,2:3])
scaledTest = scale(testSet[,2:3], 
                          center = attr(scaledTrain, "scaled:center"),
                          scale = attr(scaledTrain, "scaled:scale"))
trainSet[,2:3] = scaledTrain
testSet[,2:3] = scaledTest
=======
#Import data
data = read.csv("Data.csv")

#Dealing with missing values
data$Age = ifelse(is.na(data$Age),
                  ave(data$Age,FUN = function(x) mean(x,na.rm = TRUE)),
                  data$Age)

data$Salary = ifelse(is.na(data$Salary),
                  ave(data$Salary,FUN = function(x) mean(x,na.rm = TRUE)),
                  data$Salary)

#Encoding Categorical Data
data$Country = factor(data$Country,
                      levels = c('France','Spain','Germany'),
                      labels = c(1,2,3))
data$Purchased = factor(data$Purchased,
                      levels = c('Yes','No'),
                      labels = c(1,0))

#Splitting into Test and Train
library(caTools)
set.seed(123)
split = sample.split(data$Purchased,SplitRatio = 0.8)
trainSet = subset(data, split==TRUE)
testSet = subset(data, split==FALSE)

#Feature Scaling
scaledTrain = scale(trainSet[,2:3])
scaledTest = scale(testSet[,2:3], 
                          center = attr(scaledTrain, "scaled:center"),
                          scale = attr(scaledTrain, "scaled:scale"))
trainSet[,2:3] = scaledTrain
testSet[,2:3] = scaledTest
>>>>>>> 18e1876480b15ecc8ba58530f63fb9d0263e8f26
