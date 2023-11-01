dataset = read.csv('Salary_Data.csv')

library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
trainSet = subset(dataset, split == TRUE)
testSet =  subset(dataset, split == FALSE)

trainSet = as.data.frame(trainSet)

regressor = lm(formula = Salary ~ YearsExperience, data= trainSet)
summary(regressor)

depPred = predict(regressor, newdata = testSet)

#Visualizing the train set
library(ggplot2)
ggplot()+
  geom_point(aes(x=trainSet$YearsExperience,y=trainSet$Salary),
             colour ='red') +
  geom_line(aes(x=trainSet$YearsExperience,y=predict(regressor, newdata = trainSet)),
            colour = 'blue') +
  ggtitle('Visualizing the train set')+
  xlab('Years of Experience')+
  ylab('Salary')

#Visualizing the test set
ggplot()+
  geom_point(aes(x=testSet$YearsExperience,y=testSet$Salary),
             colour ='red') +
  geom_line(aes(x=testSet$YearsExperience,y=depPred),
            colour = 'blue') +
  ggtitle('Visualizing the test set')+
  xlab('Years of Experience')+
  ylab('Salary')
