#Support Vector Regression

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

library(e1071)
#SVR Model
regressor = svm(formula = Salary ~ Level, data = dataset,
                type='eps-regression')
summary(regressor)

#Predicting 6.5
pred = predict(regressor,newdata = data.frame(Level=6.5))

#Visualizing the Support Vector Regressor Model
library(ggplot2)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             color = 'red')+
  geom_line(aes(x=dataset$Level, y=predict(regressor,
                                           newdata = dataset)),
            color = 'blue')+
  ggtitle('Visualizing the Support Vector Regressor Model')+
  xlab('Position Level') +
  ylab('Salary')
