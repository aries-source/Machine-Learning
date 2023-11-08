dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

library(rpart)
#Fitting the decision tree regression
regressor = rpart(formula = Salary~.,data = dataset,
                  control = rpart.control(minsplit = 1))

#Predicting Level 6.5
pred = predict(regressor,newdata = data.frame(Level = 6.5))

#Visualisation in high resolution
library(ggplot2)
xgrid = seq(min(dataset$Level),max(dataset$Level),0.01)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),color = 'red')+
  geom_line(aes(x=xgrid,y=predict(regressor,newdata = data.frame(Level = xgrid))),
            color ='blue')+
  ggtitle('Decision Tree Regression')+
  xlab('Position Level')+
  ylab('Salary')
