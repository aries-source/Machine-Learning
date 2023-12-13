<<<<<<< HEAD
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#Linear Model
linReg = lm(formula = Salary ~ .,data = dataset)
summary(linReg)

# Polynomial Model
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
polyReg = lm(formula = Salary ~ .,
             data = dataset )
summary(polyReg)


#Visualizing the Simple Linear Model
library(ggplot2)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             color = 'red')+
  geom_line(aes(x=dataset$Level, y=predict(linReg,
                                           newdata = dataset)),
            color = 'blue')+
  ggtitle('Visualizing the Simple Linear Model')+
  xlab('Position Level') +
  ylab('Salary')

#Visualizing the Polynomial Linear Model
library(ggplot2)
xgrid = seq(min(dataset$Level),max(dataset$Level),0.1)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             color = 'red')+
  geom_line(aes(x=dataset$Level, y=predict(polyReg,
                                           newdata = dataset)),
            color = 'blue')+
  ggtitle('Visualizing the Polynomial Linear Model')+
  xlab('Position Level') +
  ylab('Salary')

#Predicting the Salary of Position Level 6.5
linPred = predict(linReg, data.frame(Level=6.5))
PolyPred = predict(polyReg,data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 =6.5^3))
=======
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#Linear Model
linReg = lm(formula = Salary ~ .,data = dataset)
summary(linReg)

# Polynomial Model
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
polyReg = lm(formula = Salary ~ .,
             data = dataset )
summary(polyReg)

>>>>>>> 18e1876480b15ecc8ba58530f63fb9d0263e8f26
