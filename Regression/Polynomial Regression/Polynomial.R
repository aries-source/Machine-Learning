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

