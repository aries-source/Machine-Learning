# Kernel Support Vector Machine

#Import dataset
dataset = read.csv('Social_Network_Ads.csv')

#Splitting into test and train
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.75)
trainSet = subset(dataset,split == TRUE)
testSet = subset(dataset,split == FALSE)

#Feature scaling
trainSet[,1:2] = scale(trainSet[,1:2])
testSet[,1:2] = scale(testSet[,1:2])

# Kernel SVM 
library(e1071)

classifier = svm(formula= Purchased ~., data = trainSet,
                 type = 'C-classification',
                 kernel = 'radial')

#Predicting the testSet
classPred = predict(classifier,type = 'response', newdata = testSet[-3])


# Creating the confusion matrix
matrix = table(testSet[,3],classPred)

# Visualising the Training set results
library(ElemStatLearn)
set = trainSet
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')

y_grid = predict(classifier,type = 'response', newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'dodgerblue', 'salmon'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'dodgerblue3', 'salmon3'))

# Visualising the Test set results
library(ElemStatLearn)
set = testSet
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier,type = 'response', newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'dodgerblue', 'salmon'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'dodgerblue3', 'salmon3'))
