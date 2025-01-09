# Artificial Neutral Networks

# Importing Dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]
attach(dataset)

#Encoding Categorical Variables in the Dataset
dataset$Geography = as.numeric(factor(dataset$Geography,
                              levels = c('France','Spain','Germany'),
                              labels = c(1,2,3)))

dataset$Gender = as.numeric(factor(dataset$Gender,
                           levels = c('Female','Male'),
                           labels = c(1,2)))
#Splitting into test and train sets
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited,SplitRatio = 0.8)
trainSet = subset(dataset,split == TRUE)
testSet = subset(dataset,split == FALSE)

#Feature scaling
trainSet[-11] = scale(trainSet[-11])
testSet[-11] = scale(testSet[-11])

# Building the ANN with h2o
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(trainSet),
                              activation = 'Rectifier',
                              hidden = c(6,6),
                              epochs = 100,
                              train_samples_per_iteration = -2)

#Predicting the Test set
probPred = h2o.predict(classifier, 
                       newdata = as.h2o(testSet[-11]))
yPred = as.vector(probPred > 0.5)

#Creatng a Confusion matrix
cMatrix = table(testSet[,11],yPred)

#Checking Accuracy metrics manually
TN = cMatrix[1,1]
TP = cMatrix[2,2]
FP = cMatrix[1,2]
FN = cMatrix[2,1]

# Accuracy
accuracy = (TP + TN) / sum(cMatrix)

# Recall
recall = TP / (TP + FN)


# Precision
precision = TP / (TP + FP)

# F1-Score
f1_score = 2 * (precision * recall) / (precision + recall)

#Using the Caret Package
library(caret)
actual = as.factor(as.vector(testSet[[11]]))
yPred  = as.factor(yPred)
cm = confusionMatrix(yPred, actual, positive = '1')

accuracy = cm$overall['Accuracy']
recall = cm$byClass['Sensitivity']
precision = cm$byClass['Precision']
f1_score = cm$byClass['F1']

# Print results
cat("Accuracy:", accuracy, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")

h2o.shutdown()

