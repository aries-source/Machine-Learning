<<<<<<< HEAD
# Natural Language Processing

# Import Libraries
require(tm)
require(SnowballC)
#Import Dataset
Dataset = read.delim('Restaurant_Reviews.tsv', 
                     quote = '',
                     stringsAsFactors = FALSE)
# Cleaning the Text
corpus = VCorpus(VectorSource(Dataset$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus,removeNumbers)
corpus = tm_map(corpus,removePunctuation)
corpus = tm_map(corpus,removeWords, stopwords())
corpus = tm_map(corpus,stemDocument)
corpus = tm_map(corpus,stripWhitespace)

# Creating the Bag of Words
sparseMatrix = DocumentTermMatrix(corpus)
sparseMatrix = removeSparseTerms(sparseMatrix,0.999)
Features = as.data.frame(as.matrix(sparseMatrix))

Data = Features
Data$Liked = Dataset$Liked

# Creating a Random Forest Classification Model

# Encoding
Data$Liked = factor(Data$Liked, levels = c(0,1))
#Splitting into test and train
library(caTools)
set.seed(123)
split = sample.split(Data$Liked,SplitRatio = 0.80)
trainSet = subset(Data,split == TRUE)
testSet = subset(Data,split == FALSE)

# Model Fitting
library(randomForest)
classifier = randomForest(x=trainSet[-692],y=trainSet$Liked,
                          ntree=10)

#Predicting the probabilities
classPred = predict(classifier, newdata = testSet[-692])


# Creating the confusion matrix
matrix = table(testSet[,692],classPred)
accuracy
=======
# Natural Language Processing

#Import Dataset
Dataset = read.delim('Restaurant_Reviews.tsv', 
                     quote = '',
                     stringsAsFactors = FALSE)
>>>>>>> 21cec8927b342ab0897e1485d6013fda26630e11
