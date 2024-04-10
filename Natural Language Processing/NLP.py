# Natural Language Processing
# Sentiment Analyis

# Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Import Dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter= '\t', quoting=3)

# Cleaning the Text
import re 
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    Stopwords = stopwords.words('english')
    Stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(Stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Tokenisation 
# Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
Feature = cv.fit_transform(corpus).toarray()
Label = dataset.iloc[:,-1].values

# To get the necessary input for the max_feature arg, 
# we have leave it and fit cv first and then run len(Feature[0])).
# And based on that output we can make the decision to choose how frequent
#  words included in the feature should be.

# Classification
# Splitting into Test and Train
from sklearn.model_selection import train_test_split
featureTrain,featureTest,labelTrain,labelTest = train_test_split(Feature,Label,test_size=0.2,random_state=0)

# Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(featureTrain,labelTrain)

Pred = classifier.predict(featureTest)
print(np.concatenate((Pred.reshape(len(Pred),1),labelTest.reshape(len(labelTest),1)),1))

# ROC
from sklearn.metrics import roc_curve
FPR , TPR , threshold = roc_curve(labelTest, Pred)
plt.plot([0,1],[0,1],'-',c='g')
plt.plot(FPR,TPR)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# AUC
from sklearn.metrics import roc_auc_score
aucScore = roc_auc_score(labelTest,Pred)
print(f'The area under the ROC is {aucScore*100} %')

# Metrics
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score,f1_score
Matrix = confusion_matrix(Pred,labelTest)
Accuracy = accuracy_score(labelTest,Pred)
Precision = precision_score(labelTest,Pred)
Recall = recall_score(labelTest,Pred)
F1_score = f1_score(labelTest,Pred)
print(Matrix)
print(f'Accuracy {Accuracy}')
print(f'Precision {Precision}')
print(f'Recall {Recall}')
print(f'F1_score {F1_score}')

# Predicting A Positive Review
newReview = 'I love this restaurant so much'
newReview = re.sub('[^a-zA-Z]', ' ', newReview)
newReview = newReview.lower()
newReview = newReview.split()
ps = PorterStemmer()
allStopwords = stopwords.words('english')
allStopwords.remove('not')
newReview = [ps.stem(word) for word in newReview if not word in set(allStopwords)]
newReview = ' '.join(newReview)
newCorpus = [newReview]
Input = cv.transform(newCorpus).toarray()
revPred = classifier.predict(Input)
print(revPred)

# Predicting A Negative Review
newReview = 'I hate this restaurant so much'
newReview = re.sub('[^a-zA-Z]', ' ', newReview)
newReview = newReview.lower()
newReview = newReview.split()
ps = PorterStemmer()
allStopwords = stopwords.words('english')
allStopwords.remove('not')
newReview = [ps.stem(word) for word in newReview if not word in set(allStopwords)]
newReview = ' '.join(newReview)
newCorpus = [newReview]
Input = cv.transform(newCorpus).toarray()
revPred = classifier.predict(Input)
print(revPred)

