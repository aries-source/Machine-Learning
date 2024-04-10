import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('Churn_Modelling.csv')
Features = dataset.iloc[:,3:-1].values
Label = dataset.iloc[:,-1].values

# Encoding the Categorical Columns
# Gender
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Features[:,2] = label.fit_transform(Features[:,2])

# Geography
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
objectClass = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])], remainder='passthrough')
Features = np.array(objectClass.fit_transform(Features))

# Splitting Into Test and Train Sets
from sklearn.model_selection import train_test_split
indTrain, indTest, depTrain ,depTest = train_test_split(Features,Label,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
indTrain = scaler.fit_transform(indTrain)
indTest = scaler.transform(indTest)

# Stacking
def Stacking(model,train,y,test,n_fold):
  folds=StratifiedKFold(n_splits=n_fold)
  test_pred=np.empty((test.shape[0],1),float)
  train_pred=np.empty((0,1),float)
  y = pd.DataFrame(y)
  train = pd.DataFrame(train)
  
  for train_indices,val_indices in folds.split(train,y.values):
    x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
    y_train,y_val = y.iloc[train_indices],y.iloc[val_indices]
    model.fit(X=x_train,y=y_train)
    train_pred=np.append(train_pred,model.predict(x_val))
    test_pred=np.append(test_pred,model.predict(test))
    
  return test_pred.reshape(-1,1),train_pred

model1 = DecisionTreeClassifier(criterion='entropy',random_state=1)
model2 = KNeighborsClassifier()

# train base models and create new featurs
test_pred_1 ,train_pred_1=Stacking(model=model1,n_fold=10, train=indTrain,test=indTest,y=depTrain)
test_pred_2 ,train_pred_2=Stacking(model=model2,n_fold=10,train=indTrain,test=indTest,y=depTrain)

# convert into dataframe for later use
train_pred_1=pd.DataFrame(train_pred_1)
test_pred_1=pd.DataFrame(test_pred_1)
train_pred_2=pd.DataFrame(train_pred_2)
test_pred_2=pd.DataFrame(test_pred_2)

df = pd.concat([train_pred_1, train_pred_2], axis=1)
df_test = pd.concat([test_pred_1, test_pred_2], axis=1)

model = LogisticRegression(random_state=1)
model.fit(df,indTrain)
model.score(df_test, depTest)