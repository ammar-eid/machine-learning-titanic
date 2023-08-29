#load data
import pandas as pd
X_train=pd.read_csv('train.csv')
y_train=X_train.Survived
X_train.drop(['Survived'],axis=1,inplace=True)
X_test=pd.read_csv('test.csv')
y_test=pd.read_csv('gender_submission0.csv')['Survived']
#data cleaning for training data & test data
#fill values in Age column
X_train['Age']=X_train['Age'].fillna(X_train['Age'].median())
X_test['Age']=X_test['Age'].fillna(X_test['Age'].median())

#fill values in Embarked column
X_train['Embarked']=X_train['Embarked'].fillna(X_train['Embarked'].mode()[0])

#fill values in Embarked column
X_test['Fare']=X_test['Fare'].fillna(X_test['Fare'].median())

#drop 3 columns
X_train.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)
X_test.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)

#convert string values in sex column into numeric
X_train[['Sex']]=pd.DataFrame.replace(X_train[['Sex']],to_replace=['male','female'],value=['1','0'])
X_test[['Sex']]=pd.DataFrame.replace(X_test[['Sex']],to_replace=['male','female'],value=['1','0'])

#convert string values in Embarked column into numeric
X_train[['Embarked']]=pd.DataFrame.replace(X_train[['Embarked']],to_replace=['C','S','Q'],value=['1','2','3'])
X_test[['Embarked']]=pd.DataFrame.replace(X_test[['Embarked']],to_replace=['C','S','Q'],value=['1','2','3'])

#scalling the data
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train[['Fare','Age']]=ss.fit_transform(X_train[['Fare','Age']])
X_test[['Fare','Age']]=ss.fit_transform(X_test[['Fare','Age']])

# #apply PCA
# from sklearn.decomposition import PCA
# model=PCA(n_components=3,whiten=True,svd_solver='full')
# model.fit(X_train)
# X_train=model.transform(X_train)
# X_test=model.transform(X_test)
#splitting data
from sklearn.model_selection import train_test_split
X_train,X_test0,y_train,y_test0 =train_test_split(X_train,y_train,test_size=0.3,shuffle=True,random_state=None)


#select model
from sklearn.linear_model import LogisticRegression
"""
logreg=LogisticRegression(penalty='l2', dual=False, tol=1e-4,
                          C=1.0, fit_intercept=True, intercept_scaling=1,
                          class_weight=None, random_state=None, solver='lbfgs',
                          max_iter=100, multi_class='auto', verbose=0,
                          warm_start=False, n_jobs=None, l1_ratio=None)
"""
# logreg=LogisticRegression(penalty='l2',solver='liblinear',C=0.1)
logreg=LogisticRegression(max_iter=1000,penalty='l2',solver='lbfgs',C=1.5,
                          random_state=7,tol=1e-2)
logreg.fit(X_train,y_train)
print('no iterations: ',logreg.n_iter_) 
y_pred0=logreg.predict(X_test0)
#convert from data series to array
import numpy as np
y_test=np.array(y_test)

# evaluation for CV
from sklearn.metrics import accuracy_score,confusion_matrix,zero_one_loss
print('confusion matrix 1 :\n',confusion_matrix(y_test0, y_pred0))
print('accuracy 1 :\n',accuracy_score(y_test0, y_pred0)*100,'%')
print('zero one loss 1 :\n',zero_one_loss(y_test0, y_pred0,normalize=False))

# evaluation for testing data
y_pred=logreg.predict(X_test)
print('confusion matrix :\n',confusion_matrix(y_test, y_pred))
print('accuracy :\n',accuracy_score(y_test, y_pred)*100,'%')
print('zero one loss :\n',zero_one_loss(y_test, y_pred,normalize=False))
# result=pd.DataFrame({'PassengerId':X_test['PassengerId'],'Survived':y_pred})
# result.to_csv('E://python codes & notes//titanic//gender_submission1.csv',index=False)