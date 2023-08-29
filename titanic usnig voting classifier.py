#load data
import pandas as pd
X_train=pd.read_csv('train.csv')
y_train=X_train.Survived
X_train.drop(['Survived'],axis=1,inplace=True)
X_test=pd.read_csv('test.csv')
y_test=pd.read_csv('gender_submission0.csv')['Survived']
#data cleaning for training data & test data
#fill values in Age column
X_train['Age']=X_train['Age'].fillna(X_train['Age'].min())
X_test['Age']=X_test['Age'].fillna(X_test['Age'].min())

#fill values in Embarked column
X_train['Embarked']=X_train['Embarked'].fillna(X_train['Embarked'].mode()[0])

#fill values in Embarked column
X_test['Fare']=X_test['Fare'].fillna(X_test['Fare'].median())

#drop 3 columns
X_train.drop(['Name','PassengerId','Cabin','Ticket'],axis=1,inplace=True)
PassengerId=X_test['PassengerId']
X_test.drop(['Name','PassengerId','Cabin','Ticket'],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#convert string values in Embarked column into numeric
X_train[['Embarked']]=le.fit_transform(X_train[['Embarked']])
X_test[['Embarked']]=le.fit_transform(X_test[['Embarked']])

#convert string values in sex column into numeric
X_train[['Sex']]=le.fit_transform(X_train[['Sex']])
X_test[['Sex']]=le.fit_transform(X_test[['Sex']])

#scalling the data
from sklearn.preprocessing import StandardScaler
ss=StandardScaler(with_mean=True,with_std=True)
X_train[['Fare','Age']]=ss.fit_transform(X_train[['Fare','Age']])
X_test[['Fare','Age']]=ss.fit_transform(X_test[['Fare','Age']])

#splitting data
from sklearn.model_selection import train_test_split
X_train,X_test0,y_train,y_test0=train_test_split(X_train,y_train,test_size=0.3,shuffle=True,random_state=135)

    
#modelling
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=3,
                           min_samples_leaf=3,min_samples_split=2)
lg=LogisticRegression(max_iter=1000,penalty='l2',solver='lbfgs',C=1.5,
                          random_state=7,tol=1e-2)
svc=SVC(C=10,kernel='linear',gamma='auto',coef0=0.0001,tol=1e-3,random_state=135)
rfc=RandomForestClassifier()
sgd=SGDClassifier()
clf=VotingClassifier(estimators=[('sgd',sgd),('lg',lg),('tree',dtc),('svc',svc),('rfc',rfc)],
                     voting='hard',weights=[1,3,5,7,9])

clf.fit(X_train,y_train)


y_pred0=clf.predict(X_test0)
# evaluation for CV
from sklearn.metrics import accuracy_score,confusion_matrix,zero_one_loss
print('confusion matrix 1 :\n',confusion_matrix(y_test0, y_pred0))
print('accuracy 1 :\n',accuracy_score(y_test0, y_pred0)*100,'%')
# print('zero one loss 1 :\n',zero_one_loss(y_test0, y_pred0,normalize=False))

# evaluation for test data
y_pred=clf.predict(X_test)
print('confusion matrix 1 :\n',confusion_matrix(y_test, y_pred))
print('accuracy 1 :\n',accuracy_score(y_test, y_pred)*100,'%')
# print('zero one loss 1 :\n',zero_one_loss(y_test, y_pred,normalize=False))
# result=pd.DataFrame({'PassengerId':PassengerId,'Survived':y_pred})
# result.to_csv('E://python codes & notes//titanic//gender_submission.csv',index=False)