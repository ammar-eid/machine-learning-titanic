#load data
import pandas as pd
X_train=pd.read_csv('train.csv').set_index('PassengerId')
X_test=pd.read_csv('test.csv').set_index('PassengerId')
y_test=pd.read_csv('gender_submission0.csv')['Survived']
df=pd.concat([X_train,X_test],axis=0,sort=False)

#data cleaning for training data & test data
#fill values in Age column
df['Age']=df['Age'].fillna(df['Age'].median())


#fill values in Embarked column
df['Embarked']=df['Embarked'].fillna('C')


#fill values in Embarked column
df['Fare']=df['Fare'].fillna(df['Fare'].median())

#drop 3 columns
df.drop(['Name','Cabin','Ticket'],axis=1,inplace=True)



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


#convert string values in Embarked column into numeric
df[['Embarked']]=le.fit_transform(df[['Embarked']])

#convert string values in sex column into numeric
df[['Sex']]=le.fit_transform(df[['Sex']])

#scalling the data
from sklearn.preprocessing import MaxAbsScaler
ss=MaxAbsScaler()
df[['Fare','Age']]=ss.fit_transform(df[['Fare','Age']])

y=df['Survived']
X=df.drop('Survived',axis=1)
X_train,y_train=X[:891],y[:891]
X_test=X[:-891]

#splitting data
from sklearn.model_selection import train_test_split
X_train,X_test0,y_train,y_test0=train_test_split(X_train,y_train)

#modelling
from sklearn.svm import SVC
clf=SVC(C=1,kernel='linear',gamma='auto',coef0=0.0001,tol=1e-3,random_state=135)
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
# result=pd.DataFrame({'PassengerId':X_test.index,'Survived':y_pred})
# result.to_csv('E://python codes & notes//titanic//gender_submission.csv',index=False)






