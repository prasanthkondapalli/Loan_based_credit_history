# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 11:16:38 2020

@author: Prasanth
"""

import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv('F:\ASS\LOAN')

df.columns



x=df.copy()
del x['default (Target)']

y=df['default (Target)']


x.isnull().sum()# no null values

plt.boxplot(x['months_loan_duration'])
plt.boxplot(x['amount'])
plt.boxplot(x['age'])

per=x['months_loan_duration'].quantile([0,0.925]).values
x['months_loan_duration']=x['months_loan_duration'].clip(per[0],per[1])

per=x['amount'].quantile([0,0.927]).values
x['amount']=x['amount'].clip(per[0],per[1])

per=x['age'].quantile([0,0.97]).values
x['age']=x['age'].clip(per[0],per[1])


from sklearn.preprocessing import LabelEncoder
lbe=LabelEncoder()


x['months_loan_duration']=lbe.fit_transform(x['months_loan_duration'])
x['checking_balance']=lbe.fit_transform(x['checking_balance'])
x['credit_history']=lbe.fit_transform(x['credit_history'])
x['purpose']=lbe.fit_transform(x['purpose'])
x['amount']=lbe.fit_transform(x['amount'])
x['savings_balance']=lbe.fit_transform(x['savings_balance'])
x['employment_duration']=lbe.fit_transform(x['employment_duration'])
x['age']=lbe.fit_transform(x['age'])
x['other_credit']=lbe.fit_transform(x['other_credit'])
x['housing']=lbe.fit_transform(x['housing'])
x['job']=lbe.fit_transform(x['job'])
x['phone']=lbe.fit_transform(x['phone'])


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sclr=StandardScaler()
sclr.fit(xtrain)
sclr.fit(xtest)

xtrain=sclr.transform(xtrain)
xtest=sclr.transform(xtest)


from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()

reg.fit(xtrain,ytrain)


ypred_ts=reg.predict(xtest)
ypred_tr=reg.predict(xtrain)

from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred_ts)#72
accuracy_score(ytrain,ypred_tr)#74

reg.coef_
reg.score

from sklearn.metrics import classification_report
print(classification_report(ytest,ypred_ts))
print(classification_report(ytrain,ypred_tr))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,ypred_ts))
print(confusion_matrix(ytrain,ypred_tr))




'''decision tree'''
from sklearn import tree
dct=tree.DecisionTreeClassifier()
dct.fit(xtrain,ytrain)

ypred_dtr=dct.predict(xtrain)
ypred_dts=dct.predict(xtest)

accuracy_score(ytest,ypred_dts)#66
accuracy_score(ytrain,ypred_dtr)#100


'''random forest'''
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)

ypred_rts=rf.predict(xtest)
ypred_rtr=rf.predict(xtrain)

accuracy_score(ytrain,ypred_rtr)#98
accuracy_score(ytest,ypred_rts)#77

'''#knn'''
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(xtrain,ytrain)

ypred_knts=knn.predict(xtest)
ypred_kntr=knn.predict(xtrain)

accuracy_score(ytrain,ypred_kntr)#78
accuracy_score(ytest,ypred_knts)#69


'''Naive bayes'''
from sklearn.naive_bayes import GaussianNB
gss=GaussianNB()
gss.fit(xtrain,ytrain)

ypred_nvts=gss.predict(xtest)
ypred_nvtr=gss.predict(xtrain)

accuracy_score(ytest,ypred_nvts)#71
accuracy_score(ytrain,ypred_nvtr)#73

print(classification_report(ytest,ypred_nvts))
print(classification_report(ytrain,ypred_nvtr))

gss.sigma_
gss.theta_
gss.score



'''SVM'''
from sklearn import svm
sup=svm.SVC()
sup.fit(xtrain,ytrain)

ypred_sts=sup.predict(xtest)
ypred_str=sup.predict(xtrain)

accuracy_score(ytest,ypred_sts)#74
accuracy_score(ytrain,ypred_str)#84


