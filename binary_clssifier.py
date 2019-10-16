# -*- coding: utf-8 -*-
"""binary_clssifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gUFM21KzOU6GX_sR9nWa6JPTxYX9LH-j
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt # plotting
import os
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# reading csv file  
normal=pd.read_csv("ptbdb_normal.csv",header=None) 
#normal.head(2) 
#normal.head() 
#normal

abnormal=pd.read_csv("ptbdb_abnormal.csv",header=None) 
#abnormal

test=pd.read_csv("mitbih_test.csv",header=None) 
#test.head(2)
#test

train=pd.read_csv("mitbih_train.csv",header=None) 
#train

#y=train[187].values

y1_train=train[187].values
#y1_train

y1_test=test[187].values
#y1_train

# dataframe.size 
#size = y.size 
#print("size is : ",size)
  
# dataframe.shape 
#shape = y.shape 
#print("shape is : ",shape)



# dataframe.ndim 
#train_dim = y.ndim
#print("dimension is : ",train_dim)

#print(type(y))

y_train=[]                              # making multiclass to binary class now y_train containing either 0 or 1 
for i in y1_train:
  if i not in  [0.0]:
    y_train.append(1)
    #print("value is",i)
  else:
    y_train.append(0)
    #print("value",i)
    
#print(type(y1))    
#print(y1)

y_test=[]                              # making multiclass to binary class.now y_test containing either 0 or 1 
for i in y1_test:
  if i not in  [0.0]:
    y_test.append(1)
    #print("value is",i)
  else:
    y_test.append(0)
    #print("value",i)

#train.drop(['187','186'], axis = 1, inplace = True)   #to drp more than 2 column
train.drop([187], axis = 1, inplace = True)  #removing last column from train dataset
#train

test.drop([187], axis = 1, inplace = True)   #removing last column from test dataset
#test

x_train=train.values

x_test=test.values

x_train.shape

#seed = 2
#np.random.seed(seed)
                                     #dividing the train file data in 80:20 ratio of train and test                                                       
      
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)
#x_train, x_test, y_train, y_test = train_test_split(x,y1,test_size=0.2, random_state=2)



# SVM model
a_svm=svm.SVC(gamma='scale')
a_svm.fit(x_train,y_train)
y_pred_svm=a_svm.predict(x_test)


#from sklearn.metrics import classification_report, confusion_matrix
cm_svm=confusion_matrix(y_test,y_pred_svm)
print(cm_svm)
print("********************************")

report_svm=classification_report(y_test,y_pred_svm)
print(report_svm)

#Precision=TP/(TP+FP)
#Recall=TP/(TP+FN)
#F1 Score=(2*Recall*Precision)/(Recall+Precision)
#Accuracy=(TP+TN)/(TP+TN+FN+FP)
#Accuracy_Score_svm=accuracy_score(y_test,y_pred_svm)
#print('Average Accuracy:%0.2f +/- (%0.1f) %%' % (Accuracy_Score_svm.mean()*100, Accuracy_Score_svm.std()*100))


print("********************************")


Accuracy_svm=accuracy_score(y_test, y_pred_svm)
print("Accuracy of svm is : {0}".format(Accuracy_svm))

F1_score_svm= f1_score(y_test, y_pred_svm,average='weighted')
print('F1_score of svm is : {0}'.format(F1_score_svm))

Recall_svm= recall_score(y_test, y_pred_svm,average='weighted')
print('Recall of svm is : {0}'.format(Recall_svm))
      
Precision_svm= precision_score(y_test, y_pred_svm,average='weighted')      
print('Precision of svm is : {0}'.format(Precision_svm))



# Logistic Regression
a_lr=LogisticRegression()
a_lr.fit(x_train,y_train)
y_pred_lr=a_lr.predict(x_test)

cm_lr=confusion_matrix(y_test,y_pred_lr)
print(cm_lr)
print("********************************")

report_lr=classification_report(y_test,y_pred_lr)
print(report_lr)

print("********************************")

Precision_lr= precision_score(y_test, y_pred_lr,average='weighted')      
print('Precision of lr is : {0}'.format(Precision_lr))

Recall_lr= recall_score(y_test, y_pred_lr,average='weighted')
print('Recall of lr is : {0}'.format(Recall_lr))

F1_score_lr= f1_score(y_test, y_pred_lr,average='weighted')
print('F1_score of lr is : {0}'.format(F1_score_lr))

Accuracy_lr=accuracy_score(y_test, y_pred_lr)
print("Accuracy of lr is : {0}".format(Accuracy_lr))



#Naive Bayes
a_nb=GaussianNB()
a_nb.fit(x_train,y_train)
y_pred_nb=a_nb.predict(x_test)

cm_nb=confusion_matrix(y_test,y_pred_nb)
print(cm_nb)
print("********************************")

report_nb=classification_report(y_test,y_pred_nb)
print(report_nb)

print("********************************")

Precision_nb= precision_score(y_test, y_pred_nb,average='weighted')      
print('Precision of nb is : {0}'.format(Precision_nb))

Recall_nb= recall_score(y_test, y_pred_nb,average='weighted')
print('Recall of nb is : {0}'.format(Recall_nb))

F1_score_nb= f1_score(y_test, y_pred_nb,average='weighted')
print('F1_score of nb is : {0}'.format(F1_score_nb))

Accuracy_nb=accuracy_score(y_test, y_pred_nb)
print("Accuracy of nb is : {0}".format(Accuracy_nb))



#Stochastic Gradient Descent
a_sgd=SGDClassifier(loss='modified_huber', shuffle=True, random_state=101)
a_sgd.fit(x_train,y_train)
y_pred_sgd=a_sgd.predict(x_test)

cm_sgd=confusion_matrix(y_test,y_pred_sgd)
print(cm_sgd)
print("********************************")

report_sgd=classification_report(y_test,y_pred_sgd)
print(report_sgd)

print("********************************")

Precision_sgd= precision_score(y_test, y_pred_sgd,average='weighted')      
print('Precision of sgd is : {0}'.format(Precision_sgd))

Recall_sgd= recall_score(y_test, y_pred_sgd,average='weighted')
print('Recall of sgd is : {0}'.format(Recall_sgd))

F1_score_sgd= f1_score(y_test, y_pred_sgd,average='weighted')
print('F1_score of sgd is : {0}'.format(F1_score_sgd))

Accuracy_sgd=accuracy_score(y_test, y_pred_sgd)
print("Accuracy of sgd is : {0}".format(Accuracy_sgd))



#K-Nearest Neighbours
a_knn=KNeighborsClassifier(n_neighbors=15)
a_knn.fit(x_train,y_train)
y_pred_knn=a_knn.predict(x_test)

cm_knn=confusion_matrix(y_test,y_pred_knn)
print(cm_knn)
print("********************************")

report_knn=classification_report(y_test,y_pred_knn)
print(report_knn)

print("********************************")

Precision_knn= precision_score(y_test, y_pred_knn,average='weighted')      
print('Precision of knn is : {0}'.format(Precision_knn))

Recall_knn= recall_score(y_test, y_pred_knn,average='weighted')
print('Recall of knn is : {0}'.format(Recall_knn))

F1_score_knn= f1_score(y_test, y_pred_knn,average='weighted')
print('F1_score of knn is : {0}'.format(F1_score_knn))

Accuracy_knn=accuracy_score(y_test, y_pred_knn)
print("Accuracy of knn is : {0}".format(Accuracy_knn))



# Decision Binary Tree
a_dt=DecisionTreeClassifier(max_depth=10,random_state=101,max_features= None, min_samples_leaf=15)
a_dt.fit(x_train,y_train)
y_pred_dt=a_dt.predict(x_test)

cm_dt=confusion_matrix(y_test,y_pred_dt)
print(cm_dt)
print("********************************")

report_dt=classification_report(y_test,y_pred_dt)
print(report_dt)

print("********************************")

Precision_dt= precision_score(y_test, y_pred_dt,average='weighted')      
print('Precision of dt is : {0}'.format(Precision_dt))

Recall_dt= recall_score(y_test, y_pred_nb,average='weighted')
print('Recall of dt is : {0}'.format(Recall_dt))

F1_score_dt= f1_score(y_test, y_pred_dt,average='weighted')
print('F1_score of dt is : {0}'.format(F1_score_dt))

Accuracy_dt=accuracy_score(y_test, y_pred_dt)
print("Accuracy of dt is : {0}".format(Accuracy_dt))



# Random Forest Classifier
a_rf=RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,max_features=None,min_samples_leaf=30)
a_rf.fit(x_train,y_train)
y_pred_rf=a_rf.predict(x_test)


cm_rf=confusion_matrix(y_test,y_pred_rf)
print(cm_rf)
print("********************************")

report_rf=classification_report(y_test,y_pred_rf)
print(report_rf)

print("********************************")

Precision_rf= precision_score(y_test, y_pred_rf,average='weighted')      
print('Precision of rf model is : {0}'.format(Precision_rf))

Recall_rf= recall_score(y_test, y_pred_rf,average='weighted')
print('Recall of rf model is : {0}'.format(Recall_rf))

F1_score_rf= f1_score(y_test, y_pred_rf,average='weighted')
print('F1_score of rf model is : {0}'.format(F1_score_rf))

Accuracy_rf=accuracy_score(y_test, y_pred_rf)
print("Accuracy of rf model is : {0}".format(Accuracy_rf))



models_initial = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Random Forest', 'K-Nearest Neighbors', 'Naive Bayes','Stochastic Gradient Descent'],
    'Accuracy'    : [Accuracy_lr,   Accuracy_dt,   Accuracy_svm,    Accuracy_rf,   Accuracy_knn,   Accuracy_nb,   Accuracy_sgd],
    'Precision'   : [Precision_lr,  Precision_dt,  Precision_svm,   Precision_rf,  Precision_knn,  Precision_nb,  Precision_sgd],
    'Recall'      : [Recall_lr,     Recall_dt,     Recall_svm,      Recall_rf,     Recall_knn,     Recall_nb,     Recall_sgd],
    'F1_score'    : [F1_score_lr,   F1_score_dt,   F1_score_svm,    F1_score_rf,   F1_score_knn,   F1_score_nb,   F1_score_sgd],
    }, columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1_score'])

models_initial.sort_values(by='Accuracy', ascending=False)



