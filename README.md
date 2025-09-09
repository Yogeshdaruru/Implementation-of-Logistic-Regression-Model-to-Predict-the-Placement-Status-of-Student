# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data


2.Print the placement data and salary data.


3.Find the null and duplicate values.


4.Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Dharuru Yogesh
RegisterNumber:  212224230063
*/
```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## TOP 5 ELEMENTS
<img width="1221" height="226" alt="319847010-04b0fcce-c444-42d7-b055-a7e6ac77678d" src="https://github.com/user-attachments/assets/061ea964-dbf1-4a67-89b6-d5ec6cc95346" />
<img width="1091" height="240" alt="319847115-e9963d89-9a92-4575-b01e-677a85537cf1" src="https://github.com/user-attachments/assets/574942f2-1e0b-4a92-a8ed-bb1299e7350b" />
<img width="982" height="497" alt="319847140-e21f9740-4b51-489a-a471-f59892e74f69" src="https://github.com/user-attachments/assets/61643bc0-ffd7-40fe-8d9c-c0ed8c33d0eb" />

## DATA DUPLICATE
<img width="61" height="48" alt="319847161-210eead6-4770-4e23-b794-b1a5f9428e0d" src="https://github.com/user-attachments/assets/2a663534-a103-4354-96a4-45a3966f68d1" />


## PRINT DATA
<img width="982" height="502" alt="319847188-2ad5eecf-18b0-47ba-84e0-9fc00443541e" src="https://github.com/user-attachments/assets/6c87566c-5476-43f3-9634-0489a8212481" />


## DATA_STATUS
<img width="922" height="510" alt="319847197-3a6e8cae-f77d-414e-b674-020cd6f87fae" src="https://github.com/user-attachments/assets/0da8b1d4-f47e-4cea-8cdc-a158356fe456" />


## Y_PREDICTION ARRAY
<img width="586" height="263" alt="319847224-ff5c9ca0-aa45-4b9f-a727-8ff5bb079ba9" src="https://github.com/user-attachments/assets/b8d458e4-f8a9-42d3-b0fd-4e59a903b96b" />


## CONFUSION ARRAY
<img width="762" height="71" alt="319847287-1e6c4ab6-c90b-4278-be39-1f7721487b21" src="https://github.com/user-attachments/assets/b74ada15-9aa9-46fa-9222-51bf3f377130" />


## ACCURACY VALUE
<img width="210" height="51" alt="319847271-e70452c6-a4ee-4428-be43-bfd1d677f5fa" src="https://github.com/user-attachments/assets/9acb3f1f-5b5f-4fc3-b3b9-52ba9d963cf3" />


## CLASSFICATION REPORT
<img width="582" height="176" alt="319847315-70144eaf-0396-4780-ac47-a34e16f85ad8" src="https://github.com/user-attachments/assets/c70ac803-1fff-434e-be33-d95430c46791" />


## PREDICTION
<img width="303" height="33" alt="319847345-791654fd-4387-4d63-a415-cbaa1564a6a1" src="https://github.com/user-attachments/assets/dd56c57f-1f04-426c-9a0d-6a837fe7f6fd" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
