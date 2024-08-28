# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for the marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SURESH S
RegisterNumber:  212223040215
*/


import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
data=pd.read_csv("/content/student_scores.csv")


data.head()

![image](https://github.com/user-attachments/assets/8b041b46-aa0f-4eb9-85f6-c3e8424b18ad)

data.tail()

![image](https://github.com/user-attachments/assets/d060cf89-d3ee-4875-82b3-ee637d825d17)

data.info()

![image](https://github.com/user-attachments/assets/1347b13e-d5ad-4225-8548-4d1a1646fbde)

data.describe()

![image](https://github.com/user-attachments/assets/4dd1d0ba-8b09-4906-bf7a-e703b090634a)

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

![image](https://github.com/user-attachments/assets/f8990d90-0028-469f-b76c-6ed6989752fc)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


x_train.shape

![image](https://github.com/user-attachments/assets/8b828d10-6cce-40b5-9c4d-6e1cadbde288)

x_test.shape

![image](https://github.com/user-attachments/assets/da3b4773-1f77-42bb-bd12-c07a86b05ff4)

  from sklearn.linear_model import LinearRegression
  regressor=LinearRegression()
  regressor.fit(x_train,y_train)

![image](https://github.com/user-attachments/assets/106c2bb4-3e13-4484-a303-83a5341a4da1)

y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)

![image](https://github.com/user-attachments/assets/3fb0935f-fe2d-46dd-a7ba-4c6c53b65535)

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE= ",mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

![image](https://github.com/user-attachments/assets/a7a33bb9-fa9f-4832-a627-b4012d7adaa9)

plt.scatter(x_train,y_train,color="green")
plt.plot(x_train, regressor.predict(x_train), color="red")
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test, color="blue")
plt.plot(x_test, regressor.predict(x_test), color="silver")
plt.title('Test_set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

![image](https://github.com/user-attachments/assets/0b96177d-c399-4105-af42-7d829e9514aa)
![image](https://github.com/user-attachments/assets/392464da-22f2-4e6f-a358-56985f7aefba)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
