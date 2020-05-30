import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
pd.set_option('display.max_columns', None)
train=pd.read_csv('E:\DUKE\maching learning\Demo Datasets\Lesson 4\\bigmart_train.csv')
print(train.isnull().sum())
train['Outlet_Age']=2020-train['Outlet_Establishment_Year']
print(train.groupby('Item_Fat_Content').size())
print(train['Outlet_Size'].mode())
train['Outlet_Size']=train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
train['Item_Weight']=train['Item_Weight'].fillna(train['Item_Weight'].mean())
print(train.isnull().sum())
print(train['Item_Visibility'].hist(bins=20))
Q1=train['Item_Visibility'].quantile(0.25)
Q3=train['Item_Visibility'].quantile(0.75)
IQR=Q3-Q1
filt_train=train.query('(@Q1-1.5*@IQR)<=Item_Visibility<=(@Q3+1.5*@IQR)')
print(filt_train.shape,train.shape)
train=filt_train
train['Item_Visibility_bins']=pd.cut(train['Item_Visibility'],(0.000,0.065,0.13,0.2),labels=['Low Viz','Viz','High Viz'])
print(train['Item_Visibility_bins'].value_counts())
train['Item_Visibility_bins']=train['Item_Visibility_bins'].replace(np.nan,'Low Viz',regex=True)
train['Item_Fat_Content']=train['Item_Fat_Content'].replace(['LF','low fat'],'Low Fat')
train['Item_Fat_Content']=train['Item_Fat_Content'].replace('reg','Regular')

le =LabelEncoder()

train['Item_Fat_Content']=le.fit_transform(train['Item_Fat_Content'])
train['Item_Visibility_bins']=le.fit_transform(train['Item_Visibility_bins'])
train['Outlet_Size']=le.fit_transform(train['Outlet_Size'])
train['Outlet_Location_Type']=le.fit_transform(train['Outlet_Location_Type'])
print(train.head())
dummy=pd.get_dummies(train['Outlet_Type'])
#print(dummy)
train=pd.concat([train,dummy],axis=1)
train=train.drop(['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Type','Outlet_Establishment_Year'],axis=1)
print(train.columns)
x=train.drop('Item_Outlet_Sales',axis=1)
y=train['Item_Outlet_Sales']

test=pd.read_csv('E:\DUKE\maching learning\Demo Datasets\Lesson 4\\bigmart_test.csv')
test['Outlet_Size']=test['Outlet_Size'].fillna('Medium')
test['Item_Visibility_bins']=pd.cut(test['Item_Visibility'],(0.000,0.065,0.13,0.2),labels=['Low Viz','Viz','High Viz'])
test['Item_Weight']=test['Item_Weight'].fillna(test['Item_Weight'].mean())
test['Item_Visibility_bins']=test['Item_Visibility_bins'].fillna('Low Viz')

test['Item_Fat_Content']=le.fit_transform(test['Item_Fat_Content'])
test['Item_Visibility_bins']=le.fit_transform(test['Item_Visibility_bins'])
test['Outlet_Size']=le.fit_transform(test['Outlet_Size'])
test['Outlet_Location_Type']=le.fit_transform(test['Outlet_Location_Type'])
test['Outlet_Age']=2020-test['Outlet_Establishment_Year']
dummy2=pd.get_dummies(test['Outlet_Type'])
test=pd.concat([test,dummy2],axis=1)
X_test=test.drop(['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Type','Outlet_Establishment_Year'],axis=1)
from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.3, random_state =42)
Lin=LinearRegression()
Lin.fit(x_train,y_train)
print(Lin.coef_)
print(Lin.intercept_)
prediction=Lin.predict(x_test)
print(sqrt(mean_squared_error(y_test,prediction)))
prediction2=Lin.predict(X_test)
print(prediction2)

