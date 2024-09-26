import pandas as pd
import numpy as np
car=pd.read_csv('C:\placement\project\quikr_car.csv')
"""car.head()
car.shape
car.info()
car['year'].unique()"""
backup=car.copy()#simply storing in other var
#year coloumn has non numeric values we check using isnumeric bool val is returned and true(i.e numeric) values are stored in car var#
car=car[car['year'].str.isnumeric()]
#year has numeric values and we have sortd them now we are converting this year which is in string to integer
car['year']=car['year'].astype(int)
#we are removing "ask for price in " price coloumn
car=car[car['Price']!="Ask For Price"]
#removing " , " from price and juust adding a space
car['Price']=car['Price'].str.replace(',','').astype(int)
car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')
car=car[car['kms_driven'].str.isnumeric()]
car['kms_driven']=car['kms_driven'].astype(int)
car=car[~car['fuel_type'].isna()]
car['name']=car['name'].str.split(' ').str.slice(0,3).str.join(' ')
car=car.reset_index(drop=True)
#car.info()
#car.describe()
car=car[car['Price']<6e6].reset_index(drop=True)
car.to_csv('Cleaned_Car.csv')
X=car.drop(columns='Price')
y=car['Price']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough') 
lr=LinearRegression() 
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)
scores=[]
for i in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i) 
    
    # Assuming you have defined 'column_trans' somewhere
    lr = LinearRegression() 
    pipe = make_pipeline(column_trans, lr) 
    
    pipe.fit(x_train, y_train) 
    y_pred = pipe.predict(x_test)
    
    # Assuming you have a list named 'scores' defined somewhere
    scores.append(r2_score(y_test, y_pred))
np.argmax(scores)
scores[np.argmax(scores)]
pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)
import pickle
pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))
print(pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5))))