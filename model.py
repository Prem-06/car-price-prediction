from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data=pd.read_csv('car_data.csv')
model=LinearRegression()
x = data['Present_Price'].values.reshape(-1, 1) 
y = data['Selling_Price'].values 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')

joblib.dump(model, 'model.pkl')
