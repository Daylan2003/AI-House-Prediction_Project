import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Housing.csv')

#print(df)
df_encoded = pd.get_dummies(df, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea','furnishingstatus'], drop_first=True) #one hot encode categorical columns

#Now to separate data as x and y

y = df_encoded['price']
x= df_encoded.drop('price', axis=1)
#print(y)
#print(x)

#Now data splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
#splits data into training and testing set. 20% for testing and 80# for training
#random state ensures the split is the same every time you run the code


#now lets do a linear regression
lr = LinearRegression() #makes the linear regression
lr.fit(x_train, y_train)    #trains the model between the x and y relationship

y_lr_train_pred = lr.predict(x_train)   #makes predictions on training data
y_lr_test_pred = lr.predict(x_test)     #makes predictions on unseen data


#Now to evaluate the data

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred) #calculates how well the model's predictions match y_train 
lr_train_r2 = r2_score(y_train, y_lr_train_pred)            #calculates how well the model experiences a variance in training data

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)    #calculates how well the model's predictions match y_test 
lr_test_r2 = r2_score(y_test, y_lr_test_pred)               #calculates how well the model experiences a variance in testing data


rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

# Evaluate Random Forest Regressor
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)



# Create a DataFrame to display results
results = pd.DataFrame({
    'Method': ['Linear Regression', 'Random Forest'],
    'Training MSE': [lr_train_mse, rf_train_mse],
    'Training R2': [lr_train_r2, rf_train_r2],
    'Test MSE': [lr_test_mse, rf_test_mse],
    'Test R2': [lr_test_r2, rf_test_r2]
})

#print(results)

#To create a plot of the results
plt.figure(figsize=(10,10))
plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1) #fits a polynomial of specified degree to the data
p = np.poly1d(z)                            #creates a polynomial function obtained from the function above

plt.plot(y_train, p(y_train), color='red') #regression line


plt.xlabel('Actual Prices')  # X-axis label
plt.ylabel('Predicted Prices')  # Y-axis label
plt.title('Actual vs. Predicted House Prices')  # Title of the plot


# Show the plot
plt.show()








