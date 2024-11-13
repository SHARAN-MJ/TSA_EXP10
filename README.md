## DEVELOPED BY: SHARAN MJ
## REGISTER NO: 212222240097
## DATE:
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/AMZN.csv')

# Convert 'datesold' to datetime and set as index
data['date'] = pd.to_datetime(data['Date'])
data.set_index('date', inplace=True)

# Print the available columns to check for correct column name
print(data.columns)  

# Plot the time series data, use 'Low' instead of 'Min' if 'Low' represents minimum price
plt.plot(data.index, data['Low'])  # Changed 'Min' to 'Low'
plt.xlabel('Date')
plt.ylabel('Price of amazon stock')
plt.title('Price Time Series')
plt.show()

# Function to check stationarity using ADF test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Use 'Low' for stationarity check
check_stationarity(data['Low']) # Changed 'Min' to 'Low'

# Plot ACF and PACF to determine SARIMA parameters
plot_acf(data['Low']) # Changed 'Min' to 'Low'
plt.show()
plot_pacf(data['Low']) # Changed 'Min' to 'Low'
plt.show()

# Train-test split (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data['Low'][:train_size], data['Low'][train_size:] # Changed 'Min' to 'Low'

# Define and fit the SARIMA model on training data
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Make predictions on the test set
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot the actual vs predicted values
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Low') # Changed 'Min' to 'Low'
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()

```
### OUTPUT:
![Screenshot 2024-11-13 155644](https://github.com/user-attachments/assets/9594e767-135f-4eba-80ac-7ba9a104588f)
![Screenshot 2024-11-13 155651](https://github.com/user-attachments/assets/e83a6689-20c4-4196-9163-07ae91973088)
![Screenshot 2024-11-13 155658](https://github.com/user-attachments/assets/3e169ca8-fa65-40c7-ac09-6a147a849a14)
![Screenshot 2024-11-13 155706](https://github.com/user-attachments/assets/dda03d0e-a43e-4351-bb42-506268e282ad)
![Screenshot 2024-11-13 155822](https://github.com/user-attachments/assets/20808c2d-b160-49d8-a572-51330b6baa11)



### RESULT:
Thus, the pyhton program based on the SARIMA model is executed successfully.
