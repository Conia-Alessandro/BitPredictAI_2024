# BitPredictAI_2024
bitcoin prediction for group 1

# Useful Links

## Time series Forecasting
https://1660.medium.com/time-series-forecasting-predicting-bitcoin-prices-with-machine-learning-561ba75352da

## Deep learning REPO with npm package
https://github.com/karpathy/convnetjs

## Stack overflow on Linear interpolatiomn
https://stackoverflow.com/questions/27217694/python-pandas-linear-interpolate-y-over-x

## Linear regression Github repo
https://github.com/pawlodkowski/bitcoin-prediction/blob/master/BitcoinPricePrediction.ipynb

## Code to build a simple Convolutional Neural Network (CNN) for Bitcoin price prediction in Python using our data
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Step 1: Load the data
df = pd.read_csv('btc-usd-max.csv', index_col='snapped_at')

# Step 2: Preprocess the data
# Select features and target
features = ['price', 'market_cap', 'total_volume']
target = 'price'

# Normalize features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Split data into features and target
X = df[features].values
y = df[target].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the CNN architecture
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Step 4: Train the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Step 5: Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', loss)

# Step 6: Make predictions
predictions = model.predict(X_test)

# TODO: compare predictions with actual values to assess the model's performance.

```
