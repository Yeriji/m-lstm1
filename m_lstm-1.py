import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from datetime import datetime
import streamlit as st

# Load the data
df = pd.read_csv('GE.csv')

# Display the data
st.write(df.head())  # Display the first few rows of the dataframe

# Separate dates for future plotting
train_dates = pd.to_datetime(df['Date'])
st.write(train_dates.tail(15))  # Display the last 15 dates

# Variables for training
cols = list(df)[1:6]

# Display the columns used for training
st.write(cols)

# New dataframe with only training data - 5 columns
df_for_training = df[cols].astype(float)

# Normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# Define the input data shape for LSTM
n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14  # Number of past days we want to use to predict the future.

# Empty lists to be populated using formatted training data
trainX = []
trainY = []

# Reformat input data into shape: (n_samples x timesteps x n_features)
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

# Display the shapes of trainX and trainY
st.write('trainX shape == {}.'.format(trainX.shape))
st.write('trainY shape == {}.'.format(trainY.shape))

# Define the Autoencoder model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')

# Fit the model
history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
st.pyplot()

# Predicting
n_past = 16
n_days_for_prediction = 15  # Let us predict the past 15 days

# Create a list of future dates for prediction
predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='B').tolist()

# Make prediction
prediction = model.predict(trainX[-n_days_for_prediction:])

# Perform inverse transformation to rescale back to original range
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]

# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

# Create a dataframe for the forecasted data
df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Open': y_pred_future})
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

# Filter the original data for consistent date format
original = df[['Date', 'Open']]
original['Date'] = pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2020-05-01']

# Create line plots using keyword arguments
fig, ax = plt.subplots()
sns.lineplot(x="Date", y="Open", data=original, ax=ax, label='Original')
sns.lineplot(x="Date", y="Open", data=df_forecast, ax=ax, label='Forecast')
ax.set(xlabel='Date', ylabel='Open')

# Display the plot
st.pyplot(fig)
