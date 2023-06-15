import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# Read the stock market data from a CSV file
df = pd.read_csv('PSX.csv')

# Set the index of the DataFrame to the 'Date' column as datetime
df.index = pd.to_datetime(df.Date)

# Remove the 'Date' column from the DataFrame
del df['Date']

# Plot the historical close price
plt.figure(figsize=(16, 6))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price RS (RS)', fontsize=18)
plt.savefig('graph.png')

# Extract the 'Close' column as the input data
data = df.filter(['Close'])
dataset = data.values

# Calculate the training data length (95% of the dataset)
training_data_len = int(np.ceil(len(dataset) * 0.95))
print("Length of training data:", training_data_len)

# Scale the data between 0 and 1 using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Prepare the training data
train_data = scaled_data[0:int(training_data_len), :]
x_train = []
y_train = []

# Create training sequences with a window size of 60
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()

# Convert the training data to numpy arrays and reshape for LSTM input
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')

# model.fit(x_train, y_train, batch_size=1, epochs=1)
# Train the model
history = model.fit(x_train, y_train, validation_split=0.1, batch_size=1, epochs=10)


# Prepare the testing data
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

# Create testing sequences
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the testing data to a numpy array and reshape for LSTM input
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions on the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print("Root Mean Squared Error:", rmse)

# Prepare the data for plotting
train = data[:training_data_len]
valid = data[training_data_len:]
valid.loc[:, 'Predictions'] = predictions

# Plot the actual and predicted close prices
plt.figure(figsize=(16, 6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price RS (RS)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
plt.savefig('predict_graph.png')



# Plot the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_graph.png')