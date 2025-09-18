import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Input numbers (like 1, 2, 3, 4, 5)
numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 2. Prepare data: input (X) = [1,2,3], output (y) = 4 (like predicting next number)
X = []
y = []
for i in range(len(numbers) - 3):
    X.append(numbers[i:i+3])
    y.append(numbers[i+3])

X = np.array(X)
y = np.array(y)

# 3. Reshape input to [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# 4. Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 5. Train the model
model.fit(X, y, epochs=200, verbose=0)

# 6. Predict next number after [8, 9, 10]
test_input = np.array([8, 9, 10]).reshape((1, 3, 1))
predicted = model.predict(test_input, verbose=0)
print("Predicted next number:", predicted[0][0])
