import numpy as np

#The prediction vector of the neural networks
y_pred=[0.6, 1.29, 1.99, 2.99, 3.4]

#The ground truth label
y_hat=[1, 1, 2, 2, 4]

#Mean squared error
MSE = np.sum(np.subtract(y_hat, y_pred))/len(y_hat)

print(MSE)

from sklearn.metrics import mean_absolute_error

#Example data: actual and predicted values
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 0]

#Manual calculation of MSE
man_manual = sum(abs(t-p) for t, p in zip(y_true, y_pred)) / len(y_true)
print("Mean absolute Error (Manual):", man_manual)

#2. Using sklearn for MAE
mae_sklearn  = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error (sklearn):", mae_sklearn)