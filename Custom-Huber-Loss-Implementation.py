import numpy as np
from sklearn.metrics import mean_absolute_error

def huber_loss(y_true, y_pred, epsilon=1.0):
    """
    Calculte  the huber loss manually.
    """
    error = np.array(y_true) - np.array(y_pred)
    loss = np.where(np.abs(error) <= epsilon, 0.5 * error ** 2, epsilon * (np.abs(error) - 0.5 * epsilon))
    return np.mean(loss)

#Example data
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

#Manual Huber Loss calculation
huber_loss_manual = huber_loss(y_true, y_pred, epsilon=1.0)
print( "Huber Loss (Manual):", huber_loss_manual)

#Using sklearn HuberRegressor