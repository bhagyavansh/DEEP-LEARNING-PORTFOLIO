#Implementation of loss Functions using liberaries in python 1

import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Regression Loss Functions
y_true = np.array([3.0, 5.0, 7.0, 9,0])
y_pred = np.array([2.5, 5.5, 6.8, 8.9])

#Mean squared error
mse_loss = keras.losses.MeanSquaredError()
print("MSE:" , mse_loss(y_true, y_pred).numpy())

#Mean absolute error(MAE)
mae_loss = keras.losses.MeanAbsolute.Error()
print("MAE:", mae_loss(y_true, y_pred).numpy())

#Huber loss
huber_loss = keras.losses.Huber(delta=1.0)
print("Huber Loss:", huber_loss(y_true, y_pred).numpy())

#2. Classification Loss Functions
#Binary Cross_Entropy (BCE) - Binary Classification
y_true_binary = np.array([1.0, 0.0, 1.0, 0.0])
y_pred_binary = np.array([0.9, 0.2, 0.8, 0.1])

bce_loss = keras.losses.BinaryCrossentropy()
print("Binnary Cross-Entropy:", bce_loss(y_true_binary, y_pred_binary).numpy())

#Multi-class Cross_Entropy - Multi_Class Classification
y_true_multi = np.array([0,2, 1]) #Class indices
y_pred_multi = np.array([[2.0, 1.0, 0.1],
                         [0.1, 1.0, 2.0],
                         [1.0, 2.0, 0.1]])

#Note: Use Logits=True if inputs are row logits
cross_entropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print("Multi-class Cross_Entropy:", cross_entropy_loss(y_true_multi, y_pred_multi).numpy())

def contrastive_loss(y_true, y_pred, margin=1.0):
    """Contrastive loss Function."""
    return tf.reduce.mean((1 - y_true) * tf.square(y_pred) +
                          y_true * tf.square(tf.maximum(margin - y_pred, 0)))

y_true_contrastive = np.array([1.0, 0.0, 1.0, 0.0])
y_pred_contrastive = np.array([0.5, 0.2, 0.3, 0.1])

print("Contrastive Loss:", contrastive_loss(y_true_contrastive, y_pred_contrastive).numpy())