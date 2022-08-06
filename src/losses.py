import tensorflow as tf
from tensorflow import keras
from keras import backend as K


def distance_loss(distances):
    def loss(y_true, y_pred):
        weights = K.reshape(distances + 1, tf.shape(y_true)[0])
        bce = K.binary_crossentropy(y_true, y_pred)
        return keras.backend.mean(bce*weights)
    return loss;


def iou(y_true, y_pred):
  intersection = keras.backend.sum(y_true*y_pred)
  union = keras.backend.sum(y_true) + keras.backend.sum(y_pred) - intersection
  return (union-intersection)/union + keras.losses.BinaryCrossentropy(
      label_smoothing=0.1)(y_true, y_pred)
