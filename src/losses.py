import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import layers

def distance_loss(y_true, y_pred):
  s = tf.cast(y_true.shape[1]/2, tf.int32)
  y_true_mask = y_true[:,0:(s-1)]
  y_true_dist = y_true[:,s:-1]
  bce = K.binary_crossentropy(y_true_mask, y_pred[:,0:(s-1)])
  return K.mean(y_true_dist*bce)


def iou(y_true, y_pred):
  intersection = keras.backend.sum(y_true*y_pred)
  union = keras.backend.sum(y_true) + keras.backend.sum(y_pred) - intersection
  return (union-intersection)/union + keras.losses.BinaryCrossentropy(
      label_smoothing=0.1)(y_true, y_pred)
