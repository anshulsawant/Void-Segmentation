import tensorflow as tf
from tensorflow import keras
from keras import backend as K


def distance_loss(y_true, y_pred, distances):
    def loss(y_true, y_pred):
        weights = K.reshape(distances + 1, tf.shape(y_true)[0])
        bce = K.binary_crossentropy(y_true, y_pred)
        return keras.backend.mean(bce*weights)
    return loss;
