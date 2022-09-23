import tensorflow as tf
from tensorflow import keras
K = keras.backend
import numpy as np

def reg_loss(y_true, y_pred, eps=1e-6):
    x = eps + (1 - 2 * eps) * y_pred
    y = y_true
    
    loss = K.sum(y * K.log(x), axis=-1)
    return -K.mean(loss)

def BayesSoftmax(x, prob_signal=None, prob_noise=None, prob_frac=1.):
    if prob_signal is None:
        if prob_noise is None:
            if prob_frac is None:
                raise ValueError("Must set at least one of prob_noise or prob_frac, if prob_sig is not set.")
            else:
                prob_signal = 1 / (prob_frac + 1)
        else:
            prob_signal = 1 - prob_noise
    if prob_noise is None:
        prob_noise = 1 - prob_signal
    assert abs(1 - prob_signal - prob_noise) < K.epsilon()
    logprob = tf.convert_to_tensor(np.array([np.log(prob_signal), np.log(prob_noise)], dtype=np.float32))
    return keras.activations.softmax(x+logprob)
