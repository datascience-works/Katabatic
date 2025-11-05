import numpy as np
import tensorflow as tf
from tensorflow import keras


def gradient_penalty(critic, real_samples, fake_samples, labels, batch_size):
    """
    Calculates the gradient penalty for WGAN-GP.
    
    Parameters
    ----------
    critic : keras.Model
        The critic (discriminator) model
    real_samples : tf.Tensor
        Real data samples
    fake_samples : tf.Tensor
        Generated fake samples
    labels : tf.Tensor
        Conditional labels
    batch_size : int
        Size of the batch
        
    Returns
    -------
    penalty : tf.Tensor
        Gradient penalty value
    """
    alpha = tf.random.uniform([batch_size, 1], 0., 1.)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        # Pass interpolated samples and labels as separate inputs
        predictions = critic([interpolated, labels], training=True)
    
    gradients = tape.gradient(predictions, interpolated)
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
    penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
    
    return penalty


def wasserstein_loss(y_true, y_pred):
    """
    Wasserstein loss for WGAN.
    
    Parameters
    ----------
    y_true : tf.Tensor
        True labels (1 for real, -1 for fake)
    y_pred : tf.Tensor
        Predicted values from critic
        
    Returns
    -------
    loss : tf.Tensor
        Wasserstein loss
    """
    return -tf.reduce_mean(y_true * y_pred)


def sample(*arrays, n=None, frac=None, random_state=None):
    '''
    Generate sample random arrays from given arrays. The given arrays must be same size.
    
    Parameters
    ----------
    *arrays : array-like
        Arrays to be sampled.
    n : int, optional
        Number of random samples to generate.
    frac : float, optional
        Float value between 0 and 1, Returns (float value * length of given arrays). 
        frac cannot be used with n.
    random_state : int or numpy.random.RandomState, optional
        If set to a particular integer, will return same samples in every iteration.

    Returns
    -------
    sampled_arrays : array or tuple of arrays
        The sampled array(s). Passing in multiple arrays will result in the return of a tuple.
    '''
    random = np.random
    if isinstance(random_state, int):
        random = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        random = random_state
    
    arr0 = arrays[0]
    original_size = len(arr0)
    if n is None and frac is None:
        raise Exception('You must specify one of frac or n.')
    if n is None:
        n = int(len(arr0) * frac)

    idxs = random.choice(original_size, n, replace=False)
    if len(arrays) > 1:
        sampled_arrays = []
        for arr in arrays:
            assert len(arr) == original_size
            sampled_arrays.append(arr[idxs])
        return tuple(sampled_arrays)
    else:
        return arr0[idxs]

