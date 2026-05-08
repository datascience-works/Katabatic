from tensorflow.python.ops import math_ops
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from pandas import read_csv

class softmax_weight(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be under softmax (vectorised)."""

    def __init__(self, feature_uniques):
        if isinstance(feature_uniques, np.ndarray):
            feature_idxs = np.concatenate([[0], np.cumsum(feature_uniques)])
        else:
            feature_idxs = np.concatenate([[0], np.cumsum(np.asarray(feature_uniques))])
        feature_idxs = feature_idxs.astype(np.int64)

        # Build a segment-id vector: each row of w gets a feature group id
        seg_ids = np.zeros(int(feature_idxs[-1]), dtype=np.int64)
        for i in range(len(feature_idxs) - 1):
            seg_ids[feature_idxs[i]:feature_idxs[i + 1]] = i

        self.feature_idxs = feature_idxs.tolist()
        self.seg_ids = tf.constant(seg_ids)
        self.n_segments = int(len(feature_idxs) - 1)

    def __call__(self, w):
        # Compute group-wise log-softmax in one vectorised pass.
        seg_max = tf.math.unsorted_segment_max(w, self.seg_ids, self.n_segments)
        # Broadcast group max back to each row
        max_per_row = tf.gather(seg_max, self.seg_ids)
        shifted = w - max_per_row
        exp_shifted = tf.exp(shifted)
        seg_sum = tf.math.unsorted_segment_sum(exp_shifted, self.seg_ids, self.n_segments)
        logsumexp_per_row = tf.gather(tf.math.log(seg_sum), self.seg_ids) + max_per_row
        return w - logsumexp_per_row

    def get_config(self):
        return {'feature_idxs': self.feature_idxs}

def elr_loss(KL_LOSS):
  def loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)+ KL_LOSS
  return loss

def KL_loss(prob_fake):
    return np.mean(-np.log(np.subtract(1,prob_fake)))

def get_lr(input_dim, output_dim, constraint=None,KL_LOSS=0):
    model = tf.keras.Sequential()
    # declared input shape to keras
    model.add(tf.keras.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(output_dim, activation='softmax',kernel_constraint=constraint))
    model.compile(loss=elr_loss(KL_LOSS), optimizer='adam', metrics=['accuracy'])
    #log_elr = model.fit(*train_data, validation_data=test_data, batch_size=batch_size,epochs=epochs)
    return model 

def sample(*arrays, n=None, frac=None, random_state=None):
    '''
    generate sample random arrays from given arrays. The given arrays must be same size.
    
    Parameters:
    --------------
    *arrays: arrays to be sampled.

    n (int): Number of random samples to generate.

    frac: Float value between 0 and 1, Returns (float value * length of given arrays). frac cannot be used with n.

    random_state: int value or numpy.random.RandomState, optional. if set to a particular integer, will return same samples in every iteration.

    Return:
    --------------
    the sampled array(s). Passing in multiple arrays will result in the return of a tuple.

    '''
    random = np.random
    if isinstance(random_state, int):
        random = random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        random = random_state
    
    arr0 = arrays[0]
    original_size = len(arr0)
    if n == None and frac == None:
        raise Exception('You must specify one of frac or size.')
    if n == None:
        n = int(len(arr0) * frac)

    idxs = random.choice(original_size, n, replace=False)
    if len(arrays) > 1:
        sampled_arrays = []
        for arr in arrays:
            assert(len(arr) == original_size)
            sampled_arrays.append(arr[idxs])
        return tuple(sampled_arrays)
    else:
        return arr0[idxs]

DEMO_DATASETS = {
    'adult': {
        'link':'https://raw.githubusercontent.com/chriszhangpodo/discretizedata/main/adult-dm.csv',
        'params': {
            'dtype' : int
        }
    },
    'adult-raw':{
        'link':'https://drive.google.com/uc?export=download&id=1iA-_qIC1xKQJ4nL2ugX1_XJQf8__xOY0',
        'params': {}
    }
}

def get_demo_data(name='adult'):
    """
    Download demo dataset from internet.

    Parameters
    ----------
    name : str 
        Name of dataset. Should be one of ['adult', 'adult-raw'].

    Returns
    -------
    data : pandas.DataFrame
        the demo dataset.
    """
    assert(name in DEMO_DATASETS.keys())
    return read_csv(DEMO_DATASETS[name]['link'], **DEMO_DATASETS[name]['params'])