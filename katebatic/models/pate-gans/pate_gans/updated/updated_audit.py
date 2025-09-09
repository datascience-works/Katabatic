"""
PATE-GAN implementation by same authors from 2020 currently linked to the paper but not actually used in the paper.
source: https://github.com/vanderschaarlab/mlforhealthlabpub/blob/75beead341138094f89c1315ec3d722030d047cb/alg/pategan/pate_gan.py
"""

# """PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees Codebase.
#
# Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar,
# "PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees,"
# International Conference on Learning Representations (ICLR), 2019.
# Paper link: https://openreview.net/forum?id=S1zk9iRqF7
# Last updated Date: Feburuary 15th 2020
# Code author: Jinsung Yoon (jsyoon0823@gmail.com)
# """


# Necessary packages
import warnings
import numpy as np
from collections import defaultdict
import tensorflow.compat.v1 as tf
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


tf.disable_v2_behavior()
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")


def pate_lamda(x, teacher_models, lamda):
    '''Returns PATE_lambda(x).

    Args:
      - x: feature vector
      - teacher_models: a list of teacher models
      - lamda: parameter

    Returns:
      - n0, n1: the number of label 0 and 1, respectively
      - out: label after adding laplace noise.
    '''

    y_hat = list()

    for teacher in teacher_models:
        temp_y = teacher.predict(np.reshape(x, [1, -1]))
        y_hat = y_hat + [temp_y]

    y_hat = np.asarray(y_hat)
    n0 = float(sum(y_hat == 0))
    n1 = float(sum(y_hat == 1))

    # BUG!!!
    lap_noise = np.random.laplace(loc=0.0, scale=lamda)

    out = (n1 + lap_noise) / float(n0 + n1)
    out = int(out > 0.5)

    return n0, n1, out


class PG_UPDATED_AUDIT:
    '''Basic PATE-GAN framework.

    Args:
      - x_train: training data
      - parameters: PATE-GAN parameters
        - n_s: the number of student training iterations
        - batch_size: the number of batch size for training student and generator
        - k: the number of teachers
        - epsilon, delta: Differential privacy parameters
        - lamda: noise size

    Returns:
      - x_train_hat: generated training data by differentially private generator
    '''
    def __init__(self, X_shape,
                 epsilon=1.0, delta=1e-5, lamda=1.0, num_teachers=10,
                 lr=1e-4, max_iter=10000, n_s=1, batch_size=64, record_teachers=False):

        # Reset the graph
        tf.reset_default_graph()

        self.lr = lr
        # PATE-GAN parameters
        self.max_iter = max_iter
        # number of student training iterations
        self.n_s = n_s
        # number of teachers
        self.k = num_teachers
        self.teachers_seen_data = defaultdict(set)
        # epsilon
        self.epsilon = epsilon
        # delta
        self.delta = delta
        # lamda
        self.lamda = lamda

        # Other parameters
        # alpha initialize
        self.L = 20
        self.alpha = np.zeros([self.L])
        # initialize epsilon_hat
        self.epsilon_hat = 0

        # Network parameters
        self.no, self.dim = X_shape
        # Random sample dimensions
        self.z_dim = int(self.dim)
        # Student hidden dimension
        self.student_h_dim = int(self.dim)
        # Generator hidden dimension
        self.generator_h_dim = int(4 * self.dim)

        # Partitioning the data into k subsets
        self.partition_data_no = int(self.no / self.k)
        # number of batch size for student and generator training
        self.batch_size = min(batch_size, self.partition_data_no)

        self.record_teachers = record_teachers
        if self.record_teachers:
            self.teachers_dict = {i: np.zeros([self.max_iter, self.k]) for i in range(self.k)}

        # Placeholder
        # PATE labels
        self.device_spec = tf.DeviceSpec(device_type='CPU', device_index=0)

        with tf.device(self.device_spec.to_string()):
            self.Y = tf.placeholder(tf.float32, shape=[None, 1])
            # Random Variable
            self.Z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
            # only used in white-box auditing
            self.X = tf.placeholder(tf.float32, shape=[None, self.dim])

            # NN variables
            # Student
            self.S_W1 = tf.Variable(self.xavier_init([self.dim, self.student_h_dim]))
            self.S_b1 = tf.Variable(tf.zeros(shape=[self.student_h_dim]))

            self.S_W2 = tf.Variable(self.xavier_init([self.student_h_dim, 1]))
            self.S_b2 = tf.Variable(tf.zeros(shape=[1]))

            self.theta_S = [self.S_W1, self.S_W2, self.S_b1, self.S_b2]

            # Generator
            self.G_W1 = tf.Variable(self.xavier_init([self.z_dim, self.generator_h_dim]))
            self.G_b1 = tf.Variable(tf.zeros(shape=[self.generator_h_dim]))

            self.G_W2 = tf.Variable(self.xavier_init([self.generator_h_dim, self.generator_h_dim]))
            self.G_b2 = tf.Variable(tf.zeros(shape=[self.generator_h_dim]))

            self.G_W3 = tf.Variable(self.xavier_init([self.generator_h_dim, self.dim]))
            self.G_b3 = tf.Variable(tf.zeros(shape=[self.dim]))

            self.theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]

            # session
            self.sess = tf.Session()

    # Necessary Functions for buidling NN models
    # Xavier Initialization Definition
    @staticmethod
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    # Sample from uniform distribution
    @staticmethod
    def sample_Z(m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    # Models
    def generator(self, z):
        G_h1 = tf.nn.tanh(tf.matmul(z, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        G_out = tf.nn.sigmoid(tf.matmul(G_h2, self.G_W3) + self.G_b3)

        return G_out

    def student(self, x):
        S_h1 = tf.nn.relu(tf.matmul(x, self.S_W1) + self.S_b1)
        S_out = tf.matmul(S_h1, self.S_W2) + self.S_b2

        return S_out

    def fit(self, x_train, add_X_index=False):
        # Prerocess data
        # source: https://github.com/vanderschaarlab/mlforhealthlabpub/blob/75beead341138094f89c1315ec3d722030d047cb/alg/pategan/main_pategan_experiment.py#L63C7-L63C48
        self.processor = MinMaxScaler(clip=True)
        x_train = self.processor.fit_transform(x_train)

        if self.record_teachers:
            teachers_loaders = {}
        # add index to X to keep track of "teachers"
        if add_X_index:
            x_train = np.concatenate([np.reshape(range(len(x_train)), (-1, 1)), x_train], axis=1)

        # Partitioning the data into k subsets
        self.x_partition = list()
        idx = np.random.permutation(self.no)

        for i in range(self.k):
            temp_idx = idx[int(i * self.partition_data_no):int((i + 1) * self.partition_data_no)]
            temp_x = x_train[temp_idx, :]
            self.x_partition = self.x_partition + [temp_x]

        with tf.device(self.device_spec.to_string()):
            # Loss
            self.G_sample = self.generator(self.Z)
            S_fake = self.student(self.G_sample)
            self.S_predict = self.student(self.X)

            S_loss = tf.reduce_mean(self.Y * S_fake) - tf.reduce_mean((1 - self.Y) * S_fake)
            G_loss = -tf.reduce_mean(S_fake)

            # Optimizer
            S_solver = (tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(-S_loss, var_list=self.theta_S))
            G_solver = (tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(G_loss, var_list=self.theta_G))

            clip_S = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.theta_S]

            # Sessions
            self.sess.run(tf.global_variables_initializer())

            # Iterations
            it = 0
            while self.epsilon_hat < self.epsilon and it < self.max_iter:
                it += 1

                # 1. Train teacher models
                self.teacher_models = list()

                if self.record_teachers:
                    Z_mb = self.sample_Z(self.partition_data_no, self.z_dim)
                    G_mb = self.sess.run(self.G_sample, feed_dict={self.Z: Z_mb})

                for _ in range(self.k):
                    if not self.record_teachers:
                        Z_mb = self.sample_Z(self.partition_data_no, self.z_dim)
                        G_mb = self.sess.run(self.G_sample, feed_dict={self.Z: Z_mb})

                    # BUG!!!
                    temp_x = self.x_partition[i]
                    idx = np.random.permutation(len(temp_x[:, 0]))
                    X_mb = temp_x[idx[:self.partition_data_no], :]
                    if self.record_teachers and it == 1:
                        teachers_loaders[_] = X_mb
                    if add_X_index:
                        teach_idx, X_mb = X_mb[:, 0], X_mb[:, 1:]
                        teach_idx = set(teach_idx.astype(int))
                        self.teachers_seen_data[_].update(teach_idx)

                    X_comb = np.concatenate((X_mb, G_mb), axis=0)
                    Y_comb = np.concatenate((np.ones([self.partition_data_no,]), np.zeros([self.partition_data_no,])), axis=0)

                    model = LogisticRegression()
                    model.fit(X_comb, Y_comb)
                    self.teacher_models = self.teacher_models + [model]

                # 2. Student training
                for _ in range(self.n_s):

                    Z_mb = self.sample_Z(self.batch_size, self.z_dim)
                    G_mb = self.sess.run(self.G_sample, feed_dict={self.Z: Z_mb})
                    Y_mb = list()

                    for j in range(self.batch_size):
                        n0, n1, r_j = pate_lamda(G_mb[j, :], self.teacher_models, self.lamda)
                        Y_mb = Y_mb + [r_j]

                        # Update moments accountant
                        q = np.log(2 + self.lamda * abs(n0 - n1)) - np.log(4.0) - (self.lamda * abs(n0 - n1))
                        q = np.exp(q)

                        # Compute alpha
                        for l in range(self.L):
                            temp1 = 2 * (self.lamda**2) * (l + 1) * (l + 2)
                            temp2 = (1 - q) * (((1 - q) / (1 - q * np.exp(2 * self.lamda)))**(l + 1)) + q * np.exp(2 * self.lamda * (l + 1))
                            self.alpha[l] = self.alpha[l] + np.min([temp1, np.log(temp2)])

                    # PATE labels for G_mb
                    Y_mb = np.reshape(np.asarray(Y_mb), [-1, 1])

                    # Update student
                    _, D_loss_curr, _ = self.sess.run([S_solver, S_loss, clip_S], feed_dict={self.Z: Z_mb, self.Y: Y_mb})

                # Generator Update
                Z_mb = self.sample_Z(self.batch_size, self.z_dim)
                _, G_loss_curr = self.sess.run([G_solver, G_loss], feed_dict={self.Z: Z_mb})

                # epsilon_hat computation
                curr_list = list()
                for l in range(self.L):
                    temp_alpha = (self.alpha[l] + np.log(1 / self.delta)) / float(l + 1)
                    curr_list = curr_list + [temp_alpha]

                self.epsilon_hat = np.min(curr_list)

                if self.record_teachers:
                    if it == 1:
                        teachers_noise = self.sample_Z(self.partition_data_no, self.z_dim)

                        def BCELoss(y_true, y_pred):
                            _eps = 1e-7
                            y_pred = np.clip(y_pred, _eps, 1 - _eps)
                            term_0 = y_true * np.log(y_pred + _eps)
                            term_1 = (1 - y_true) * np.log(1 - y_pred + _eps)
                            return -np.mean(term_0 + term_1, axis=0)

                    fake_data = self.sess.run(self.G_sample, feed_dict={self.Z: teachers_noise})
                    label_fake = np.zeros([self.partition_data_no,])

                    for teacher_id in range(self.k):
                        teacher_model = self.teacher_models[teacher_id]

                        for teacher_j in range(self.k):
                            teacher_data = teachers_loaders[teacher_j]
                            label_real = np.ones([teacher_data.shape[0],])

                            data_combined = np.concatenate((teacher_data, fake_data), axis=0)
                            label_combined = np.concatenate((label_real, label_fake), axis=0)

                            output_combined = teacher_model.predict_proba(data_combined)[:, 1]
                            loss_combined = BCELoss(label_combined, output_combined)

                            self.teachers_dict[teacher_id][it - 1, teacher_j] = loss_combined
                    print(it, self.teachers_dict[0][it - 1])

    def generate(self, n_samples):
        with tf.device(self.device_spec.to_string()):
            # Outputs
            x_train_hat = self.sess.run([self.G_sample], feed_dict={self.Z: self.sample_Z(n_samples, self.z_dim)})[0]

            # ### Renormalization
            x_train_hat = self.processor.inverse_transform(x_train_hat)
            x_train_hat = np.clip(x_train_hat, self.processor.data_min_, self.processor.data_max_)

        return x_train_hat

    def sd_predict(self, x):
        with tf.device(self.device_spec.to_string()):
            x = self.processor.transform(x)
            s_predict = self.sess.run([self.S_predict], feed_dict={self.X: x})[0]
            s_predict = expit(s_predict)

        return s_predict

    def td_predict(self, x):
        with tf.device(self.device_spec.to_string()):
            x = self.processor.transform(x)
            t_predict = np.array([teacher.predict_proba(x)[:, 1] for teacher in self.teacher_models]).mean(axis=0)

        return t_predict
