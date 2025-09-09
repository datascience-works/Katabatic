"""
Original PATE-GAN implementation from 2018 used in the paper.
source: https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/0b0190bcd38a76c405c805f1ca774971fcd85233/alg/pategan/PATE_GAN.py
"""

# """
# Jinsung Yoon (0*/13/2018)
# PATEGAN
# """


# %% Packages
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import tensorflow.compat.v1 as tf
from scipy.special import expit


tf.disable_v2_behavior()
tf.get_logger().setLevel("ERROR")


# %% Function Start
class PG_ORIGINAL_AUDIT:
    def __init__(self, X_shape,
                 epsilon=1.0, delta=5, num_teachers=5,
                 niter=10000,
                 ):
        tf.reset_default_graph()

        # %% Parameters
        # Privacy
        self.epsilon = epsilon
        self.delta = delta
        self.num_teachers = num_teachers
        self.teachers_seen_data = defaultdict(set)
        self.niter = niter

        # Feature no and Sample no
        self.no, self.X_dim = X_shape[0], X_shape[1]
        # Random variable dimension
        self.z_dim = int(self.X_dim / 4)
        # Hidden unit dimensions
        self.h_dim = int(self.X_dim)
        # self.C_dim = 1

        # Batch size
        self.mb_size = min(128, self.no // self.num_teachers)
        # WGAN-GP Parameters
        self.lam = 10
        self.lr = 1e-4

        # BUG!!!
        self.lamda = np.sqrt(2 * np.log(1.25 * (10 ^ self.delta))) / self.epsilon

        # %% Algorithm Start
        # %% Placeholder
        self.device_spec = tf.DeviceSpec(device_type='CPU', device_index=0)

        with tf.device(self.device_spec.to_string()):
            # Feature
            self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
            # Label
            # self.Y = tf.placeholder(tf.float32, shape=[None, self.C_dim])
            # Random Variable
            self.Z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
            # Conditional Variable
            self.M = tf.placeholder(tf.float32, shape=[None, 1])

            # %% Discriminator
            # Discriminator
            # self.D_W1 = tf.Variable(self.xavier_init([self.X_dim + self.C_dim, self.h_dim]))
            self.D_W1 = tf.Variable(self.xavier_init([self.X_dim, self.h_dim]))
            self.D_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

            self.D_W2 = tf.Variable(self.xavier_init([self.h_dim, self.h_dim]))
            self.D_b2 = tf.Variable(tf.zeros(shape=[self.h_dim]))

            self.D_W3 = tf.Variable(self.xavier_init([self.h_dim, 1]))
            self.D_b3 = tf.Variable(tf.zeros(shape=[1]))

            self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]

            # %% Generator
            # self.G_W1 = tf.Variable(self.xavier_init([self.z_dim + self.C_dim, self.h_dim]))
            self.G_W1 = tf.Variable(self.xavier_init([self.z_dim, self.h_dim]))
            self.G_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

            self.G_W2 = tf.Variable(self.xavier_init([self.h_dim, self.h_dim]))
            self.G_b2 = tf.Variable(tf.zeros(shape=[self.h_dim]))

            self.G_W3 = tf.Variable(self.xavier_init([self.h_dim, self.X_dim]))
            self.G_b3 = tf.Variable(tf.zeros(shape=[self.X_dim]))

            self.theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]

            # session
            self.sess = tf.Session()

    # %% Necessary Functions
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

    # Sample from the real data
    @staticmethod
    def sample_X(m, n):
        return np.random.permutation(m)[:n]

    # %% Functions
    def generator(self, z):
        # inputs = tf.concat([z, y], axis=1)
        G_h1 = tf.nn.tanh(tf.matmul(z, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        G_log_prob = tf.nn.sigmoid(tf.matmul(G_h2, self.G_W3) + self.G_b3)

        return G_log_prob

    def discriminator(self, x):
        # inputs = tf.concat([x, y], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        out = (tf.matmul(D_h2, self.D_W3) + self.D_b3)

        return out

    def fit(self, X_train, add_X_index=False):
        # %% Data Preprocessing
        X_train = np.asarray(X_train)
        # X_train, Y_train = X_train[:, :-1], X_train[:, -1]

        # %% Data Normalization
        self.Min_Val = np.min(X_train, 0)
        X_train = X_train - self.Min_Val

        self.Max_Val = np.max(X_train, 0)
        X_train = X_train / (self.Max_Val + 1e-8)

        # add index to X to keep track of "teachers"
        if add_X_index:
            X_train = np.concatenate([np.reshape(range(len(X_train)), (-1, 1)), X_train], axis=1)

        with tf.device(self.device_spec.to_string()):
            # %%
            # Structure
            # self.G_sample = self.generator(self.Z, self.Y)
            self.G_sample = self.generator(self.Z)
            # D_real = self.discriminator(self.X, self.Y)
            self.D_real = self.discriminator(self.X)
            # D_fake = self.discriminator(self.G_sample, self.Y)
            D_fake = self.discriminator(self.G_sample)

            # %%
            D_entire = tf.concat(axis=0, values=[self.D_real, D_fake])

            # %%

            # Replacement of Clipping algorithm to Penalty term
            # 1. Line 6 in Algorithm 1
            eps = tf.random_uniform([self.mb_size, 1], minval=0., maxval=1.)
            X_inter = eps * self.X + (1. - eps) * self.G_sample

            # 2. Line 7 in Algorithm 1
            # grad = tf.gradients(self.discriminator(X_inter, self.Y), [X_inter, self.Y])[0]
            grad = tf.gradients(self.discriminator(X_inter), [X_inter])[0]
            grad_norm = tf.sqrt(tf.reduce_sum((grad)**2 + 1e-8, axis=1))
            grad_pen = self.lam * tf.reduce_mean((grad_norm - 1)**2)

            # Loss function
            D_loss = tf.reduce_mean((1 - self.M) * D_entire) - tf.reduce_mean(self.M * D_entire) + grad_pen
            G_loss = -tf.reduce_mean(D_fake)

            # Solver
            D_solver = (tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5).minimize(D_loss, var_list=self.theta_D))
            G_solver = (tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5).minimize(G_loss, var_list=self.theta_G))

            # %%
            # Sessions
            self.sess.run(tf.global_variables_initializer())

            # %%
            # Iterations
            for it in tqdm(range(self.niter), leave=False):

                for _ in range(self.num_teachers):
                    # %% Teacher Training
                    Z_mb = self.sample_Z(self.mb_size, self.z_dim)

                    # BUG!!!
                    # Teacher 1
                    X_idx = self.sample_X(self.no, self.mb_size)
                    X_mb = X_train[X_idx, :]
                    if add_X_index:
                        teach_idx, X_mb = X_mb[:, 0], X_mb[:, 1:]
                        teach_idx = set(teach_idx.astype(int))
                        self.teachers_seen_data[_].update(teach_idx)

                    # Y_mb = np.reshape(Y_train[X_idx], [self.mb_size, 1])

                    # %%
                    M_real = np.ones([self.mb_size,])
                    M_fake = np.zeros([self.mb_size,])

                    M_entire = np.concatenate((M_real, M_fake), 0)

                    # BUG!!!
                    Normal_Add = np.random.normal(loc=0.0, scale=self.lamda, size=self.mb_size * 2)

                    M_entire = M_entire + Normal_Add

                    M_entire = (M_entire > 0.5)

                    M_mb = np.reshape(M_entire.astype(float), (2 * self.mb_size, 1))

                    # _, D_loss_curr = self.sess.run([D_solver, D_loss], feed_dict={self.X: X_mb, self.Z: Z_mb, self.M: M_mb, self.Y: Y_mb})
                    _, D_loss_curr = self.sess.run([D_solver, D_loss], feed_dict={self.X: X_mb, self.Z: Z_mb, self.M: M_mb})

                # %% Generator Training
                Z_mb = self.sample_Z(self.mb_size, self.z_dim)

                X_idx = self.sample_X(self.no, self.mb_size)
                X_mb = X_train[X_idx, :]

                # Y_mb = np.reshape(Y_train[X_idx], [self.mb_size, 1])

                # _, G_loss_curr = self.sess.run([G_solver, G_loss], feed_dict={self.Z: Z_mb, self.Y: Y_mb})
                _, G_loss_curr = self.sess.run([G_solver, G_loss], feed_dict={self.Z: Z_mb})

    def generate(self, n_samples):
        with tf.device(self.device_spec.to_string()):
            # %%
            # %% Output Generation
            # New_X_train = self.sess.run([self.G_sample], feed_dict={self.Z: self.sample_Z(n_samples, self.z_dim), self.Y: np.reshape(y, [len(y), 1])})
            New_X_train = self.sess.run([self.G_sample], feed_dict={self.Z: self.sample_Z(n_samples, self.z_dim)})
            New_X_train = New_X_train[0]

            # ### Renormalization
            New_X_train = New_X_train * (self.Max_Val + 1e-8)
            New_X_train = New_X_train + self.Min_Val

        # return np.concatenate((New_X_train, y), axis=1)
        return New_X_train

    def sd_predict(self, X):
        with tf.device(self.device_spec.to_string()):
            X = X - self.Min_Val
            X = X / (self.Max_Val + 1e-8)
            s_predict = self.sess.run([self.D_real], feed_dict={self.X: X})[0]
            s_predict = expit(s_predict).squeeze()

        return s_predict
