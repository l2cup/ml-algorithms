%tensorflow_version 1.x
%matplotlib inline

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

tf.reset_default_graph()

def create_feature_matrix(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)

# Constants
filename = '/content/corona.csv'

lambdas = [0.001, 0.01, 0.1, 0, 1, 10, 100]

degree = 3

nb_epochs = 100

colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']

legends = []

data = dict()

losses = np.ones(shape=(len(lambdas), nb_epochs))

data['x'], data['y'] = np.loadtxt(filename, delimiter=',', unpack=True,
                                  skiprows=0, usecols=(0, 1))

np.set_printoptions(suppress=True, precision=5)

nb_samples = len(data['x'])

indices = np.random.permutation(nb_samples)

data['x'] = data['x'][indices]
data['y'] = data['y'][indices]
data['x'] = (data['x'] - np.mean(data['x'])) / np.std(data['x'])
data['y'] = (data['y'] - np.mean(data['x'])) / np.std(data['y'])

plt.scatter(data['x'], data['y'])

for index, lambd in enumerate(lambdas):

  dataX = create_feature_matrix(data['x'], degree)

  X = tf.placeholder(shape=(None, degree), dtype=tf.float32)
  Y = tf.placeholder(shape=(None), dtype=tf.float32)
  w = tf.Variable(tf.zeros(degree))
  bias = tf.Variable(0.0)

  w_col = tf.reshape(w, (degree, 1))
  hyp = tf.add(tf.matmul(X, w_col), bias)

  Y_col = tf.reshape(Y, (-1, 1))

  l2_reg = lambd * tf.reduce_mean(tf.square(w))

  mse = tf.reduce_mean(tf.square(hyp - Y_col))
  loss = tf.add(mse, l2_reg)

  opt_op = tf.train.AdamOptimizer().minimize(loss)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(nb_epochs + 1):
      epoch_loss = 0
      for i in range(nb_samples):
        feed = {X: dataX[i].reshape((1, degree)),
                Y: data['y'][i]}
        _, curr_loss = sess.run([opt_op, loss], feed_dict=feed)
        epoch_loss += curr_loss

      losses[index][epoch - 1] = epoch_loss

    xs = create_feature_matrix(np.linspace(min(dataX[:, 0]), max(dataX[:, 0]), 100), degree)

    hyp_val = sess.run(hyp, feed_dict={X: xs})

    plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color=colors[index])

    legends.append("%.3f" % lambd if (lambd > 0) and (lambd < 1) else "%d" % lambd)

plt.legend(legends, loc=0)

plt.title('Ridge regression')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()

plt.xlabel('Epoch')
plt.ylabel('Loss function value')

plt.title('Loss over time')
ax = np.arange(0, nb_epochs)

for i in range(0, len(lambdas)):
  plt.plot(ax, losses[i], color=colors[i])

plt.legend(legends, loc=0)
plt.show()

# Mozemo da primetimo da sa dodavanjem sve veceg squared magnitude penalty-a
# dobijamo sve veci loss, tj. desava se under-fitting podataka,
# kada je on 0 dobijamo obicnu polinomijalnu regresivnu krivu.
