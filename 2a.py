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

np.set_printoptions(suppress=True, precision=5)

data = dict()

data['x'], data['y'] = np.loadtxt(filename, delimiter=',', unpack=True,
                                  skiprows=0, usecols=(0, 1))

filename = '/content/corona.csv'

nb_samples = len(data['x'])
nb_epochs = 100

degrees = np.arange(1,7)

colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']

legends = []

data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

indices = np.random.permutation(nb_samples)

data['x'] = (data['x'] - np.mean(data['x'])) / np.std(data['x'])
data['y'] = (data['y'] - np.mean(data['x'])) / np.std(data['y'])

plt.scatter(data['x'], data['y'])

losses = np.ones(shape=(max(degrees), nb_epochs))

for degree in degrees:

  dataX = create_feature_matrix(data['x'], degree)

  X = tf.placeholder(shape=(None, degree), dtype=tf.float32)
  Y = tf.placeholder(shape=(None), dtype=tf.float32)
  w = tf.Variable(tf.zeros(degree))
  bias = tf.Variable(0.0)

  w_col = tf.reshape(w, (degree, 1))
  hyp = tf.add(tf.matmul(X, w_col), bias)

  Y_col = tf.reshape(Y, (-1, 1))

  mse = tf.reduce_mean(tf.square(hyp - Y_col))

  opt_op = tf.train.AdamOptimizer().minimize(mse)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(nb_epochs + 1):

      epoch_loss = 0
      for i in range(nb_samples):
        feed = {X: dataX[i].reshape((1, degree)),
                Y: data['y'][i]}
        _, curr_loss = sess.run([opt_op, mse], feed_dict=feed)
        epoch_loss += curr_loss

      losses[degree - 1][epoch - 1] = epoch_loss

    xs = create_feature_matrix(np.linspace(min(dataX[:, 0]), max(dataX[:, 0]), 100), degree)

    hyp_val = sess.run(hyp, feed_dict={X: xs})

    plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color=colors[degree - 1])

    legends.append("degree %d" % degree)

plt.legend(legends, loc=0)

plt.title('Polinomyal regression ')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()

plt.xlabel('Epoch')
plt.ylabel('Mean squared error')

plt.title('Mean squared error over time')
ax = np.arange(0, nb_epochs);

for degree in degrees:
  plt.plot(ax, losses[degree - 1], color=colors[degree - 1])

plt.legend(legends, loc=0)
plt.show()

# Mozemo da primetimo da regresivne krive nizeg stepena (1, 2) imaju visoki bias
# tj. ne mogu u potpunosti ili uopste da opisu ulazne podatke.
# Dok u drugu ruku regresivne krive veceg stepena (3, 4, 5, 6) imaju nizak bias
# kao sto mozemo da vidimo na grafiku Mean sqared error funkcije ali istovremeno
# nastaje problem overfitting-a.
