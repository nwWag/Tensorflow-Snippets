import matplotlib; matplotlib.use("GTK3Agg")

import numpy as np
import array as ar
import pylab as pl
import tensorflow as tf
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# taking care of dataset and datatype
(train, label_train), (test, label_test) = tf.keras.datasets.mnist.load_data()

num_test, num_train = 10000, 60000
test, label_test = test[:num_test], label_test[:num_test]
train, label_train = train[:num_train], label_train[:num_train]

test = test.astype(np.float32)/255
train = train.astype(np.float32)/255

dtype, inp_shape, out_shape = tf.float32, [None, 28, 28, 1], [None, 28, 28, 1]

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# building res-like decoding steps

def decode(net, scaling, training):
  # scale
  net = tf.image.resize_nearest_neighbor(net,[net.shape[1] * scaling, net.shape[2] * scaling])
  
  # weighted reduction and bypass
  a = tf.layers.conv2d(
        inputs=net,
        filters=1,
        kernel_size=[1, 1],
        padding="same",
        activation=None)
  
  
  # conv
  y = tf.layers.conv2d(
        inputs=a,
        filters=16,
        kernel_size=[5, 5],
        padding="same",
        activation=None)
  
  # normalize
  y = tf.layers.batch_normalization(y, training = training)

  # stack all together
  net = tf.concat([y,a], axis = 3)
  
  # avoid overfitting
  #net = tf.layers.dropout(net)
  
  return net

def encode(net, scaling, training):
  # scale
  net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=scaling)
  
  # weighted reduction and bypass
  a = tf.layers.conv2d(
        inputs=net,
        filters=1,
        kernel_size=[1, 1],
        padding="same",
        activation=None)
  
  
  # conv
  y = tf.layers.conv2d(
        inputs=a,
        filters=16,
        kernel_size=[5, 5],
        padding="same",
        activation=None)
 
  
  # normalize
  y = tf.layers.batch_normalization(y, training = training)
  
  # stack all together
  net = tf.concat([y,a], axis = 3)
  
  # avoid overfitting
  #net = tf.layers.dropout(net)
  
  return net

def reduce_activate(net, num_filter):
  # weighted reduction with activation
  net = tf.layers.conv2d(
        inputs=net,
        filters=num_filter,
        kernel_size=[1, 1],
        padding="same",
        activation=tf.nn.sigmoid)
  return net

def decoding_block(net, num_filter = 1, scaling = 4, training = True):
  return reduce_activate(decode(net, scaling, training), num_filter)

def encoding_block(net, num_filter = 1, scaling = 4, training = True):
  return reduce_activate(encode(net, scaling, training), num_filter)
  
  

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# building network and goals
tf.reset_default_graph()
phase = tf.placeholder(tf.bool, name='phase') 
X = tf.placeholder(dtype=dtype, shape=inp_shape)
Y = tf.placeholder(dtype=dtype, shape=out_shape)

net = decoding_block(decoding_block(encoding_block(encoding_block(
    tf.identity(X), 5, 2, phase), 3, 2, phase), 5, 2, phase), 1, 2, phase)

loss = -tf.reduce_mean(Y*tf.log(net)+(1-Y)*tf.log(1-net))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
step = tf.train.AdamOptimizer(1E-2).minimize(loss)
step = tf.group([step, update_ops])

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# building session and perform training

# hyperparameter
num_iterations, print_every, batch_size = 2**16, 2**14, 2**6

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    counter = 0;
    for iteration in range(num_iterations):
        indices = np.random.choice(len(train), batch_size)
        batch_X = np.expand_dims(train[indices], 3)
        batch_Y = np.expand_dims(train[indices], 3)
        feed_dict = {X: batch_X, Y: batch_Y, phase: True}
    
        _, lo = sess.run([step, loss], feed_dict=feed_dict)
        if iteration % print_every == print_every-1:
          print("Round ", counter)
          counter += 1
          batch_X = np.expand_dims(test[:4], 3)
          batch_Y = sess.run(net, feed_dict={X: batch_X, phase: False})

          cmap = "viridis"
          pl.subplot(241)
          pl.imshow(batch_X[0,:,:,0], cmap = cmap)
          pl.subplot(242)
          pl.imshow(batch_Y[0,:,:,0], cmap = cmap)
          pl.subplot(243)
          pl.imshow(batch_X[1,:,:,0], cmap = cmap)
          pl.subplot(244)
          pl.imshow(batch_Y[1,:,:,0], cmap = cmap)
          pl.subplot(245)
          pl.imshow(batch_X[2,:,:,0], cmap = cmap)
          pl.subplot(246)
          pl.imshow(batch_Y[2,:,:,0], cmap = cmap)
          pl.subplot(247)
          pl.imshow(batch_X[3,:,:,0], cmap = cmap)
          pl.subplot(248)
          pl.imshow(batch_Y[3,:,:,0], cmap = cmap)
          pl.show()
