import tensorflow as tf
import numpy as np
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONFIGURATION
tf.reset_default_graph()
n_embedding, n_voc, n_len = 32, 10000, 256
n_hidden, n_attention = 64, 64
lr = 1E-2
n_iterations, batch_size, print_every = 2**14 , 2**8, 2**6

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DATA
# taking care of dataset processing
(train, label_train), (test, label_test) = tf.keras.datasets.reuters.load_data(num_words=n_voc)
test_size = 256
test = test[:test_size]
label_test = label_test[:test_size]
# data utils, could be done with tf, more flexible
def one_hot(label_train, label_test):
  classes = np.max(label_train) + 1
  targets = np.array([label_train]).reshape(-1)
  label_train = np.eye(classes)[targets]
  targets = np.array([label_test]).reshape(-1)
  label_test = np.eye(classes)[targets]
  return label_train, label_test, classes

# bit of cheating, using keras, subject to be updated as an active learning process
def pad_seq(train, test, max_len=256, padding="post", truncating="post"):
  train = tf.keras.preprocessing.sequence.pad_sequences(train, 
                                                        maxlen=max_len, 
                                                        padding=padding, 
                                                        truncating=truncating)
  test = tf.keras.preprocessing.sequence.pad_sequences(test, 
                                                        maxlen=max_len, 
                                                        padding=padding, 
                                                        truncating=truncating)
  return train, test

train, test = pad_seq(train, test, n_len)
label_train, label_test, n_classes = one_hot(label_train, label_test)
dtype = tf.float32

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MODEL

def model(X):
  # embedding of words
  embedding = tf.Variable(tf.random_uniform([n_voc, n_embedding], -1.0, 1.0))
  X_e = tf.nn.embedding_lookup(embedding, X)
  
  # put in GRU https://i.pinimg.com/originals/3a/b6/06/3ab606bb2f3cb966821704cc47724181.jpg
  #bidirectional
  fw, bw = tf.nn.rnn_cell.GRUCell(n_hidden), tf.nn.rnn_cell.GRUCell(n_hidden)
  rec, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, X_e, dtype=tf.float32)

  conv1 = conv2D_attention_layer(rec)
  flat = tf.layers.flatten(conv1)
  y = tf.layers.dense(flat, n_classes, activation=tf.nn.softmax)

  return y

def conv1D_attention_layer(bi_rnn_in):
  bi_rnn_in = tf.concat(bi_rnn_in, 2)
  
  # reduction vector
  C = tf.Variable(tf.truncated_normal([n_attention]))
  
  # convolution
  conv1 = tf.layers.conv1d(bi_rnn_in, n_attention, [5], padding="same")
  # activation seperated, in case of batch norm this could be beneficial
  act = tf.nn.tanh(conv1)
  #reduce
  red = tf.tensordot(act, C, axes=1) 
  weight = tf.nn.softmax(red)
  
  # weighted reduction
  y = tf.reduce_sum(bi_rnn_in * tf.expand_dims(weight, -1), 1)
  return y


def conv2D_attention_layer(bi_rnn_in):
  bi_rnn_in = tf.concat(bi_rnn_in, 2)

  # reduction vector
  C = tf.Variable(tf.truncated_normal([n_attention]))
  bi_rnn_in = tf.expand_dims(bi_rnn_in, axis=-1)
  
  # convolution
  conv1 = tf.layers.conv2d(bi_rnn_in, 32, [3,3], padding="valid")
  avg1 = tf.layers.average_pooling2d(conv1, [2,2], 1)
  conv2 = tf.layers.conv2d(avg1, 32, [2,2], padding="same")
  avg2 = tf.layers.average_pooling2d(conv2, [2,2], 1)
  y = tf.reduce_mean(avg2, axis=-1)
  conv3d1 = tf.layers.conv1d(y, n_attention, [5], padding="same")
  # activation seperated, in case of batch norm this could be beneficial
  act = tf.nn.tanh(conv3d1)
  
  #reduce
  red = tf.tensordot(act, C, axes=1)
  weight = tf.nn.softmax(red)
  
  # weighted reduction
  y = tf.reduce_sum(tf.squeeze(bi_rnn_in, axis=-1) * tf.expand_dims(weight, -1), 1)
  return y


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# TRAINING & TESTING
def loss_f(Y, y):
  return tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y + 1E-10), reduction_indices=[1]))

def optimizer(loss):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  opt = tf.train.AdamOptimizer(lr).minimize(loss)
  return tf.group([opt, update_ops])

def train_test():
  with tf.Session() as sess:
      sess.run(init)
      for iteration in range(n_iterations):
          indices = np.random.choice(len(train), batch_size)
          batch_X, batch_Y = train[indices], label_train[indices]
          feed_dict = {X: batch_X, Y: batch_Y}
          _, lo = sess.run([step, loss], feed_dict=feed_dict)
          
          if iteration % print_every == 0:
            print("loss", lo)
            acc = sess.run(accuracy, feed_dict={X: test, Y: label_test})
            print("acc ", acc)
            if acc > 0.9:
              break
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GOALS
X = tf.placeholder(tf.int32, [None, n_len])
Y = tf.placeholder(dtype, [None, n_classes])
y = model(X)              
              
loss = loss_f(Y,y)
step = optimizer(loss)
hits = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))
init = tf.global_variables_initializer()
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MAIN
train_test()



