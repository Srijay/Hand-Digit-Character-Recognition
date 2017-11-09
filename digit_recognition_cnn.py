from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random

def batch_norm(x, n_out, phase_train):

    with tf.variable_scope('bn'):

        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),

                                     name='beta', trainable=True)

        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),

                                      name='gamma', trainable=True)

        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():

            ema_apply_op = ema.apply([batch_mean, batch_var])

            with tf.control_dependencies([ema_apply_op]):

                return tf.identity(batch_mean), tf.identity(batch_var)


        mean, var = tf.cond(phase_train,

                            mean_var_with_update,

                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    return normed

def cnn(x,phase_train):

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    batch_norm_conv2d_1 = batch_norm(conv2d(x_image, W_conv1), 32, phase_train)
    h_conv1 = tf.nn.relu(batch_norm_conv2d_1 + b_conv1)

  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    batch_norm_conv2d_2 = batch_norm(conv2d(h_pool1, W_conv2), 64, phase_train)
    h_conv2 = tf.nn.relu(batch_norm_conv2d_2 + b_conv2)

  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):

  trainDataImages = []
  trainDataLabels = []
  testImages = []

  with open("Data/train.csv", 'r') as trainfile:
    next(trainfile)
    for line in trainfile:
      data = line.split(",")
      label = int(data[0])
      image = map(float,data[1:])
      trainDataImages.append(image)
      trainDataLabels.append(label)

    trainDataImages = np.array(trainDataImages,dtype=np.float32)

    trainDataLabels = np.array(trainDataLabels)
    onehot = np.zeros((len(trainDataLabels),10))
    onehot[np.arange(len(trainDataLabels)),trainDataLabels] = 1
    trainDataLabels = np.array(onehot,dtype=np.float32)

    print("Train data is parsed")
    trainDataSize = len(trainDataLabels)
    trainDataIds = range(0,trainDataSize)
    random.shuffle(trainDataIds)

    print("Random split into train and test data")
    trainIds = trainDataIds[:int(trainDataSize*0.7)]
    devIds = trainDataIds[int(trainDataSize*0.7):]
    trainImages = [trainDataImages[k] for k in trainIds]
    devImages = [trainDataImages[k] for k in devIds]
    trainLabels = [trainDataLabels[k] for k in trainIds]
    devLabels = [trainDataLabels[k] for k in devIds]
    print("length of train data and dev data are " + str(len(trainIds)) + " and " + str(len(devIds)))

    with open("Data/test.csv", 'r') as testfile:
      next(testfile)
      for line in testfile:
        data = line.split(",")
        image = map(int, data)
        testImages.append(image)

    testImages = np.array(testImages, dtype=np.float32)
    print("Test data is parsed")

  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  phase_train = tf.placeholder(tf.bool,name='phase_train')

  y_conv, keep_prob = cnn(x,phase_train)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #training step to minimize cross entropy

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  with tf.name_scope('output'):
    testoutput = tf.argmax(y_conv,1)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    model_path = "./model.ckpt";
    Iterations = 20000
    dev_accuracy=0
    for i in range(Iterations):
      minibatchIds = random.sample(range(0,len(trainImages)),50)
      miniBatchImages = [trainImages[k] for k in minibatchIds]
      miniBatchLabels = [trainLabels[k] for k in minibatchIds]
      if i % 50 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: miniBatchImages, y_: miniBatchLabels, keep_prob: 1.0, phase_train: False})
        curr_dev_accuracy = accuracy.eval(feed_dict={
          x: devImages, y_: devLabels, keep_prob: 1.0, phase_train: False})
        if(curr_dev_accuracy>dev_accuracy):
          dev_accuracy = curr_dev_accuracy
          saver.save(sess,model_path)
          print('step %d, training accuracy %g' % (i, train_accuracy))
          print('step %d, validation accuracy %g' % (i, dev_accuracy))
      train_step.run(feed_dict={x: miniBatchImages, y_: miniBatchLabels, keep_prob: 0.5, phase_train: True})

    saver.restore(sess,model_path)
    print("Model restored")

    outputs = testoutput.eval(feed_dict={x:testImages,keep_prob:1.0,phase_train: False})

    outfile = open("output.csv",'w')
    outfile.write("ImageId,Label\n")
    id=1
    for output in outputs:
      outfile.write(str(id)+","+str(output)+"\n")
      id+=1

if __name__ == '__main__':
  tf.app.run(main=main)