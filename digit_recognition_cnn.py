from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random

def cnn(x):

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

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

  trainImages = []
  trainLabels = []
  testImages = []

  with open("Data/train.csv", 'r') as trainfile:
    next(trainfile)
    for line in trainfile:
      data = line.split(",")
      label = int(data[0])
      image = map(float,data[1:])
      trainImages.append(image)
      trainLabels.append(label)

    trainImages = np.array(trainImages,dtype=np.float32)

    trainLabels = np.array(trainLabels)
    onehot = np.zeros((len(trainLabels),10))
    onehot[np.arange(len(trainLabels)),trainLabels] = 1
    trainLabels = np.array(onehot,dtype=np.float32)
    print("Train data is parsed")
    trainDataSize = len(trainImages)

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

  y_conv, keep_prob = cnn(x)

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
    sess.run(tf.global_variables_initializer())
    print("Hello")
    saver = tf.train.Saver()
    print("By")
    model_path = "./model.ckpt";
    Iterations = 0
    for i in range(Iterations):
      minibatchIds = random.sample(range(0,trainDataSize),50)
      miniBatchImages = [trainImages[k] for k in minibatchIds]
      miniBatchLabels = [trainLabels[k] for k in minibatchIds]
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: miniBatchImages, y_: miniBatchLabels, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        saver.save(sess,model_path)
      train_step.run(feed_dict={x: miniBatchImages, y_: miniBatchLabels, keep_prob: 0.5})

    saver.restore(sess,model_path)
    print("Model restored")

    outputs = testoutput.eval(feed_dict={x:testImages,keep_prob:1.0})

    outfile = open("output.csv",'w')
    outfile.write("ImageId,Label\n")
    id=1
    for output in outputs:
      outfile.write(str(id)+","+str(output)+"\n")
      id+=1

if __name__ == '__main__':
  tf.app.run(main=main)