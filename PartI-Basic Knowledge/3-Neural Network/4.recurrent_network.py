#!python35
#_*_ coding:utf-8 _*_
####################
#
# 循环神经网络 LSTM
#
####################
#Libraries
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
#load mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data/', one_hot =True)
#parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10
#network parameters
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

#tf Graph input
x = tf.placeholder('float',[None, n_steps,n_input])
y = tf.placeholder('float',[None, n_classes])

#权重
weights = {
    'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))
}
#偏置
biases = {
    'out':tf.Variable(tf.random_normal([n_classes]))
}
def RNN(x, weights, biases):
    #重构数据shape以符合rnn函数参数要求
    #[batchsize,n_steps,n_input] to nesteps [batch_size,n_input]
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    #
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    #lstm cell
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    #get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype = tf.float32)
    #

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
print("heh")
#loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
#initializing
init = tf.global_variables_initializer()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
