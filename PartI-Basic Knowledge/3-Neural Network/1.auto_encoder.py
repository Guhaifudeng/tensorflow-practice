#!python35
# _*_ coding:utf-8 _*_
####################
#
# 自编码
#
####################
###Libraries
#Standard Libraries
#Third Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
#载入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/',one_hot = True)

#参数
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

#网络参数
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784

#tf Graph input
X = tf.placeholder('float',[None, n_input])

weights = {
    'encoder_h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2':tf.Variable(tf.random_normal([n_input]))
}

#构建编码器

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                biases['encoder_b2']))
    return layer_2

#构建解码器
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                biases['decoder_b2']))
    return layer_2

#构建模型
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
#预测
y_pred = decoder_op

#目标
y_true = X

#代价函数、RMS优化
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#初始化
init = tf.global_variables_initializer()

#运行计算图
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    #训练
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _ , c = sess.run([optimizer, cost], feed_dict = {X: batch_xs})

        if epoch % display_step == 0:
            print("Epoch:","%04d" % (epoch+1),
                'cost=',"{:.9f}".format(c))
    print('Optimization Finished!')

    #测试
    encode_decode = sess.run(
        y_pred, feed_dict = {X: mnist.test.images[:examples_to_show]})

    #对比重构和原有的图像
    f ,a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
