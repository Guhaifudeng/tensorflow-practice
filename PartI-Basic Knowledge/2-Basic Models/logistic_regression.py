#!python35
#_*_ coding:utf-8 _*_
###Libraries
#Standard Library

#Third Library
import numpy as np
import tensorflow as tf
#载入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data/',one_hot = True)
#训练参数：学习速率、迭代次数、小样本集
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
###设计计算图
#计算图的Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# 初始化模型参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#逻辑斯谛模型
pred = tf.nn.softmax(tf.matmul(x,W) + b)

#交叉熵代价函数
cost = tf.reduce_mean(- tf.reduce_sum(y * tf.log(pred), reduction_indices = 1))

#梯度下降
#(根据cost、x确定是随机梯度下降还是在线梯度下降还是梯度下降)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#初始化参数
init = tf.global_variables_initializer()

###运行计算图
with tf.Session() as sess:
    sess.run(init)#去掉会怎样
    #训练
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        #遍历所有样本集

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #print(batch_xs.shape,batch_ys.shape)
            _, c = sess.run([optimizer, cost], feed_dict = {x:batch_xs, y:batch_ys})
            #计算平均损失
            avg_cost += c / total_batch

        if(epoch +1 ) % display_step == 0:
            print('Epoch','%04d' % (epoch+1),'cost=','{:.9f}'.format(avg_cost))

    print('Optimization Finished!')

    #测试模型
    # argmax(*, 1) 行选取最大值索引
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    #计算准确度
    #cast 转换张量数值类型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
