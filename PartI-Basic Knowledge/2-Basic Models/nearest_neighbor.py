#!python35
#_*_ coding:utf-8 _*_
###Libraries
#Standard Library

#Third Library
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data/',one_hot = True)

#样本数据
Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)
#print(Ytr.shape) (50000,10)
#tf Graph Input
xtr = tf.placeholder('float', [None ,784]) #数据集
xte = tf.placeholder('float', [784])       #测试集中的某一点x

###k-邻近算法
#距离：L1范数
#reduction_indices = 1 表示行求和
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices = 1)
#最小距离索引
# 0 - 按列取最小值坐标
pred = tf.arg_min(distance, 0)
#精确度
accuracy = 0

#初始化变量
init = tf.global_variables_initializer()

#运行计算图
with tf.Session() as sess:
    sess.run(init)

    #遍历测试数据
    for i in range(len(Xte)):
        nn_index = sess.run(pred , feed_dict = {xtr:Xtr,xte: Xte[i,:]})
        print(nn_index)
        #根据最近点索引得到
        print('Test',i,"pred:",np.argmax(Ytr[nn_index]),'true class:',np.argmax(Yte[i]))

        if np.argmax(Ytr[nn_index]) ==np.argmax(Yte[i]):
            accuracy += 1
    print('Accuracy:',accuracy/len(Xte))
