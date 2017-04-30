#!python35
#_*_ coding:utf-8 _*_
####################
#
# 卷积网络
#
####################
###Libraries
#Standard libraries
#Third Libraries
import tensorflow as tf
#载入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data/',one_hot = True)

#参数
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

#维度
n_input = 784
n_classes = 10
#不置0比例
dropout = 0.75

###计算图
#设置计算图输入数据格式
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

#卷积层
#x - 二维
#W - 共享权重
#b - 偏置
def conv2d(x, W, b, strides=1):
    #
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

#最大池化层
#x - 二维
#k - k*k像素块降为1*1
def maxpool2d(x, k= 2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides = [1,k,k,1],padding = 'SAME')

#卷积网络
def conv_net(x, weights,biases, dropout):
    #x(i)-[1,784]重构 [28,28,1]
    x = tf.reshape(x,shape = [-1, 28,28,1])
    #卷积层1
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1,k = 2)
    #卷积层2
    conv2 = conv2d(conv1, weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2, k = 2)
    #全链接层
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    #输出层
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']),biases['out'])
    return out

weights = {
    'wc1':tf.Variable(tf.random_normal([5, 5, 1,32])),
    'wc2':tf.Variable(tf.random_normal([5, 5,32,64])),
    'wd1':tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out':tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal([64])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

#构造模型
pred = conv_net(x, weights, biases, keep_prob)

#损失函数、优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)

#正确预估列向量
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#精确率
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
init = tf.global_variables_initializer()

# 运行计算图
with tf.Session() as sess:
    print("hehe")
    sess.run(init)
    step = 1
    # 训练，直到得到最大索引
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 运行优化器
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # 计算代价，准确率
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        accuracy.eval({x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob:1.}))
