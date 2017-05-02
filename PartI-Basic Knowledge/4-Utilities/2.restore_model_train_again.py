#!python35
#_*_ coding:utf-8 _*_
####################
#
# 多层感知机:载入再训练
#
####################
#Libraries
#Standard Libraries
#Third Libraries
import tensorflow as tf

#load data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/', one_hot= True)

#parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
#设置model存储路径
model_path = 'data/model.ckpt'


#network parameters
n_hidden_1 = 512
n_hidden_2 = 256
n_input = 784
n_classes = 10

#tf Graph input
x = tf.placeholder('float',[None, n_input])
y = tf.placeholder('float',[None, n_classes])

#creat model
def  multilayer_perceptron(x, weights, biases, keep_probs):
    #隐藏层1:ReLU\dropout
    layer_1 = tf.add(tf.matmul(x, weights['h1']),biases['h1'] )
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_probs['h1'])
    #隐藏层1:ReLU\dropout
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']), biases['h2'] )
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_probs['h2'])
    #输出层
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer

#设置weight、bias、dropout
weights = {
    'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'h1':tf.Variable(tf.random_normal([n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

keep_probs ={
    'h1':0.75,
    'h2':0.95
}

#构建模型
pred = multilayer_perceptron(x, weights, biases,keep_probs)
#定义cost 和optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
#初始化参数
init = tf.global_variables_initializer()


#'saver' op to save and restore all the variabels
saver = tf.train.Saver()

#running a new session
#test
with tf.Session() as sess:
    sess.run(init)
    load_path = saver.restore(sess,model_path)
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("run1:Accuracy:", accuracy.eval(
        {x: mnist.test.images, y: mnist.test.labels}))

#运行计算图
with tf.Session() as sess:
    sess.run(init)
    load_path = saver.restore(sess,model_path)
    # Training cycle
    for epoch in range(15):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("run2:Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


