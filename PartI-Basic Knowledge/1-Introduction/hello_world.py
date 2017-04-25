#!python35
from __future__ import print_function
import tensorflow as tf

#1.creat a constant op
#2.add the op as a node to the default graph
#3.value returned by the constructor
hello = tf.constant('Hello, TensorFlow!')

#Start tf session
sess = tf.Session()

#run the op
print(sess.run(hello))
