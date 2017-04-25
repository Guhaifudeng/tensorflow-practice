#!python35
from __future__ import print_function
import tensorflow as tf
### basic constant operations
a = tf.constant(4)
b = tf.constant(6)

with tf.Session() as sess:
    print("a = 4, b = 6")
    print("additions with constants: %i",sess.run(a+b))
    print("multiplication with constants: %i",sess.run(a*b))

#basic operations with variable as graph input
#tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

#define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)
# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

###matric operations
X = tf.constant([[3,3]])   #1x2 [3,3] is 2 without []
Y = tf.constant([[2],[1]])#2x1

product = tf.matmul(X, Y)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
