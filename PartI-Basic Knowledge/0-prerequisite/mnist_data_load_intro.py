#!python35
#import MNIST
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot = True)

#load data
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

print("X_train:",np.shape(X_train),"Y_train",np.shape(Y_test))
print("X_test:",np.shape(X_test),"Y_test:",np.shape(Y_test))

#get the next 64 images and labels
#random sampling
batch_X,batch_Y = mnist.train.next_batch(64)
