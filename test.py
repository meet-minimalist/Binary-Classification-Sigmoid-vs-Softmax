# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:52:47 2021

@author: Meet
"""

import numpy as np
import tensorflow as tf
np.random.seed(42)

def softmax(x):
    # x : [B x 2]
    
    numer = np.exp(x)
    denom = np.sum(np.exp(x), axis=1, keepdims=True) + 1e-6

    softmax_val = numer / denom
    
    return softmax_val

def sigmoid(x):
    # x : [B x 1]
    
    sigmoid_val = 1 / (1 + np.exp(-x))
    
    return sigmoid_val


B = 32              # batch_size
input_shape = 10
x = np.float32(np.random.randn(B, input_shape))
label = np.float32(np.random.randint(low=0, high=2, size=B))

# Case - 1  : 10 valued input to softmax layer
positive_class_index = 0
output_shape_softmax = 2        # 2 classes for binary classification
w_softmax = np.random.normal(size=[input_shape, output_shape_softmax])
z_softmax_logits = np.matmul(x, np.float32(w_softmax))
z_softmax = softmax(z_softmax_logits)
print("Softmax output shape: ", z_softmax.shape)


# Case - 2 : Softmax in the form of Sigmoid
if positive_class_index == 0:
    # if 1nd index is positive class in softmax example, then resultant sigmoid counter part will fetch positive class probability only
    w_sigmoid =  w_softmax[:, 0:1] - w_softmax[:, 1:2]          
else:
    # if 2nd index is positive class in softmax example, then resultant sigmoid counter part will fetch positive class probability only
    w_sigmoid =  - w_softmax[:, 0:1] + w_softmax[:, 1:2]        
    
z_sigmoid_logits = np.matmul(x, np.float32(w_sigmoid))
z_sigmoid = sigmoid(z_sigmoid_logits)
print("Sigmoid output shape: ", z_sigmoid.shape)

print("Softmax output first 2 node output: ", z_softmax[:2])
print("Sigmoid output first 2 node output: ", z_sigmoid[:2])

if z_softmax[:, positive_class_index:positive_class_index+1].all() == z_sigmoid.all():
    print("Sigmoid representation of Softmax is same.")
    print("Weights 10x2 of softmax can be represented by two vectors of 10x1 shape : w0 and w1.")
    print("Weights 10x2 of softmax is squished into weights 10x1 in sigmoid counterpart.")
    print("Corresponding 10x1 weight of sigmoid representation can be found by w_sigmoid = w0 - w1 or w1 - w0.")




print("\n\n")
print("Comparing loss of both the cases")
label_softmax = np.stack([label, 1 - label], axis=1)
softmax_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_softmax, logits=z_softmax_logits))
sigmoid_ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=np.float32(label[:, np.newaxis]), logits=z_sigmoid_logits))

numpy_sigmoid_ce_loss = np.mean((-1) * label * np.log(np.squeeze(z_sigmoid)) + (-1) * (1-label) * np.log(1 - np.squeeze(z_sigmoid)))
numpy_softmax_ce_loss = np.mean(np.sum((-1) * label_softmax * np.log(z_softmax), axis=1))

with tf.Session() as sess:
    softmax_ce_loss, sigmoid_ce_loss = sess.run([softmax_ce, sigmoid_ce])
    print("Case-1:[NPY] Softmax cross entropy loss: ", numpy_softmax_ce_loss)
    print("Case-2:[NPY] Sigmoid cross entropy loss: ", numpy_sigmoid_ce_loss)
    print("Case-1:[TF] Softmax cross entropy loss: ", softmax_ce_loss)
    print("Case-1:[TF] Sigmoid cross entropy loss: ", sigmoid_ce_loss)
    
    if round(sigmoid_ce_loss, 4) == round(softmax_ce_loss, 4):
        print("Loss value is matching upto 4 digits")
    