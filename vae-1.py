#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:20:02 2018

@author: vivian
"""

from __future__ import division
from __future__ import print_function
import os.path

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST')

input_dim = 784
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
lam = 0

# Weights and Bias initializer
# —————————————————————————————————————————————————————————————————————————————
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

# Batch Normalization
# —————————————————————————————————————————————————————————————————————————————
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

def batch_norm_layer(x,train_phase,scope_bn='bn'):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
                          updates_collections=None,
                          is_training=True,
                          reuse=None, # is this right?
                          trainable=True,
                          scope=scope_bn)
    bn_test = batch_norm(x, decay=0.999, center=True, scale=True,
                         updates_collections=None,
                         is_training=False,
                         reuse=True, # is this right?
                         trainable=True,
                         scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_test)
    return z

#%% VAE
class VAE(object):
    '''
    keep_porb=the probability of a node being kept in this layer
    '''
    def __init__(self, x,  input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, 
                 keep_prob=1, problem='regression', BN=False, training = None, epsl = 1e-3, lam = 0):

        self.l2_loss = tf.constant(0.0)

        W_encoder_input_hidden = weight_variable([input_dim,hidden_encoder_dim])
        b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
        self.l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)
        
        # Hidden layer encoder
        hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)
        # Hidden layer encoder Batch-norm
        if BN: 
            scale_en = tf.Variable(tf.ones([hidden_encoder_dim]))
            shift_en = tf.Variable(tf.zeros([hidden_encoder_dim]))
            mean_en, var_en = tf.nn.moments(hidden_encoder, [0])
            hidden_encoder = tf.nn.batch_normalization(hidden_encoder,
                                  mean_en, var_en, shift_en, scale_en, epsl)
        # Hidden layer encoder activation function 'relu'
        hidden_encoder = tf.nn.relu(hidden_encoder)
        # Hidden layer encoder Dropout
        hidden_encoder_drop = tf.nn.dropout(hidden_encoder, keep_prob)
        
        W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim])
        b_encoder_hidden_mu = bias_variable([latent_dim])
        self.l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)
        
        # Mu encoder
        mu_encoder = tf.matmul(hidden_encoder_drop, W_encoder_hidden_mu) + b_encoder_hidden_mu
        
        W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim])
        b_encoder_hidden_logvar = bias_variable([latent_dim])
        self.l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)
        
        # Sigma encoder
        logvar_encoder = tf.matmul(hidden_encoder_drop, W_encoder_hidden_logvar) + b_encoder_hidden_logvar
        
        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')
        
        # Sample latent variable
        std_encoder = tf.exp(0.5 * logvar_encoder)
        self.z = mu_encoder + tf.multiply(std_encoder, epsilon)
        
        W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim])
        b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
        self.l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)
        
        # Hidden layer decoder
        hidden_decoder = tf.nn.relu(tf.matmul(self.z, W_decoder_z_hidden) + b_decoder_z_hidden)
        # Hidden layer decoder Batch-norm
        if BN:
            scale_de = tf.Variable(tf.ones([hidden_decoder_dim]))
            shift_de = tf.Variable(tf.zeros([hidden_decoder_dim]))
            mean_de, var_de = tf.nn.moments(hidden_decoder, [0])
            hidden_decoder = tf.nn.batch_normalization(hidden_decoder,
                                  mean_de, var_de, shift_de, scale_de, epsl)
        hidden_decoder = tf.nn.relu(hidden_decoder)
        # Hidden layer decoder Dropout
        hidden_decoder_drop = tf.nn.dropout(hidden_decoder, keep_prob)
        
        W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
        b_decoder_hidden_reconstruction = bias_variable([input_dim])
        self.l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)
        
        KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2)\
                                   - tf.exp(logvar_encoder), reduction_indices=1)
        
        self.pred = tf.matmul(hidden_decoder_drop, W_decoder_hidden_reconstruction) \
                    + b_decoder_hidden_reconstruction
        if problem == "classification":
            BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.pred, labels=self.x_weighted), reduction_indices=1)
        if problem == "regression":
            BCE = tf.sqrt(tf.nn.l2_loss(self.pred-x)*2)
        
        self.loss = tf.reduce_mean(BCE + KLD)
        
        self.regularized_loss = self.loss + lam * self.l2_loss        
        self.train_step = tf.train.AdamOptimizer(0.01).minimize(self.regularized_loss)

#%% Training

# add op for merging summary
summary_op = tf.summary.merge_all()

# add Saver ops
saver = tf.train.Saver()

n_steps = int(1e6)
batch_size = 100

with tf.Session() as sess:
  summary_writer = tf.summary.FileWriter('experiment',
                                          graph=sess.graph)
  if os.path.isfile("save/model.ckpt"):
    print("Restoring saved parameters")
    saver.restore(sess, "save/model.ckpt")
  else:
    print("Initializing parameters")
    sess.run(tf.global_variables_initializer())

    x = tf.placeholder("float", [None, batch_size], 'input')
    y = tf.placeholder("float", [None, batch_size], 'input')
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')
    vae = VAE(x, batch_size, 
                    hidden_encoder_dim=int(0.5* batch_size), hidden_decoder_dim=int(0.5* batch_size), 
                    latent_dim=int(0.1* batch_size), keep_prob=keep_prob, BN=True)
        
  for step in range(1, n_steps):
    batch = mnist.train.next_batch(batch_size)
    feed_dict = {x: batch[0]}
    _, cur_loss, summary_str = sess.run([vae.train_step, vae.loss, summary_op], feed_dict=feed_dict)