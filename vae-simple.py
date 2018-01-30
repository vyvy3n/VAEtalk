'''
Description: to build a variational autoencoder with Keras.
Reference  : "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras import objectives
import tensorflow as tf

#%% Load data 

## by Keras
#from keras.datasets import mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
# by tensorflow
from tensorflow.examples.tutorials.mnist import input_data
#or manually download from: http://yann.lecun.com/exdb/mnist/
mnist = input_data.read_data_sets('MNIST')
#mnist = input_data.read_data_sets("/home/vivian/VAE/MNIST", one_hot=True)
x_train = mnist.train.images
x_test  = mnist.test.images
y_test  = mnist.test.labels

#%% Build VAE

np.random.seed(0)  #for reproducibility
            
dim_x       = 784
dim_latent  = 2 
dim_hidden  = 256
batch_size  = 100 
epochs      = 50
decay       = 1e-4 # L2 regularization
epsilon_std = 1.0
use_loss    = 'xent' # 'mse' or 'xent'

use_bias    = True

## Encoder
x = Input(batch_shape=(batch_size, dim_x))

h_encoded = Dense(dim_hidden, 
                  kernel_regularizer=l2(decay), bias_regularizer=l2(decay), 
                  use_bias=use_bias, activation='tanh')(x)
z_mean    = Dense(dim_latent, 
                  kernel_regularizer=l2(decay), bias_regularizer=l2(decay), 
                  use_bias=use_bias)(h_encoded)
z_log_var = Dense(dim_latent, 
                  kernel_regularizer=l2(decay), bias_regularizer=l2(decay), 
                  use_bias=use_bias)(h_encoded)

## Sampler
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal_variable(shape=(batch_size, dim_latent), mean=0.,
                                       scale=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(dim_latent,))([z_mean, z_log_var])

## Decoder
decoder_hidden = Dense(dim_hidden, 
                     kernel_regularizer=l2(decay), bias_regularizer=l2(decay), 
                     use_bias=use_bias, activation='tanh')
decoder_output = Dense(dim_x, 
                     kernel_regularizer=l2(decay), bias_regularizer=l2(decay), 
                     use_bias=use_bias, activation='sigmoid')
x_hat          = decoder_output(decoder_hidden(z))

                
## Loss
def loss(x, x_hat):
    loss_kl   = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    loss_xent = dim_x * objectives.binary_crossentropy(x, x_hat)
    loss_mse  = dim_x * objectives.mse(x, x_hat) 
    if use_loss == 'xent':
        return loss_kl + loss_xent
    elif use_loss == 'mse':
        return loss_kl + loss_mse
    else:
        raise Exception('Undefined Loss: %s'%(use_loss))

## define Model
vae = Model(x, x_hat)
vae.compile(optimizer='rmsprop', loss=loss)
## train the VAE on MNIST digits
vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

#%% Visualization: Latent Space

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
fig = plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()
fig.savefig('z_{}.png'.format(use_loss))

#%% Visualization: 

## build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(dim_latent,))
_x_hat = decoder_output(decoder_hidden(decoder_input))
generator = Model(decoder_input, _x_hat)


#%%
# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

fig = plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
fig.savefig('x_{}.png'.format(use_loss))

# data imputation
figure = np.zeros((digit_size * 3, digit_size * n))
x = x_test[:batch_size,:]
x_corupted = np.copy(x)
x_corupted[:, 300:400] = 0
x_encoded = vae.predict(x_corupted, batch_size=batch_size).reshape((-1, digit_size, digit_size))
x = x.reshape((-1, digit_size, digit_size))
x_corupted = x_corupted.reshape((-1, digit_size, digit_size))
for i in range(n):
    xi = x[i]
    xi_c = x_corupted[i]
    xi_e = x_encoded[i]
    figure[:digit_size, i * digit_size:(i+1)*digit_size] = xi
    figure[digit_size:2 * digit_size, i * digit_size:(i+1)*digit_size] = xi_c
    figure[2 * digit_size:, i * digit_size:(i+1)*digit_size] = xi_e

fig = plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
fig.savefig('i_{}.png'.format(use_loss))