'''
Description: Keras implementation of a variational autoencoder

    1. This VAE is built with MLP (multi-layer perceptron) and tested on MNIST.
    1. You can save the trained model, then restore it later to use directly.
    3. Visualization includes: latent space, 2D manifold, imputation.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
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
    epsilon = K.random_normal_variable(shape=(z_mean.get_shape()), mean=0.,
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

#%% Define (or restore) Models

# check if there exists previously trained models
savepath = "./save/"
if not os.path.exists(savepath): os.mkdir(savepath)
if os.path.exists(savepath+"vae.h5"): #change this line to "if False" to always train a new model
    vae       = load_model(savepath+"vae.h5", custom_objects = {'loss': loss, 'epsilon_std': epsilon_std})
    encoder   = load_model(savepath+"encoder.h5")
    generator = load_model(savepath+"generator.h5", custom_objects = {'epsilon_std': epsilon_std})
    print("\nModels trained for %d epochs successfully restored."%(np.load(savepath+"epochs.npy")))
    
else:
    vae = Model(x, x_hat)
    vae.compile(optimizer='rmsprop', loss=loss)
    ## train the VAE on MNIST digits
    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test))    
    vae.save(savepath+"vae.h5")
    
    ## build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    encoder.save(savepath+"encoder.h5")
    
    ## build a digit generator that can sample from the learned distribution
    z_sampled = Input(shape=(dim_latent,))
    x_decoded = decoder_output(decoder_hidden(z_sampled))
    generator = Model(z_sampled, x_decoded)
    generator.save(savepath+"generator.h5")
    
    ## document training info
    np.save(savepath+"epochs", epochs)
 
# you can summarize your model if necessary
vae.summary()

#%% Visualization: Latent Space

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
fig = plt.figure(figsize=(4, 4))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()
fig.savefig(savepath+"z_{}.png".format(use_loss))

#%% Visualization: 2D manifold of the digits

n = 15  # figure with 15x15 digits
m = 28
figure = np.zeros((m * n, m * n))
# linearly spaced coordinates on the unit square were transformed through 
# the inverse CDF (ppf) of the Gaussian to produce values of the latent 
# variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(m, m)
        figure[i * m: (i + 1) * m,
               j * m: (j + 1) * m] = digit

fig = plt.figure(figsize=(6, 6))
plt.imshow(figure, cmap='Greys_r')
plt.show()
fig.savefig(savepath+"x_{}.png".format(use_loss))

#%% Visualization: Data imputation

figure = np.zeros((m * 3, m * n))
x_origin = x_test[:batch_size,:]
x_corrupted = np.copy(x_origin)
x_corrupted[:, 300:400] = 0
x_reconstructed = vae.predict(x_corrupted, batch_size=batch_size).reshape((-1, m, m))
x_origin = x_origin.reshape((-1, m, m))
x_corrupted = x_corrupted.reshape((-1, m, m))
for i in range(n):
    figure[:m,      i*m:(i+1)*m] = x_origin[i]
    figure[ m:2*m,  i*m:(i+1)*m] = x_corrupted[i]
    figure[   2*m:, i*m:(i+1)*m] = x_reconstructed[i]

fig = plt.figure(figsize=(6, 3))
plt.imshow(figure, cmap='Greys_r')
plt.title('Image Imputation')
plt.xticks([])
plt.yticks(m*np.array([.5,1.5,2.5]),['Original','Corrupt','Re-con'])
fig.savefig(savepath+"i_{}.png".format(use_loss))
