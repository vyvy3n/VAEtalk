'''
Description: to build a variational autoencoder with Keras.
Reference  : "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import tensorflow as tf
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras import objectives
from keras.models import load_model

#%% Load data 

## by Keras
#from keras.datasets import mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
# by tensorflow
from tensorflow.examples.tutorials.mnist import input_data
#or manually download from: http://yann.lecun.com/exdb/mnist/
mnist = input_data.read_data_sets('./MNIST')
#mnist = input_data.read_data_sets("./MNIST", one_hot=True)
x_train = mnist.train.images
x_test  = mnist.test.images
y_test  = mnist.test.labels

#%% Parameters

np.random.seed(0)     #for reproducibility

dim_x       = x_train.shape[1]
dim_latent  = 2
dim_hidden  = 256
batch_size  = 100 
epochs      = 50
decay       = 1e-4    # L2 regularization
epsilon_std = 1.0
use_loss    = 'xent'  # 'mse'(mean square error) or 'xent'(cross entropy)
use_bias    = True
plot_on     = True

##for shell only
#____________________________________________________________
if __name__ == '__main__':
    import sys
    try:
        params = sys.argv[sys.argv.index('vae.py')+1:]
        try:    
            dim_latent = int(params[0])
            plot_on    = False #no plotting when run from the terminal
        except: pass
        try:    dim_hidden = int(params[1]) 
        except: pass
    except:
        pass
##for i in {2,3,5,10,15,20}; do python vae.py $i; done
#____________________________________________________________

#%% Build VAE

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

def loss_kl(x, x_hat):
    return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

#%% Train the VAE on MNIST Digits || or Load VAE

#modelpath = './save/vae_'+str(dim_latent)
#
#try: 
#    from keras.models import model_from_json
#    # load json and create model
#    json_file = open(modelpath+".json", "r")
#    loaded_model_json = json_file.read()
#    json_file.close()
#    vae = model_from_json(loaded_model_json)
#    # load weights into new model
#    vae.load_weights(modelpath+".h5")
#    print("Loaded model from disk")

## Define Model
vae = Model(x, x_hat)
vae.compile(optimizer='rmsprop', loss=loss, metrics=[loss_kl])
#vae.compile(optimizer='rmsprop', loss=loss)

#(optional setting)
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#patience: number of epochs with no improvement after which training will be stopped.

#timing the training process
from datetime import datetime
t0 = datetime.now()#timing(start)

#(optional) training logs monitor
from keras.callbacks import TensorBoard
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)  
#to open Tensorboard, direct to the directory of this script in the terminal 
#then run: Tensorboard --logdir=logs
#then visit: http://localhost:6006 in your browser.

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test),#validation_split=0.2,
        callbacks=[tb])#callbacks=[early_stopping, tb])

time_use = (datetime.now() - t0).total_seconds() #timing(end)

#os.mkdir("./save/")
## serialize model to JSON
#model_json = vae.to_json()
#with open(modelpath+".json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#vae.save_weights(modelpath+".h5")
#print("Saved model to ", modelpath)

#%% Display Training History

print(vae.history.history.keys()) #list all keys in history
logs = vae.history.history
epoch_real = len(logs['loss'])

## output history (loss) as .csv
import csv
with open("./saveLoss.csv", "a+") as fp:
    csv.writer(fp, dialect='excel', lineterminator='\n').writerow(logs['loss'])
    csv.writer(fp, dialect='excel', lineterminator='\n').writerow(logs['val_loss'])

if plot_on:
    plt.plot(logs['loss']    , color=[.090,.773,.804])#'LightSeaGreen'
    plt.plot(logs['val_loss'], color=[.816,.126,.565])
    plt.title('Model Loss, epochs=%s'%(str(epoch_real)))
    plt.ylabel('Loss_KL')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('./fig/loss_{}_latent_{}_ep_{}.png'.format(use_loss,dim_latent,epoch_real))
    plt.show()
    
## output history (loss_kl) as .csv
import csv
with open("./saveLoss_kl.csv", "a+") as fp:
    csv.writer(fp, dialect='excel', lineterminator='\n').writerow(logs['loss_kl'])
    csv.writer(fp, dialect='excel', lineterminator='\n').writerow(logs['val_loss_kl'])

if plot_on:
    plt.plot(logs['loss_kl']    , color=[.090,.773,.804])#'LightSeaGreen'
    plt.plot(logs['val_loss_kl'], color=[.816,.126,.565])
    plt.title('Model Loss, epochs=%s'%(str(epoch_real)))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('./fig/loss_kl_{}_latent_{}_ep_{}.png'.format(use_loss,dim_latent,epoch_real))
    plt.show()

#%% Visualization: Latent Space

# build a model to encode inputs(x) to the latent space(z)
encoder = Model(x, z_mean)
# display a 2D plot of the digit classes in the latent space
z_encoded_from_x = encoder.predict(x_test, batch_size=batch_size)

if dim_latent == 2 and plot_on:
    fig = plt.figure()#fig = plt.figure(figsize=(8, 6))
    plt.scatter(z_encoded_from_x[:, 0], 
                z_encoded_from_x[:, 1], c=y_test)
    plt.colorbar()
    plt.title('2D Latent Space of Digits')
    fig.show()
    fig.savefig('./fig/z_{}_latent_{}_ep_{}.png'.format(use_loss,dim_latent,epoch_real))

if dim_latent == 3 and plot_on:
    fig = plt.figure()#fig = plt.figure(figsize=(8, 6))
    axs = fig.add_subplot(111, projection='3d')
    pic = axs.scatter(z_encoded_from_x[:, 0], 
                      z_encoded_from_x[:, 1], 
                      z_encoded_from_x[:, 2], c=y_test)
    fig.colorbar(pic)
    plt.title('3D Latent Space of Digits')
    plt.show()
    plt.savefig('./fig/z_{}_latent_{}_ep_{}.png'.format(use_loss,dim_latent,epoch_real))

#%% Visualization: 2D Digits Manifold

if dim_latent == 2 and plot_on:

    ## build a digit generator that can sample from the learned distribution
    z_sampled = Input(shape=(dim_latent,))
    x_decoded = decoder_output(decoder_hidden(z_sampled))
    generator = Model(z_sampled, x_decoded)
    
    # display a 2D manifold of the digits
    n = 20 #figure with 15x15 digits
    m = int(np.sqrt(dim_x)) #digit size
    figure = np.zeros((m * n, m * n))
    # linearly spaced coordinates on the unit square were transformed through the 
    # inverse CDF (ppf) of the Gaussian to produce values of the latent variables 
    # z, since the prior of the latent space is Gaussian.
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decode = generator.predict(z_sample)
            figure[i * m: (i + 1) * m,
                   j * m: (j + 1) * m] = \
                   x_decode[0].reshape(m, m)
                   
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title('2D Digits Manifold (%s x %s)'%(n,n))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    fig.savefig('./fig/x_{}_latent_{}_ep_{}_n_{}.png'.format(use_loss,dim_latent,epoch_real,n))

#%% Visualization: Reconstruction

if plot_on: 

    n = 20 #figure with 15x15 digits
    m = int(np.sqrt(dim_x)) #digit size
    
    figure = np.zeros((m * 2, m * n))
    x = x_test[:batch_size,:]
    x_recon = vae.predict(x, batch_size=batch_size).reshape((-1, m, m))
    x = x.reshape((-1, m, m))
    x_recon = x_recon.reshape((-1, m, m))
    for i in range(n):
        figure[:m,      i*m:(i+1)*m] = x[i]
        figure[ m:2*m,  i*m:(i+1)*m] = x_recon[i]
    
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title('Image Reconstruction')
    plt.xticks([])
    plt.yticks(m*np.array([.5,1.5,2.5]),['Origin','Re-con'])
    fig.savefig('./fig/re_{}_latent_{}_ep_{}_n_{}.png'.format(use_loss,dim_latent,epoch_real,n))
    plt.show()

#%% Visualization: Image Imputation

if plot_on: 
    
    n = 20 #figure with 15x15 digits
    m = int(np.sqrt(dim_x)) #digit size
    
    figure = np.zeros((m * 3, m * n))
    x = x_test[:batch_size,:]
    x_corrupted = np.copy(x)
    x_corrupted[:, 300:400] = 0
    x_reconstruted = vae.predict(x_corrupted, batch_size=batch_size).reshape((-1, m, m))
    x = x.reshape((-1, m, m))
    x_corrupted = x_corrupted.reshape((-1, m, m))
    for i in range(n):
        figure[:m,      i*m:(i+1)*m] = x[i]
        figure[ m:2*m,  i*m:(i+1)*m] = x_corrupted[i]
        figure[   2*m:, i*m:(i+1)*m] = x_reconstruted[i]
    
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title('Image Imputation')
    plt.xticks([])
    plt.yticks(m*np.array([.5,1.5,2.5]),['Original','Corrupt','Re-con'])
    fig.savefig('./fig/i_{}_latent_{}_ep_{}_n_{}.png'.format(use_loss,dim_latent,epoch_real,n))
    plt.show()