'''
Description:
    To test the stability of VAE: 
        A summary of VAE performance on MINST in multiple experiments.

Notes:
    1. In "saveLoss_multi.csv":
    each odd  number line is "loss" of one experiment;
    each even number line is "loss_val" (validation) of the corresponding experiment;
    
    2. To get results as in "csv/saveLoss_multi.csv", run this in linux shell:
    i=0; while (( $i < 30 )); do python vae.py; ((i=$i+1)); done;
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'family': 'normal',
        'weight': 'bold',
        'size': 14}
matplotlib.rc('font', **font)

# Read history
file = open('csv/saveLoss_multi.csv').read().split('\n') 
num  = int((len(file)-1)/2)
logs = []
epoch_max = 0
epoch_min = 1e4
for i in range(2*num):
    logs.append(np.array([float(i) for i in file[i].split(',')]))
    epoch_max = max([len(logs[i]), epoch_max])
    epoch_min = min([len(logs[i]), epoch_min])

fig = plt.figure(figsize=(8,6))
fig = plt.subplot(111)

# Plotting each experiment
for i in range(num):
    plt.plot(logs[2 * i], color='PowderBlue', label='Train')
    plt.plot(logs[2*i+1], color='MistyRose' , label='Test' )

# Calculate mean_loss and mean_loss_val (validation)
mean_loss = []
for v in range(epoch_max):
    loss, n_exps = 0, 0
    for i in range(0,num*2,2):
        try:
            loss = logs[i][v]+loss
            n_exps += 1
        except:
            pass
    mean_loss.append(loss/n_exps)

mean_loss_val = [] 
for v in range(epoch_max):
    loss, n_exps = 0, 0
    for i in range(1,num*2,2):
        try:
            loss = logs[i][v]+loss
            n_exps += 1
        except:
            pass
    mean_loss_val.append(loss/n_exps)
 
# Create directory to save the figure
import os
if not os.path.exists("./fig"): os.mkdir("./fig")

# Plotting mean
plt.plot(mean_loss,     color=[.090,.773,.804], label='mean(Train)')
plt.plot(mean_loss_val, color=[.816,.126,.565], label='mean(Test)')
plt.title('Model Loss, %s Experiments'%str(num))
plt.ylabel('Loss' , fontsize=18)
plt.xlabel('Epoch', fontsize=18)
handles, labels = fig.get_legend_handles_labels()
plt.legend(handles[0:2]+handles[::-1][0:2][::-1],
           labels[0:2]+labels[::-1][0:2][::-1],  
           loc='upper right')
plt.savefig("./fig/loss_{}_exps".format(num), dpi=300)#dpi=1200
plt.show()