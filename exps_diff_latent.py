'''
Description:
    To summary VAE with different dim_latent performance on MINST.

Notes: 
    To get results as in "csv/saveLoss_latent.csv", run this in linux shell:
    for i in {2,3,5,10,15,20}; do python vae.py $i; done
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'family': 'normal',
        'weight': 'bold',
        'size': 14}
matplotlib.rc('font', **font)

# read history
file = open('csv/saveLoss_latent.csv').read().split('\n')
try:
    tick = file.index('')
    file = file[:tick]
except: 
    pass

logs = []
epoch_max = 0
epoch_min = 1e4
for i in range((len(file))):
    temp = file[i].split(',')
    try:
        tick = temp.index('')
        temp = temp[:tick]
    except: 
        pass
    logs.append(np.array([float(i) for i in temp]))
    epoch_max = max([len(logs[i]), epoch_max])
    epoch_min = min([len(logs[i]), epoch_min])

fig = plt.figure(figsize=(8,6))
fig = plt.subplot(111)
color = ['crimson','darkorange','green','darkcyan','midnightblue','purple']
label = ['2','3','5','10','15','20']
#label = ['latent = 2','latent = 3','latent = 5','latent = 10','latent = 15','latent = 20']
# plotting each experiment
for i in range(0,int(len(file)/2)):
    plt.plot(logs[2 * i], color=color[i], label=label[i])
    plt.plot(logs[2*i+1], color=color[i], alpha=0.3, linewidth=2)

plt.title('Model Loss, VAE with different dim_latent')
plt.ylabel('Loss' , fontsize=18)
plt.xlabel('Epoch', fontsize=18)
#plt.axis([-5,75,100,210])
#handles, labels = fig.get_legend_handles_labels()
plt.legend()
plt.savefig("./fig/latent_exps", dpi=300)#dpi=1200)
plt.show()