import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import scienceplots
from stabfunc import *
plt.style.use(['science','ieee'])
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'figure.dpi':'100'})

mpl.rcParams['legend.frameon'] = 'True'
mpl.rcParams['legend.facecolor'] = 'w'



os.chdir("../data")
datadir = os.getcwd()
print(datadir)
#fname,ddir = select_file(datadir)

txt_name = []
df_txt = []
df_txt2 = []

for file in os.listdir(datadir):
    #if file.endswith('paramsFIT.txt'):
    if file.endswith('dataFIT.txt'):
        path = os.path.normpath(file)
        txt_name.append(file)
        df_txt.append(np.loadtxt(file))
    elif file.endswith('data.txt'):
        df_txt2.append(np.loadtxt(file))

fig = plt.figure(figsize=(3.3,2.8))
markers = ["v", "s", "*", "^", "d", "v", "s", "*", "^"]

Vmax = np.array([30,70,40,55])
arg_sort = np.argsort(Vmax)

for i in range(len(df_txt2)):
    U = df_txt2[arg_sort[i]][:,0]
    plt.plot(U, df_txt2[arg_sort[i]][:,1], markersize=2, label=r'$V_{m}$=%5.1f $V_{rms}$ ' %Vmax[arg_sort[i]])

plt.xlabel('Voltage (Vrms)')
plt.ylabel('Contac angle (deg)')
plt.legend()
plt.savefig('AllPlots.pdf', bbox_inches = 'tight')

for i in range(len(txt_name)):
    fig = plt.figure()
    print(fig.get_size_inches())
        # 1st increasing V out
#        y = y[x.argmax():]
#        x = x[x.argmax():]
    U = df_txt[i][:,0]
    arg_sort = np.argsort(U)
    U = np.take_along_axis(U, arg_sort, axis=0)
    y1 = np.take_along_axis(df_txt[i][:,1], arg_sort, axis=0)
    y2 = np.take_along_axis(df_txt[i][:,2], arg_sort, axis=0)
    plt.plot(np.abs(U),y1 ,'.',markersize=2 ,label='data')
    plt.plot(np.abs(U), y2, label='fit')

    plt.xlabel('Voltage (Vrms)')
    plt.ylabel('Contac angle (deg)')
    plt.legend()
    plt.savefig(txt_name[i][:-4]+'.pdf', bbox_inches = 'tight')

