import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sigGen as sg
import matplotlib as mpl
#from funaux import *
from stabfunc import *

import scienceplots
plt.style.use(['science','ieee'])
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'figure.dpi':'100'})

mpl.rcParams['legend.frameon'] = 'True'
mpl.rcParams['legend.facecolor'] = 'w'

if __name__ == "__main__":
    os.chdir("../data")
    datadir = os.getcwd()
    csv_name = []
    for file in os.listdir(datadir):
        if file.endswith('.csv'):
            csv_name = file
            a = input('Plot file'+str(file)+' ?(y,n): ')
            if a == 'y':
                break
    
    path = os.path.normpath(csv_name)
    df = pd.read_csv(path,skiprows=2,skipinitialspace=True)
    global real_data, U_rms
    real_data_h = df.iloc[:,12].to_numpy().flatten()
    real_data_ca = df.iloc[:,10].to_numpy().flatten()
#        signal_Obj = sg.Signal.fromparams(ddir+'/'+csv_name[:-9]+'params_'+csv_name[-5]+'.txt')
#        signal = np.array(signal_Obj.signal[0]).flatten()
#        signal_rms = window_rms(signal,5000)
#        U_rms = signal_rms[::int(len(signal)/len(real_data))]
    #plt.plot(U_rms[:len(real_data)],real_data,'--')
    fig, ax1 = plt.subplots()
    ax1.plot(real_data_h,'--',label='Height')
    ax2 = ax1.twinx()
    ax2._get_lines.prop_cycler = ax1._get_lines.prop_cycler
    ax2.plot(real_data_ca,'--',label='CA')
    ax1.legend()
    ax2.legend()
    plt.show()

