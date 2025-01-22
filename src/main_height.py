import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


import pandas as pd
import os
import sys
import cv2
import matplotlib.ticker as mtick
import imageio
from stabfunc import *
from funaux import *
from ene_model_faux import *
from scipy.optimize import fmin
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import differential_evolution
from scipy.optimize import brute
from scipy.optimize import curve_fit
from multiprocessing.pool import ThreadPool
import sigGen as sg
import matplotlib as mpl
import time
import json


from lmfit import create_params, minimize, fit_report, Minimizer

import scienceplots
plt.style.use(['science','ieee'])
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'figure.dpi':'100'})

mpl.rcParams['legend.frameon'] = 'True'
mpl.rcParams['legend.facecolor'] = 'w'

#def red_csv():

if __name__ == "__main__":
    #args = sys.argv[1:]
    #if args [0] == 'm1': 
    #    print('m1')
    os.chdir("../data")
    datadir = os.getcwd()
    fname,ddir = select_file(datadir)
    flag_read_csv = 1

    if flag_read_csv == 1:
        csv_name = []
        fig, ax = plt.subplots(figsize=(3,3))
        i = 0
        labels = []
        df_rv = []
        Vmax = np.array([70,55,40,30])
        arg_sort = np.argsort(Vmax)
        for file in sorted(os.listdir(ddir)):
            if file.endswith('.csv') and file.startswith('G'):
                csv_name = file
                exp_n = int(csv_name[-5])+1
                #labels.append(f'V {exp_n}')
                labels.append(r'$V_{m}$=%5.1f $V_{rms}$ ' %Vmax[i])
                print(file)
                path = os.path.normpath(csv_name)
                df = pd.read_csv(path,skiprows=2,skipinitialspace=True)
                real_data = df.iloc[:,12].to_numpy().flatten()
                real_data = real_data-real_data[0]
                df_rv.append(real_data)
                signal_Obj = sg.Signal.fromparams(ddir+'/'+csv_name[:-9]+'params_'+csv_name[-5]+'.txt')
                signal = np.array(signal_Obj.signal[0]).flatten()
                signal_rms = window_rms(signal,5000)
                U_rms = signal_rms[::int(len(signal)/len(real_data))]
                plt.plot(U_rms[:len(real_data)], real_data, label=labels[i],markersize=1)
                d_s = np.column_stack((U_rms[:len(real_data)], real_data))
                np.savetxt(file+'_height_data.txt',d_s)
                i = i+1
#        paths = [os.path.normpath(csv_name[i]) for i in range(len(csv_name))]
#        data_u = []
#        data_theta = []

        #plt.plot(U_rms[:len(real_data)],real_data,'--')
        plt.xlabel('Voltage (Vrms)')
        plt.ylabel(r'$\propto$ $(h-h_0)$ ')
        plt.legend()
        plt.savefig('All_csv_plots_height'+'.pdf', bbox_inches = 'tight')
        plt.show()





