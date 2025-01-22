import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import pandas as pd
import os
import sys
import cv2
import matplotlib.ticker as mtick
#import imageio
from stabfunc import *
from funaux import *
from ene_model_faux import *
from scipy.optimize import fmin
from scipy.interpolate import interp1d
#from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import differential_evolution
from scipy.optimize import brute
from scipy.optimize import curve_fit
from multiprocessing.pool import ThreadPool
import sigGen as sg
import matplotlib as mpl
import time
import json
from sklearn.metrics import r2_score 

import matplotlib.image as mpimg

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps 
from lmfit import create_params, minimize, fit_report, Minimizer

import scienceplots
plt.style.use(['science','ieee'])
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#plt.rcParams.update({'figure.dpi':'100'})

mpl.rcParams['legend.frameon'] = 'True'
mpl.rcParams['legend.facecolor'] = 'w'

#def red_csv():

def calc_cpin(gamma,theta_a,theta_r):
    return gamma*(np.cos(theta_r*np.pi/180)-np.cos(theta_a*np.pi/180))/2
def calc_theta0(theta_r):
    theta_0 = np.arccos(-cpin_v/gamma_lg+np.cos(theta_r*np.pi/180))*180/np.pi

if __name__ == "__main__":
    #args = sys.argv[1:]
    #if args [0] == 'm1': 
    #    print('m1')
    os.chdir("../data")
    datadir = os.getcwd()
    fname,ddir = select_file(datadir)
    gamma_lg = 64.8/1000 # g/s^2 from microkat.gr
    #dtheta = 0.2
    eps_d = 2
    d = 0.5e-6
    #V = 3.25e-9 # 0.8 uL in m3
    #V = 2.73e-9 # 0.8 uL in m3 3.1 (90)
    V = 0.7e-9
    #V = 0.104e-9
    #V = 3.005e-9 # 3.005
    #V = 0.0627e-9  # 0.8 uL in m3 3.1 (90) 0.62 REAL
    #V = 0.0617e-9  # 0.8 uL in m3 3.1 (90) 0.62 REAL
    #V = 0.074e-9  # 0.8 uL in m3 3.1 (90) 0.62 REAL
    #V = 0.048e-9  # 0.8 uL in m3 3.1 (90) 0.62 REAL
    N_U = 100

    theta_0 = 100.4
    theta_0_2 = 98.8
    C_g = 1.88
    C_g2 = 3.3
    U_shift = 0
    theta_shift = 0
    theta_shift2 = 0
    chi = 0.08e-3   # Coefficient of con0-65-2.5-2023.09.28 16.39.59tact line friction [Kg/(s.m)] From paper not glyucerine
    eta = 5e-4 # Kinematic viscosity of glycerine m2/s
    u1 = 15
    u2 = 70
    u3 = 100
    U_max = u3
    Afunc = A_calc(fname,ddir,V=V)
    flag_create_table = 0
    flag_read_csv = 1
    flag_fitting = 1
    flag_theta_stable = 0
    flag_min_theta = 0
    flag_stable_wtable = 0
    flag_plot_height = 0
    flag_all_curves = 0
    flag_all_csv = 0
    flag_inv_model = 0
    flag_all_wfit = 0
    tripod_fit = False
    
    theta_r = 98.05 #93.58
    theta_a = 103.434 #103.417
#    for t_a in theta_a:
#        cpin.append(calc_cpin(gamma_lg,t_a, theta_r))
#    plt.plot(theta_r, np.array(cpin))
#    plt.show()
    cpin_v = calc_cpin(gamma_lg, theta_a, theta_r)#*0.7
    theta_0 = np.arccos(-cpin_v/gamma_lg+np.cos(theta_r*np.pi/180))*180/np.pi

    #print('cpin value: ', cpin_v)
    #print('thet_0 value: ', theta_0)
    k1 =  (2.1-0.9)/2*eta/((chi+6*eta))
    #print('k1...: ', k1)
    #print('k1...: ', k1*(chi+6*eta)/eta)
    #V_c = np.pi*(1160e-6)**3/3*f_theta(92)/np.sin(92*np.pi/180)
    h = 1279
    t_eq = 98.6
    V_c = np.pi*(h*1e-6)**3/3*f_theta(t_eq)/(1-np.cos(t_eq*np.pi/180))**3
    h = 384
    t_eq = 96
    V_c2 = np.pi*(h*1e-6)**3/3*(2+np.cos(t_eq*np.pi/180))/(1-np.cos(t_eq*np.pi/180))
    #print('Volume: ', V_c*1e9)
    #print('Volume2: ', V_c2*1e9)
    #print('r: ', h*np.sin(t_eq*np.pi/180)/(1-np.cos(t_eq*np.pi/180)))
    
    if flag_all_wfit:
        V_l = []
        # I firstly plot all the data
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
                r_d = df.iloc[:,10].to_numpy().flatten()
                r_d = r_d
                df_rv.append(r_d)
                signal_Obj = sg.Signal.fromparams(ddir+'/'+csv_name[:-9]+'params_'+csv_name[-5]+'.txt')
                signal = np.array(signal_Obj.signal[0]).flatten()
                signal_rms = window_rms(signal,5000)
                Urms = signal_rms[::int(len(signal)/len(r_d))]
                x = Urms[:len(r_d)]
                x2 = x[:-1].copy()
                x2[x[:-1] > x[1:]] = -x2[x[:-1] > x[1:]]
                V_l.append(x2)
                plt.scatter(Urms[:len(r_d)], r_d, label=labels[i],s=10,marker='v', color=cycle[i],alpha=0.1)
                d_s = np.column_stack((Urms[:len(r_d)], r_d))
                np.savetxt(file+'_height_data.txt',d_s)
                i = i+1
        # I fit all the data
        fit_params = create_params(C_g=dict(value=1.33, max = 1.4, min=1.2, vary=True),
                                   C_g2=dict(expr='C_g', max=5.5, min=0.5),    #expr='C_g'
                                   #C_g2=dict(value=2.116, max=2.45, min=1.85, vary=False),
                                   theta0=dict(value=99.63, max=102, min=98, vary=False),
                                   #theta0=dict( expr= 'acos(-cpin/0.0648+cos(98.05*pi/180))*180/pi'),
                                   theta0r=dict(expr='theta0', max=103, min=97, vary=False),
                                   C1=dict(value=0.08, max=1.1e-1, min=-1.8e-1, vary=False),
                                   cpin=dict(value=7.14e-4, max=1e-3, min=1e-4, vary=False),
                                   cpin2=dict(expr='cpin', max=1e-2, min=1e-5, vary=False)
                                   #cpin2=dict(value=7.14e-4, max=1e-3, min=1e-5, vary=True)
                                   )
        C_g = fit_params['C_g'].value
        C_g2 = fit_params['C_g2'].value
        theta0 = fit_params['theta0'].value
        theta0r = fit_params['theta0r'].value
        C1 = fit_params['C1'].value
        cpin = fit_params['cpin'].value
        model = 1
        d_theta = 5
        dt = 0.001
        theta_x = theta0
        fname_csv = 'r_gf_' + fname+'.csv'
        data_table = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
        #print(data_table.shape)
        act = EH_Actuator(V, gamma_lg, theta0, d, eps_d, theta0r = theta0r
                         ,C_g = C_g, C_g2 = C_g2, cpin = cpin, uth1 = 1
                         , uth2 = 120, model=model)    # Create actuator w/params
        act.C1 = C1
        act.load_table(data_table)
        find_uth1_uth2(act, 65, model)
        V_l = []
        V_l.append(np.linspace(-65,65,100))
        for i in range(len(V_l)):
            t = time.time()
            #fit = theta_func0_lmfit(fit_params, V_l[i], V, d, gamma_lg, eps_d, fname)
            fit = h_func0_lmfit(fit_params, V_l[i], V, d, gamma_lg, eps_d, fname)
            print('Fitting time: ', time.time()-t)
            plt.plot(np.abs(V_l[i]), fit, '-.', label='fit', color=cycle[i])

        plt.xlabel('Voltage (Vrms)')
        plt.ylabel('Contact Angle (deg)')
        plt.legend()
        plt.savefig('All_wfit_CA'+'.pdf', bbox_inches = 'tight')
        #plt.show()

    if flag_read_csv == 1:
        csv_name = []
        fromGUI = False
        fromPYDSA = True
        d_equiv = 32
        if fromPYDSA:
            if fromGUI==False:
                for file in os.listdir(ddir):
                    if file.endswith('.csv'):
                        if file.startswith('G'):
                            csv_name = file
            else:
                
                csv_name = 'static-hyst-loops_IDS.csv'
                #csv_name = 'HLoop-C-03-01_IDS_0.csv'
            print(f'CSV name: {csv_name}')
            path = os.path.normpath(csv_name)
            df = pd.read_csv(path,skiprows=2,skipinitialspace=True)
            global real_data, U_rms
            #real_data = df.iloc[:,10].to_numpy().flatten()
            real_data_t = df.iloc[:,10].to_numpy().flatten()
            real_data = df.iloc[:,12].to_numpy().flatten()#*0.981
            real_data_r = df.iloc[:,11].to_numpy().flatten()#*0.981
            real_data_v = df.iloc[:,14].to_numpy().flatten()#*0.981**3
            if fromGUI==False:
                signal_Obj = sg.Signal.fromparams(ddir+'/'+csv_name[:-9]+'params_'+csv_name[-5]+'.txt')
                signal = np.array(signal_Obj.signal[0]).flatten()
                #signal = signal[::200]
                signal_rms = window_rms(signal,500)
                signal_rms = signal_rms[::200]
                U_rms = signal_rms[::int(len(signal)/len(real_data))]
                #U_rms = signal_rms[2:]
            else:
                sign_name = 'input - static-hyst-loops.txt'
                #sign_name = 'input-HLoop-C-03.txt'
                
                sig = np.loadtxt(sign_name)
                U_rms = sig[:,0]#[::200]
        else:
            csv_name = 'static-hyst-loops_IDS.csv'
            fname_csv = 'hyst-tripod-h.csv'
            #data = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
            data = np.genfromtxt(fname_csv, delimiter =' ',dtype=None)
            U_rms = data[:,0]
            real_data = data[:,1]*1e6 +d_equiv
            real_data_t = np.arange(len(real_data))/1000
        # Resample data to have the same points in voltage and in height
        x_data = np.linspace(0,1,len(real_data))
        x_u = np.linspace(0,1,len(U_rms))
        f = interp1d(x_u, U_rms)
        U_rms = f(x_data)
        print('len U_rms:', len(U_rms))
        print('len h data:', real_data.shape)
        # Filter
        
#%%
        try:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
            #ax[0].plot(U_rms[:len(real_data)],real_data,'--')
            #ax[1].plot(U_rms[:len(real_data_t)],real_data_t,'--')
            x,y = U_rms[:len(real_data_t)],real_data
            N = 10800
            ax.plot(x[N:],y[N:],'--')
            x = np.array([0,65])
            y = np.array([1,1])
            #for i in np.arange(37,42):
                #ax.fill_between(x,((20+10)*i+20)*y, ((20+10)*i+0)*y,color='gold', alpha=0.3)
                #ax.fill_between(x,((20+10)*i+30)*y, ((20+10)*i+20)*y,color='r', alpha=0.2)
            ax.set_xlabel('Voltage (Vrms)')
            ax.set_ylabel('Contact radius (um)')
            #ax[2].axhline(1190,0,60,c='b')
            plt.show()
            
        except:
            pass
        
        #%%

    if flag_inv_model == 1:

            b = 1150
            a = 1010
            np.random.seed(2)
            h_r = (b-a)*np.random.randint(0,100,20)/100+a
            print(h_r)
            #np.savetxt('h_d.txt',h_r)

            fit_params = create_params(C_g=dict(value=1.756, max = 2.2, min=1.7, vary=True),
                                       C_g2=dict(value=1.66, max=2.0, min=1.5, vary = False ),    #expr='C_g'
                                       #C_g2=dict(value=1.1, max=30.1, min=0.1, vary=False),
                                       theta0=dict(value=97.16, max=98.5, min=97, vary=False),
                                       #theta0=dict( expr= 'acos(-cpin/0.0648+cos(98.05*pi/180))*180/pi'),
                                       theta0r=dict(expr='theta0', max=103, min=97, vary=False),
                                       C1=dict(value=0.0, max=1.1e1, min=-1.8e-1, vary=False),
                                       cpin=dict(value=1.0e-3, max=2.5e-3, min=1e-3, vary=False),
                                       cpin2=dict(expr='cpin', max=1e-2, min=1e-5, vary=False)
                                       #cpin2=dict(value=7.14e-4, max=1e-3, min=1e-5, vary=True)
                                       )

            C_g = fit_params['C_g'].value
            C_g2 = fit_params['C_g2'].value
            theta0 = fit_params['theta0'].value
            theta0r = fit_params['theta0r'].value
            C1 = fit_params['C1'].value
            cpin = fit_params['cpin'].value
            model = 1
            d_theta = 5
            dt = 0.001
            theta_x = theta0
            fname_csv = 'r_gf_' + fname+'.csv'
            data_table = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
            #print(data_table.shape)
            act = EH_Actuator(V, gamma_lg, theta0, d, eps_d, theta0r = theta0r
                             ,C_g = C_g, C_g2 = C_g2, cpin = cpin, uth1 = 1
                             , uth2 = 120, model=model)    # Create actuator w/params
            act.C1 = C1
            act.load_table(data_table)
            find_uth1_uth2(act, 60, model)

            x = np.linspace(-60,60,1000)
            
            fit = h_func0_lmfit(fit_params, x, V, d, gamma_lg, eps_d, fname)
            np.savetxt('fit_val.txt',fit)
            np.savetxt('u_val.txt',x)
            #%%

            
            fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(3.3,3))
            
            
            axs.plot(x,fit,'--')
            
            f_inv_adv = interp1d(fit[x>=0],x[x>=0])
            f_inv_rec = interp1d(fit[x<0],x[x<0])
            h_aux = np.arange(1000,1145)
            axs.plot(f_inv_adv(h_aux),h_aux)
            axs.plot(f_inv_rec(h_aux),h_aux)
            
            #%%
            h_r = np.loadtxt('h_d.txt')
            dif = np.diff(h_r)
            dif[dif>0] = 0
            dif[dif<0] = 1
            dif = np.insert(dif,0,1.0)
            U_r = []
            U_r.append(f_inv_adv(h_r[0]))
            for i in range(len(h_r[1:])):
                if h_r[i+1]<h_r[i]:
                    U_r.append(f_inv_adv(h_r[i+1]))
                else:
                    U_r.append(f_inv_rec(h_r[i+1]))
            fig, ax1 = plt.subplots()
            ax1.set_ylabel('Height')
            ax1.plot(h_r, '-*' ,color='tab:red')
            ax1.tick_params(axis='y',labelcolor='tab:red')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Voltage')
            ax2.legend()
            ax2.plot(U_r,'-*' ,color='k')
            ax2.tick_params(axis='y',labelcolor='k')
            plt.show()
            
#%%
#            fig, ax1 = plt.subplots()
#            ax1.set_ylabel('Contact Angle')
#            ax1.plot(t_r, '-*' ,color='tab:red')
#            ax1.tick_params(axis='y',labelcolor='tab:red')

#            ax2 = ax1.twinx()
#            ax2.set_ylabel('Voltage')
            
#            ax2.plot(U_r,'-*' ,color='k')
#            ax2.tick_params(axis='y',labelcolor='k')
#            plt.show()

            f = 1000
            t = 5
            N = t*f*len(h_r)
            sig = np.ravel(np.resize(U_r,(t*f,len(h_r))).T)
            sig = np.abs(sig)
            h_dt = np.ravel(np.resize(h_r,(t*f,len(h_r))).T)
            signals = np.column_stack((sig,sig,sig))
            #np.savetxt('input_OL.txt',signals)
            fig, ax1 = plt.subplots()
            ax1.set_ylabel('Height')
            
            ax1.plot(np.arange(N)/f,h_dt, '--',label='sp' ,color='tab:red')
            ax1.tick_params(axis='y',labelcolor='tab:red')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('Voltage')
            ax2.set_xlabel('Time (s)')
            ax2.plot(np.arange(N)/f,sig,label='signal')
            ax2.legend()
            #ax2.plot(U_r,'-*' ,color='k')
            ax2.tick_params(axis='y',labelcolor='k')
            plt.show()


#            print(t_r)
#            print(dif)
#            theta = 90

            plt.figure()
            x_data_extended = np.linspace(-70,70,100)
            #fit = theta_func0_lmfit(fit_params, x_data_extended, V, d, gamma_lg, eps_d, fname)
            plt.plot(np.abs(x), fit, '-.', label='fit')
            #U = inv_theta(act, t_r)
            plt.plot(np.abs(U_r), h_r,'.', label='Inverse')
            #U = inv_theta(act, t_r, advancing=np.zeros_like(t_r))
           # plt.plot(U, t_r,'.', label='Inverse')
            plt.xlabel('Voltage (Vrms)')
            plt.ylabel('Contac angle (deg)')
            plt.legend()
            plt.grid()
            
            
            #plt.show()
            #%%
            
            if flag_read_csv == 1:
                #arr_img = plt.imread('wscale2.png', format='png')
                #arr_img2 = plt.imread('wscale1.png', format='png')
                arr_img = Image.open('wscale2.png').convert('L')
                arr_img = ImageOps.invert(arr_img)
                arr_img2 = Image.open('wscale1.png').convert('L')
                arr_img2 = ImageOps.invert(arr_img2)
                
                imagebox = OffsetImage(arr_img, cmap=plt.cm.gray_r, zoom=0.088)
                imagebox2 = OffsetImage(arr_img2, cmap=plt.cm.gray_r, zoom=0.088)
                imagebox.image.axes = ax
                xy = [5,1150]
                xy2 = [60,1000]
                ab = AnnotationBbox(imagebox, xy,
                    xybox=(10., -40.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.1,
                    arrowprops=dict(
                        arrowstyle="->",
                        mutation_scale=5,
                        connectionstyle="angle,angleA=0,angleB=90,rad=1"),
                    bboxprops=dict(linewidth=0.2)
                    )
                ab2 = AnnotationBbox(imagebox2, xy2,
                    xybox=(-72., 11.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.1,
                    arrowprops=dict(
                        arrowstyle="->",
                        mutation_scale=5),#,
                        #connectionstyle="angle=,angleA=0,angleB=90,rad=2")
                    bboxprops=dict(linewidth=0.2)
                    )
                
                fig, ax = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(3.3,2.4))
                ax.add_artist(ab)
                ax.add_artist(ab2)
                ax.text(0.15,0.35, r'Droplet', horizontalalignment='left',
                     verticalalignment='top', transform=ax.transAxes,
                     backgroundcolor='1')
                ax.quiver(11,1054,-2,6.5,scale=60,headwidth=8,headlength=10,headaxislength=10, width=0.003, zorder=10)
                ax.quiver(12,1039, 12,-7,scale=60,headwidth=8,headlength=10,headaxislength=10, width=0.003, zorder=10)
                ax.plot(np.abs(x)[::5], fit[::5], '--', label='Model',zorder=20)
                ax.plot(U_rms[:len(real_data_t)], real_data,'.', markersize=2, label='Data')
                ax.set_ylabel(r'Droplet height ($\mu$m)')
                ax.set_xlabel(r'Voltage ($V_{rms}$)')
                
                ax.legend()
                ax.quiver(32,1150, 22,-22,scale=60,headwidth=10,headlength=10,headaxislength=10, width=0.003)
                ax.quiver(46,1025,-22,22,scale=60,headwidth=10,headlength=10,headaxislength=10, width=0.003)
                #ax.arrow(38,1040,-23,70,head_width=2, width=0.000001)
                fig.savefig('Alldata-w-model'+'.pdf', bbox_inches = 'tight')
                        
#%%
    if flag_all_csv ==1:
        csv_name = []
        fig, ax = plt.subplots(figsize=(3,3))
        i = 0
        labels = []
        df_rv = []
        for file in sorted(os.listdir(ddir)):
            if file.endswith('.csv') and file.startswith('G'):
                csv_name = file
                exp_n = int(csv_name[-5])+1
                labels.append(f'exp {exp_n}')
                print(file)
                path = os.path.normpath(csv_name)
                df = pd.read_csv(path,skiprows=2,skipinitialspace=True)
                #real_data = df.iloc[:,10].to_numpy().flatten()
                real_data = df.iloc[:,12].to_numpy().flatten()
                df_rv.append(real_data)
                signal_Obj = sg.Signal.fromparams(ddir+'/'+csv_name[:-9]+'params_'+csv_name[-5]+'.txt')
                signal = np.array(signal_Obj.signal[0]).flatten()
                signal_rms = window_rms(signal,2000)
                U_rms = signal_rms[::int(len(signal)/len(real_data))]
                
                x = U_rms[:len(real_data)]
                print('xMax', max(x))
                print('len x data: ', len(x))
                y = real_data
                N=4
                mean_kernel = np.ones(N)/N
                print(len(y))
                fs = np.convolve(y, mean_kernel, mode='same')
                print(len(fs))
                y = fs
                # To remove the 1st cycle 
                xd = x[:-1] - x[1:]
                xd[xd< -0.01 ] =  -1
                xd[xd>  0.01 ] = 1
                for i in range(1):
                    y = y[xd.argmax():]
                    x = x[xd.argmax():]
                    xd = xd[xd.argmax():]

                    y = y[xd.argmin():]
                    x = x[xd.argmin():]
                    xd = xd[xd.argmin():]
                
                #plt.plot(U_rms[:len(real_data)], real_data, '-.',label=labels[i],markersize=1)
                plt.plot(x[y>1000], y[y>1000], '.-',label=labels[i],markersize=1)
                d_s = np.column_stack((U_rms[:len(real_data)], real_data))
                np.savetxt(file+'_data.txt',d_s)
                i = i+1
#        paths = [os.path.normpath(csv_name[i]) for i in range(len(csv_name))]
#        data_u = []
#        data_theta = []

        #plt.plot(U_rms[:len(real_data)],real_data,'--')
        plt.xlabel('Voltage (Vrms)')
        plt.ylabel('Height (um)')
        #plt.legend()
        plt.savefig('All_csv_plots'+'.pdf', bbox_inches = 'tight')
        plt.show()

#%%
    if flag_fitting:
        print('ASD')
        x = U_rms[:len(real_data)]
        print('xMax', max(x))
        print('len x data: ', len(x))
        fig_all, ax = plt.subplots(figsize=(3.3,2.5))
        #ax.plot(x_data, y_data, label='Data')
        x = U_rms[:len(real_data)]
        y = real_data
        y = real_data_t
        ax.plot(x, y, '.',markersize=1,label='Data')
        ax.set_ylabel('Height (um)')
        ax.set_xlabel('Voltage (vrms)')
        fig_all.savefig(fname+'-Alldata-'+'.pdf', bbox_inches = 'tight')
        #%%
        # To remove the 1st cycle 
        if not tripod_fit:
            xd = x[:-1] - x[1:]
            xd[xd< -0.01 ] =  -1
            xd[xd>  0.01 ] = 1
            
            for i in range(1): # To get the 0-30 (range(3)), 0-45 (range(2)),  0-60 (range(1))
                y = y[xd.argmax():]
                x = x[xd.argmax():]
                xd = xd[xd.argmax():]
    
                y = y[xd.argmin():]
                x = x[xd.argmin():]
                xd = xd[xd.argmin():]
            # To get ONLY the last cycle
            
            print(xd.argmax())
            #xd = xd[xd.argmax():]
            #
            
    
    
            for i in range(1):
                #y = y[:xd.argmax()]
                #x = x[:xd.argmax()]
                idx1 = xd.argmax()
                xd = xd[xd.argmax():]
                print(xd.argmin())
                y = y[:(xd.argmin()+idx1)]
                x = x[:(xd.argmin()+idx1)]
                xd = xd[:(xd.argmin()+idx1)]    

        
        
        print('xMaxafter', max(x))
        print('len x data: ', len(x))
        # ---------------------------

        b = 0
        x_l = []
        y_l = []

        x2 = x[:-1].copy()
        x2[x[:-1] > x[1:]] = -x2[x[:-1] > x[1:]]
        x_data_all = np.append(np.array([0]),x2)
        
        #----------------------
        '''
        # This is for today remove later
        U_OFFSET = 5
        #path = '2024.01.19 10.57.53.avi_IDS_0_2Vs.csv'
        #path = '2024.01.22 08.37.28.avi_IDS_0_1Vs.csv'
        #path = '2024.01.25 13.00.07.avi_IDS_0_1Vs.csv'
        #path = '2024.01.29 11.25.49.avi_IDS_0_1Vs.csv'
        #path = '2024.01.29 15.22.07.avi_IDS_0_1Vs.csv'
        path = '2024.01.30 10.56.45.avi_IDS_0_1Vs.csv'
        
        df = pd.read_csv(path,skiprows=2,skipinitialspace=True)
        h = df.iloc[:,12].to_numpy().flatten()
        h_max_t = max(h)
        t = df.iloc[:,1].to_numpy().flatten()
        u = np.loadtxt('u_val_1VoltSec60Vmax.txt')
        
        u = u-U_OFFSET
        u[u<0] = 0
        t_sig = np.arange(len(u))/1000

        u_resamp = np.interp(t, t_sig, u)
        N = 849*2 + 1 # 849
        x_data_all = u_resamp[N:]
        x_data_all[np.gradient(x_data_all)<0] = -x_data_all[np.gradient(x_data_all)<0]
        y = h[N:]
        plt.plot()
        '''
        # ----------------
        
        Ns = 100
        flag_lmfit = 1
        model = 1
        x_data = x_data_all[::Ns]
        print(f'len x data: {len(x_data)}')
        y_data = y[::Ns]
        fig, ax = plt.subplots(figsize=(3.3,2.5))
        ax.plot(x_data, y_data, label='Data')
        x = U_rms[:len(real_data)]
        y = real_data
        #ax.plot(x, y, label='Data_orig')
        fig.legend()
        plt.show()
        #plt.show()
        print('x len: ', len(x_data))
        #%%
        # Fitting 
        if model == 0:
            bounds = ((1.2, 2, 1e-7, 95),(2, 5, 1e-3,  105))
            p0 = (1.88, 3.3, 0.00026, 99.2)
            popt, pcov = curve_fit(theta_func, x_data, y_data, p0 = p0
                                  ,bounds = bounds, xtol = 1e-8, method= 'trf', verbose = 2, diff_step = 1e-3)
            
        elif model == 1:
            print('HERE')
            bounds = ((1.2, 2, 95,95, -0.000001 ,-1),(2.5, 4, 105,105,0.000001,2))
            p0 = (1.88, 3.33, 100 , 98, 0.0000,0.6)
            # Fitting using lmfit
            if flag_lmfit:
                fit_params = create_params(C_g=dict(value=0.482, max = 0.55, min=0.40, vary = False),
                                           C_g2=dict(value=0.45, max=0.52, min=0.40, vary = False),    #expr='C_g'                                           #C_g2=dict(value=1.1, max=30.1, min=0.1, vary=False),
                                           theta0=dict(value=98.68, max=100, min=97, vary=False),
                                           #theta0=dict( expr= 'acos(-cpin/0.0648+cos(98.05*pi/180))*180/pi'),
                                           theta0r=dict(expr='theta0', max=103, min=50, vary=False),
                                           C1=dict(value=0.0, max=1.1e1, min=-1.8e-1, vary=False),
                                           cpin=dict(value=0.86e-3, max=1.1e-3, min=0.8e-3, vary=True),
                                           cpin2=dict(expr='cpin', max=1e-2, min=1e-5, vary=False)
                                           #cpin2=dict(value=7.14e-4, max=1e-3, min=1e-5, vary=True)
                                           )
                
                fit_params = create_params(C_g=dict(value=1.06, max = 1.1, min=1.0, vary = False),
                                           C_g2=dict(value=1.5, max=3.52, min=0.40, vary = False),    #expr='C_g'                                           #C_g2=dict(value=1.1, max=30.1, min=0.1, vary=False),
                                           theta0=dict(value=100.3, max=102, min=92, vary=False),
                                           #theta0=dict( expr= 'acos(-cpin/0.0648+cos(98.05*pi/180))*180/pi'),
                                           theta0r=dict(expr='theta0', max=103, min=50, vary=False),
                                           C1=dict(value=0.0, max=1.1e1, min=-1.8e-1, vary=False),
                                           cpin=dict(value=2.0e-3, max=1.1e-3, min=0.1e-3, vary=False),
                                           cpin2=dict(expr='cpin', max=1e-2, min=1e-5, vary=False)
                                           #cpin2=dict(value=7.14e-4, max=1e-3, min=1e-5, vary=True)
                                           )
                
                fit_flag = False
                fig, ax = plt.subplots(figsize=(3.3,2.5))
                ax.plot(np.abs(x_data), y_data, '.', label='data for model')
                if fit_flag == True:
                    print('Start fitting')
                    fitter = Minimizer(theta_func0_lmfit, fit_params, fcn_args=(x_data, V, d, gamma_lg, eps_d, fname, y_data))
                    #fitter = Minimizer(h_func0_lmfit, fit_params, fcn_args=(x_data, V, d, gamma_lg, eps_d, fname, y_data))
                    t = time.time()
                    out = fitter.minimize(method='brute', Ns=6, keep=2, workers=-1)
                    print('Fitting time: ', time.time()-t)
                    #out.show_candidates()
                        

                    #out = minimize(theta_func0_lmfit, fit_params, args=(x_data, V, d, gamma_lg, eps_d, fname), kws={'data': y_data}, max_nfev=50, method='brute')
                    print('fitted')
                    print(fit_report(out))
                    fit = theta_func0_lmfit(out.params, x_data, V, d, gamma_lg, eps_d, fname)
                    #fit = h_func0_lmfit(out.params, x_data, V, d, gamma_lg, eps_d, fname)
#                    fit0 = theta_func0_lmfit(fit_params, x_data, V, d, gamma_lg, eps_d, fname)

                    data_save = np.column_stack((x_data, y_data, fit))
                    np.savetxt(csv_name+'_dataFIT.txt',data_save)
                    with open (csv_name+'_paramsFIT.txt', 'w') as fp:
                        json.dump(out.params.valuesdict(), fp)
                    print(out.params)
                    print('Out fitted')
                else:
                    #x_data_extended = np.linspace(min(x_data),max(x_data),50)
                    x_data_extended = x_data
                    fit = theta_func0_lmfit(fit_params, x_data_extended, V, d, gamma_lg, eps_d, fname)
                    #fit = h_func0_lmfit(fit_params, x_data_extended, V, d, gamma_lg, eps_d, fname)

                if fit_flag==True:
                    ax.plot(x_data, fit, '*', label='fit')
                    ax.legend()
                    fig.savefig('Fit_asdasd'+fname+'Model_'+str(model)+'_'+'.pdf', bbox_inches = 'tight')
                    fig.show()

                    fig, ax = plt.subplots(figsize=(3.3,2.5))
                    ax.plot(fit, label='fit')
#                    plt.plot(fit0, label='no min')
                    ax.plot(y_data,'.', label='data')
            #        plt.xlabel('Voltage (Vrms)')
                    ax.set_ylabel('Contact angle (deg)')
                    fig.savefig('Fit_asdasd2'+fname+'Model_'+str(model)+'_'+'.pdf', bbox_inches = 'tight')
                    fig.legend()
                    fig.show()
                #plt.plot(x_data_extended, theta_func0_lmfit(out.params,x_data_extended, V,d, gamma_lg, eps_d,fname=fname),label='model')
                if fit_flag==False: 
                    #np.savetxt('fit_val'+csv_name+'.txt',fit)
                    #np.savetxt('u_val'+csv_name+'.txt',x_data_extended)
                    
                    #textstr = textstr + '\n' + r'$\theta_{da}$ =%5.2f , $\theta_{dr}$ =%5.2f ' % tuple(popt[4:])
#                    textstr = textstr + '\n' + r'$\theta_{da}$ =%5.2f , $\theta_{dr}$ =%5.2f ' % tuple(popt[4:])
 #                   plt.text(0.03, 0.05, textstr, transform=ax.transAxes, fontsize=8,
  #                              verticalalignment='bottom', bbox=props)
                    b, a = scipy.signal.iirfilter(4, Wn=0.9, fs=15, btype="low", ftype="butter") # Cutt off 2.5 Hz
                    #fit = scipy.signal.filtfilt(b, a, fit)
                    plt.plot(x_data_extended, fit,'.',label='model',markersize=2)
                    data_save = np.column_stack((x_data_extended, y_data, fit))
                    with open (csv_name+'_paramsFIT.txt', 'w') as fp:
                        json.dump(fit_params.valuesdict(), fp)
                    np.savetxt(csv_name+'_dataFIT.txt',data_save)

                    plt.legend()
                    plt.show()

 #                   plt.plot(fit, label='fit')
#                    plt.plot(fit0, label='no min')
#                    plt.plot(y_data, label='data')
            #        plt.xlabel('Voltage (Vrms)')
#                    plt.ylabel('Contact angle (deg)')
#                    plt.legend()
#                    plt.show()
            else:
                popt, pcov = curve_fit(theta_func0, x_data, y_data, p0 = p0
                                      ,bounds = bounds, xtol = 1e-4, method= 'trf', verbose = 2, diff_step = 1e-1)
        if flag_lmfit:
            y_tf = fit
            if fit_flag==False:
                popt = list(fit_params.valuesdict().values())
                popt_d = fit_params.valuesdict()
            else:
                popt = list(out.params.valuesdict().values())
                popt_d = out.params.valuesdict()
        else:
            if model == 1:
                Cg, Cg_r, t0,tr, c1, cpin = popt
                theta0r = theta0
            elif model ==0:
               Cg, Cg_r, theta0, theta0r, dta,dtr = popt
               cpin = 0
            act = EH_Actuator(V, gamma_lg, theta0, d, eps_d, theta0r = theta0r
                         ,C_g = Cg, C_g2 = Cg_r, cpin = cpin, uth1 = 1
                         , uth2 = 120, model=1, C1=c1)    # Create actuator w/param
            fname_csv = 'r_gf_' +fname+'.csv'
            data = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
            act.load_table(data)
            find_uth1_uth2(act,np.max(x_data))

            print('Uth1= ', act.uth1)
            print('Uth2= ', act.uth2)

        fig, ax = plt.subplots(figsize=(3,2.5))
        if not flag_lmfit:
            if model == 1:
                y_tf =  theta_func(x_data, *popt)
            elif model == 0:
                y_tf =  theta_func0(x_data, *popt)

        #plt.plot(np.abs(x_data), y_tf, 'r--', label='fit',markersize=4,zorder=10)
        #plt.scatter(np.abs(x_data), y_data, marker='o',label='data', c='k', 
        #            s=15, alpha =0.3,linewidths=0.5)
        r2 = r2_score(y_data, y_tf) 
        plt.plot(np.abs(x_data), y_tf, '--',color='0',
                 label=fr'fit ($R^2={r2:1.3f})$',markersize=4,zorder=10, linewidth=1)
        plt.scatter(np.abs(x_data), y_data, marker='o',label='Exp. data', color='0.6',
                    s=20, alpha=1 ,linewidths=0.5)
        plt.legend()
        props = dict(boxstyle='round', facecolor='white', alpha=0.5, linewidth = 0.5)
        textstr = ''
        print(popt_d)
        if flag_lmfit:
            i=1
            for k in popt_d:
                if k == 'theta0':
                    textstr += r'$\theta_{0}$'+ '='+r'%5.2f' % popt_d[k] + ' '
                #elif k == 'theta0r':    
                    #textstr += r'$\theta_{0r}$'+ '='+r'%5.2f' % popt_d[k] + ' '
                elif k == 'cpin':    
                    textstr += r'${a}$'.format(a=str(k))+ '='+r'%5.2e' % popt_d[k] + ' '
                if (i % 3) == 0:
                    textstr += '\n'
                i = i+1
            #textstr = r'Cg=%5.2f, Cgr=%5.2f' % tuple(popt[:2])
            #textstr = textstr + '\n' + r'$\theta_{0a}$=%5.2f, $\theta_{0r}$=%5.2f' % tuple(popt[2:4])
            #textstr = textstr + '\n' + r'$\theta_{da}$ =%5.2f , $\theta_{dr}$ =%5.2f ' % tuple(popt[4:])
        else: 
            if model == 1:
                textstr = r'Cg=%5.2f, Cgr=%5.2f' % tuple(popt[:2])
                textstr = textstr + '\n' + r'cpin=%.2e, $\theta_0$=%5.2f' % tuple(popt[2:])
            elif model == 0:
                textstr = r'Cg=%5.2f, Cgr=%5.2f' % tuple(popt[:2])
                textstr = textstr + '\n' + r'$\theta_{0a}$=%5.2f, $\theta_{0r}$=%5.2f' % tuple(popt[2:4])
                textstr = textstr + '\n' + r'$\theta_{da}$ =%5.2f , $\theta_{dr}$ =%5.2f ' % tuple(popt[4:])

     #   plt.text(0.03, 0.05, textstr, transform=ax.transAxes, fontsize=8,
      #          verticalalignment='bottom', bbox=props)
        
        plt.xlabel('Voltage ($V_{rms}$)')
        #plt.ylabel(r'Equivalent Height $h^\prime$ ($\mu$m)')
        plt.ylabel('Contact angle (deg)')
        plt.legend(loc='upper right',fontsize=7)
        plt.savefig('Fit_'+fname+'Model_'+str(model)+'_'+'.pdf', bbox_inches = 'tight')
        
        
        fig, ax = plt.subplots(figsize=(3,2.5))
        error_data = y_tf-y_data 
        plt.plot(x_data[x_data>=0], error_data[x_data>=0], '.',label='advancing',markersize=1)
        plt.plot(np.abs(x_data[x_data<0]), error_data[x_data<0], '.',label='receding',markersize=1)
        print('Mean |error|: ', np.mean(np.abs(error_data)))
        
        MSE = np.sum(error_data**2) /np.size(error_data)
        print('MSE: ', MSE)
        textstr = r'std error  = %.2f (um)' % np.sqrt(MSE)
        plt.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', bbox=props)
        plt.xlabel('Voltage (Vrms)')
        plt.ylabel('Error in height (um)')
        plt.legend()
        plt.savefig('Error_fit_'+fname+'Model_'+str(model)+'_'+'.pdf', bbox_inches = 'tight')

        plt.show()


    act = EH_Actuator(V, gamma_lg, theta_0, d, C_g, eps_d)
    if flag_all_curves:
        act2 = EH_Actuator(V, gamma_lg, theta_0_2, d, C_g2, eps_d)
    
    if flag_create_table:
        create_table(ddir,fname,V)

    if flag_stable_wtable:
        d_theta = 5
        dt = 0.01
        theta_x = theta_0
        fname_csv = 'r_gf_' +fname+'.csv'
        data = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
        act.load_table(data)
        U = np.linspace(0,U_max,N_U)
        U2 = U[U<u3]
        theta_stable = []
        h_stable = []
        E_surf = []
        E_U = []
        if flag_all_curves:
            theta_x_2 = theta_0
            theta_stable_2 = []
            h_stable_2 = []
        for u_step in U[0:]:
            ranges = (slice(theta_x-d_theta, theta_x+d_theta, dt),)
            theta_x = brute_parallel(act.f_ene_from_table,80,105,0.01,
                                                args=(u_step))
            #theta_x = brute(act.f_ene_from_table,ranges,
            #                                    args=(data,u_step))
            #            theta_x = minimize_scalar(act.f_ene_from_table,(75,110),
            #                          args=(act.table, u_step), tol = 1e-14, method = 'Golden').x

            #E_sval, E_uval = act.f_ene_from_table(theta_x,data,u_step,SplitEne=True)
            E_sval, E_uval = act.f_ene_from_table(theta_x,u_step,SplitEne=True)
            E_surf.append(E_sval)
            E_U.append(E_uval)
            theta_stable.append(theta_x+theta_shift)
            h_stable.append(act.h_calc(theta_stable[-1]))
            if flag_all_curves:
                if u_step < u3:
                    ranges = (slice(theta_x_2-d_theta, theta_x_2+d_theta, dt),)
                    #theta_x_2 = brute(act2.f_ene_from_table,ranges,
                            #                                    args=(data,u_step))
                    theta_x_2 = minimize_scalar(act.f_ene_from_table,(75,110),
                                    args=(u_step), tol = 1e-14, method = 'Golden').x
                    theta_stable_2.append(theta_x_2+theta_shift2)
                    h_stable_2.append(act2.h_calc(theta_stable[-1]))

            print('Calculated one theta... Theta diff_evolution:  ',theta_x)
        
        fig, ax = plt.subplots(figsize=(3,2.5))
        color_line1 = cycle[0]
        #ax.plot(U[U>u1]+U_shift,theta_stable[U>u1]+np.ones_like(theta_stable[U>u1])*theta_shift,'--',label='Simulation',zorder=2, color=color_line1)
        ax.plot(U,theta_stable,'--',label='Simulation',zorder=2, color=color_line1)
        #idx = U>u1 if flag_all_curves else U>-1
        #idx = np.array(theta_stable).flatten() < (theta_0_2+theta_shift2)
        #x = U[idx]+U_shift
        #y = np.array(theta_stable)[idx]
        #ax.plot(x, y,'--',label='Simulation',zorder=2, color=color_line1)

        if flag_all_curves:
            x = U[U>=u3]
            y = (theta_stable_2[-1]-theta_stable[-1])/(u3-u2)*(x-u3)+theta_stable_2[-1]
            ax.plot(U2+U_shift,theta_stable_2,'--',zorder=2, color=color_line1)
            ax.hlines(theta_0_2+theta_shift2,0,x[0],linestyles='--',zorder=2)    #,alpha=0.6,linewidth=0.5)
#            ax.hlines(theta_0,0,u1,linestyles='--',alpha=0.6,linewidth=0.5)
            ax.plot(x, y,'--',zorder=2, color=color_line1)
        if flag_read_csv:
            print(' Read_csv')
            ax.plot(U_rms[:len(real_data)],real_data,'--',label='Experiment', zorder=1,markersize=2, color=cycle[1])
        
#        stairs = findstairs(V=V, W=10, G=2, n0 = 50, nf = 100)
#        res = [func(l) for l in theta_stable for func in (min, max)] 
#        idx = np.where((stairs > min(res)-0.1)&(stairs < max(res)+0.1))
#        stairs = np.array(stairs)[idx]
#        ax.hlines(stairs,0,U_max,linestyles='--',alpha=0.6,linewidth=0.5)
        ax.set_xlabel('Voltage (Vrms)')
        ax.set_ylabel('Contact angle (deg)',color=color_line1)
        
        textstr = '; '.join((
            r'$C_g=%.3f$  ' % (C_g, ),
            r'$V=%.2f$ (uL) ' % (V*1e9, ),
            r'$d =%.2f$ (um) ' % (d*1e6, )))
        textstr = textstr + '\n'
        textstr = textstr+'; '.join((
            r'$\theta_0 =%.1f$ (deg)' % (theta_0, ),
            r'$\theta_d =%.2f$ (deg)' % (theta_shift, ),
            r'$\gamma_{lg}=%.2f$ ($g/s^2$)' % (gamma_lg*1000, )))
        if flag_all_curves:
            textstr = textstr + '\n'
            textstr = textstr+'; '.join((
                r'$\theta_0^2 =%.1f$ (deg)' % (theta_0_2, ),
                r'$\theta_d^2 =%.2f$ (deg)' % (theta_shift2, ),
                r'$C_g^2=%.2f$ ($g/s^2$)' % (C_g2, )))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.5, linewidth = 0.5)

        # place a text box in upper left in axes coords
        #ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=8,
        #        verticalalignment='top', bbox=props)
        if flag_plot_height:
            ax12 = ax.twinx()
            color_line2 = cycle[3]
            ax12.plot(U,np.array(h_stable)*1e6,'--+',color=color_line2)
            ax12.set_ylabel('Height (um)',color=color_line2)
            ax12.tick_params(axis='y',labelcolor=color_line2)

        ax.legend()
        plt.title(textstr,fontsize=8)
        plt.savefig('Energy_model_from_table'+fname+'_'+str(C_g)+'U_shift_'+str(U_shift)+'.pdf', bbox_inches = 'tight')
        # Surf_ener vs. U energy
        fig2, ax2 = plt.subplots(figsize=(4,4))
        idx = np.array(theta_stable).flatten() < (theta_0_2+theta_shift2)
        x = U.flatten()
        y2_surf = np.array(E_surf).flatten()
#        y2_surf = np.array(E_surf)[idx].flatten()
#        y2_U = np.array(E_U)[idx].flatten()
        y2_U = np.array(E_U).flatten()
        ax2.plot(x, y2_surf,label='E_surf',zorder=2, color=color_line1)
        ax2.plot(x, y2_U,label='E_u',zorder=2, color=color_line1)
        ax2.set_ylabel('Energy')
        ax2.set_xlabel('Voltage (Vrms)')
        ax2.legend()
        plt.savefig('EMFT_Energy_'+fname+'_'+str(C_g)+'U_shift_'+str(U_shift)+'.pdf', bbox_inches = 'tight')

        fig3, ax3 = plt.subplots(figsize=(4,4))
        #ax3 = ax2.twinx()
        ax3.plot(x[1:],np.diff(y2_U),label='d E_u')
        ax3.plot(x[1:],np.diff(y2_surf),label='d E_surf')
        ax3.set_ylabel('Change in Energy')
        ax3.set_xlabel('Voltage (Vrms)')
        ax3.legend()
        plt.savefig('EMFT_dEnergy_'+fname+'_'+str(C_g)+'U_shift_'+str(U_shift)+'.pdf', bbox_inches = 'tight')
        plt.show()

       
    if flag_min_theta:
        fname_csv = 'r_gf_' +fname+'.csv'
        data = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
        U_static = 70
        thetas = np.linspace(70,100,300)
        ene_values = []
        for th in thetas:
            ene_values.append(act.f_ene_from_table(th,data,U_static))
        idx = ene_values.index(min(ene_values))
        fig, ax = plt.subplots(figsize=(4,4))
        ax.plot(thetas[idx-50:idx+50],ene_values[idx-50:idx+50],'--*')
        ax.set_xlabel('Contact angle')
        ax.set_ylabel('Energy')
        plt.title(f'Voltage= {U_static} V')
        plt.savefig('Ene_fix_theta_'+fname+'.pdf', bbox_inches = 'tight')
        plt.show()

    if flag_theta_stable == 1:
        d_theta = 3
        dt = 0.1
        theta_x = 100 - d_theta
        U = np.linspace(0,U_max,40)
        theta_stable = []
        h_stable = []
        #theta_1 = brentq(f_dEne,100,115,args=(Afunc, 0, V))
        # res = minimize(f_ene,theta_1,args=(Afunc,0,V,gamma_sl,gamma_lg),method='COBYLA') #'Nelder-Mead')  #disp=False)
        #res = differential_evolution(f_ene,[[60,115]],args=(Afunc,0,V,gamma_sl,gamma_lg, 2,d))
        #theta_x = res.x
        ranges = (slice(theta_x-d_theta, theta_x+d_theta, dt),)
        theta_x = brute(f_ene,ranges,args=(Afunc,0,V,gamma_sl,gamma_lg, 2,d))
        #theta_stable.append(brentq(f_dEne,100,115,args=(Afunc, 0, V)))
        theta_stable.append(theta_x)
        h_stable.append(h_calc(V,theta_stable[-1]))
        print('Calculated one theta...: ', theta_stable[-1])
        for u_step in U[1:]:
            #theta_1 = brentq(f_dEne,60,115,args=(Afunc, u_step, V))
            #res = minimize(f_ene,theta_1,args=(Afunc,u_step,V,gamma_sl,gamma_lg, 2,d))#,method='Nelder-Mead')  #disp=False)
            #theta_x = res.x
            #theta_stable.append(theta_x)
            #print('Calculated one theta... Theta:  ',theta_x)
            #res = differential_evolution(f_ene,[[60,115]],args=(Afunc,u_step,V,gamma_sl,gamma_lg, 2,d))
            #,method='Nelder-Mead')  #disp=False)
            #theta_x = res.x
            ranges = (slice(theta_x-d_theta, theta_x+d_theta, dt),)
            theta_x = brute(f_ene,ranges,args=(Afunc,u_step,V,gamma_sl,gamma_lg, 2,d))
            theta_stable.append(theta_x)
            h_stable.append(h_calc(V,theta_stable[-1]))
            print('Calculated one theta... Theta diff_evolution:  ',theta_x)
        print(theta_stable)
        
        fig, ax = plt.subplots(figsize=(4,4))
        color_line1 = cycle[0]
        ax.plot(U,theta_stable,'--*')

        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Contact angle (deg)',color=color_line1)
        
        textstr = '; '.join((
            r'$C_g=%.3f$  ' % (C_g, ),
            r'$V=%.2f$ (uL) ' % (V*1e9, ),
            r'$d =%.2f$ (um)' % (d*1e6, ),
            r'$\gamma_{lg}=%.2f$ ($g/s^2$)' % (gamma_lg*1000, )))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.5, linewidth = 0.5)

        # place a text box in upper left in axes coords
        #ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=8,
        #        verticalalignment='top', bbox=props)
        ax2 = ax.twinx()
        color_line2 = cycle[3]
        ax2.plot(U,np.array(h_stable)*1e6,'--+',color=color_line2)
        ax2.set_ylabel('Height (um)',color=color_line2)
        ax2.tick_params(axis='y',labelcolor=color_line2)
        plt.title(textstr,fontsize=8)
        plt.savefig('Energy_model_'+fname+'.pdf', bbox_inches = 'tight')
        plt.show()


