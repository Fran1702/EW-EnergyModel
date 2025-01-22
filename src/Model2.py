#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:39:39 2023

@author: hector.ortiz
"""
#%%
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
from scipy.optimize import fsolve
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

#%%
os.chdir("../data")
datadir = os.getcwd()
#fname,ddir = select_file(datadir)

csv_name = 'static-hyst-loops_IDS.csv'
sign_name = 'input - static-hyst-loops.txt'
#    if file.endswith('.csv'):
#        if file.startswith('G'):
#            csv_name = file
            
path = os.path.normpath(csv_name)
df = pd.read_csv(path,skiprows=2,skipinitialspace=True)
global real_data, U_rms
#real_data = df.iloc[:,10].to_numpy().flatten()
real_data_t = df.iloc[:,10].to_numpy().flatten()
real_data = df.iloc[:,12].to_numpy().flatten()#*1280/1284.5#*0.981
real_data_r = df.iloc[:,11].to_numpy().flatten()#*1280/1284.5#*0.981
real_data_v = df.iloc[:,14].to_numpy().flatten()#*0.981**3
'''
signal_Obj = sg.Signal.fromparams(csv_name[:-9]+'params_'+csv_name[-5]+'.txt')
signal = np.array(signal_Obj.signal[0]).flatten()
signal = signal[::200]
signal_rms = window_rms(signal,500)
U_rms = signal_rms[::int(len(signal)/len(real_data))]
'''
sig = np.loadtxt(sign_name)
U_rms = sig[:,0][::200]

#%% Pre treatment data
x = U_rms[:len(real_data)]
print('xMax', max(x))
print('len x data: ', len(x))
y = real_data

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
print('xMaxafter', max(x))
print('len x data: ', len(x))
# --------------------------- Advancing and receding period
x2 = x[:-1].copy()
x2[x[:-1] > x[1:]] = -x2[x[:-1] > x[1:]]
x_data_all = np.append(x2,[0])
y_adv = y[x_data_all>=0]
x_adv = x_data_all[x_data_all>=0]
y_rec = y[x_data_all<0]
x_rec = np.abs(x_data_all[x_data_all<0])

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True,figsize=(3.3,2))
#ax.plot(x_data_all, y,'--')
ax.plot(np.arange(len(x))/6, x,'--')
#ax.plot(x_rec, y_rec,'.')
ax.set_ylabel('Voltage (Vrms)')
ax.set_xlabel('Time (s)')
#ax[2].axhline(1190,0,60,c='b')
plt.show()
#%%
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True,figsize=(5.3,5))
#ax.plot(x_data_all, y,'--')
ax.plot(x_adv, y_adv,'k.')
ax.plot(x_rec, y_rec,'k.')
ax.set_xlabel('Voltage (Vrms)')
ax.set_ylabel('Height (um)')
#ax[2].axhline(1190,0,60,c='b')
plt.show()

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True,)
#ax[0].plot(U_rms[:len(real_data)],real_data,'--')
#ax[1].plot(U_rms[:len(real_data_t)],real_data_t,'--')
ax.plot(U_rms[:len(real_data_t)],real_data_r,'--')
x = np.array([0,65])
y = np.array([1,1])
for i in np.arange(39,43):
    ax.fill_between(x,((20+10)*i+20)*y, ((20+10)*i+0)*y,color='gold', alpha=0.3)
    #ax.fill_between(x,((20+10)*i+30)*y, ((20+10)*i+20)*y,color='r', alpha=0.2)
ax.set_xlabel('Voltage (Vrms)')
ax.set_ylabel('Contact radius (um)')
#ax[2].axhline(1190,0,60,c='b')
plt.show()

#%%
def r_calc(V,theta, shift):
    ''' Given the volume V and the contact angle theta, this function returns
        the value of the contact radius
    '''
    
    return (3*V/np.pi*np.sin(theta*np.pi/180)**3/f_theta(theta))**(1/3) - shift

class Model:

    def __init__(self, V,x0=0, e_width=20, e_gap=10):
        '''
        V: Droplet volume
        '''
        self.V = V
        self.x0 = x0
        self.e_width=e_width
        self.e_gap = e_gap
        self.calc_hpos()
        self.u_adv = 0
        self.h_adv = 0
        self.h_rec = 0
        #rint(self.h_eq)
        
    def fun_adv(self, u, *args):
        '''
        args is an array with the shape [h0, u0, u1,....un]
        h0 in (m)
        u is the voltage
        '''
        
        h0 = args[0]
            
        uarr = np.array(args[1:])
        if u>min(uarr):
            self.u_adv = uarr[uarr < u].max()
        elif self.u_adv ==0:
            self.u_adv = 0
        # I just use the values of the h_eq positions smaller than h0
        h_pos = self.h_eq[self.h_eq<(h0*0.98)]
        if u<uarr[0]:
            #if h0>self.h_adv:
            #self.h_adv = h0
            return h0
        elif u<uarr[1]:
            self.h_adv = h0+(h_pos[0]-h0)/(uarr[1]-uarr[0])*(u-uarr[0])
            if self.h_adv>=self.h_rec:
                return self.h_rec
            else:
                return self.h_adv
        else:
            for i in range(2,len(uarr)+1):
                if i==len(uarr)-1:
                    if u>uarr[i]:
                        self.h_adv = h_pos[i-1]
                        if self.h_adv>=self.h_rec:
                            return self.h_rec
                        else:
                            return self.h_adv
                    else:
                        self.h_adv =  h_pos[i-2]
                        if self.h_adv>=self.h_rec:
                            return self.h_rec
                        else:
                            return self.h_adv
                elif u<uarr[i]:
                    self.h_adv =  h_pos[i-2]
                    if self.h_adv>=self.h_rec:
                        return self.h_rec
                    else:
                        return self.h_adv
        
    def fun_rec(self, u, *args):
        '''
        args is an array with the shape [h0, du, uth, h1,u0,h2, h3,u1,h4,....]
        h0 in (m)
        u is the voltage
        '''      
        h0 = args[0]
        Ui = np.array(args[4::3])
        Uth = args[2]
        hi = np.array(args[3::3])
        hf = args[5::3]
        du = args[1]
        idx = hi[hi > self.h_adv]
        if len(idx)>0:
            u_rec_max = Ui[np.argmin(hi[hi > self.h_adv])]
            i_rmax, = np.where(Ui==u_rec_max)
            i_rmax = i_rmax[0]
        else:
            u_rec_max = 0
            i_rmax = 0
        if self.h_adv is not None:
            if u<=Uth:
                self.h_rec = h0
                return h0
            elif u<=Ui[0]:
                #if hi[0]<h0:
                if hi[0]>self.h_adv:
                    h = h0+(hi[0]-h0)/(Ui[0]-Uth)*(u-Uth)
                    self.h_rec = h
                else:
                    #h = h0+(hf[0]-h0)/(Ui[0]-Uth)*(u-Uth)
                    h = h0+(self.h_adv-h0)/(Ui[0]-Uth)*(u-Uth)
                    self.h_rec = h
                return h
            
            elif u >= (self.u_adv-du):
                self.h_rec = self.h_adv
                return self.h_adv
            
            elif u >= u_rec_max:
                #print('123')
               if u_rec_max == 0:
                   h = self.h_adv
                   self.h_rec = h
               else:
                   h = hi[i_rmax]+(self.h_adv-hi[i_rmax])/((self.u_adv-du)-u_rec_max)*(u-(u_rec_max))
                   self.h_rec = h
               return h
               #else:
              #      h = hi[j]+(self.h_adv-hi[j])/((self.u_adv-du)-Ui[j])*(u-(Ui[j]))
               #    returnh # hi[i_rmax-1]#+(self.h_adv-hi[i_rmax])/((self.u_adv-du)-u_rec_max)*(u-(u_rec_max))
            else:
                j= 0
                for ui in Ui:
                    if ui>=u:
                        i, = np.where(Ui==ui)
    #                    if i[0]==len(Ui):
                        if hi[i[0]] > self.h_adv:
                            h = hi[i[0]]-(hi[i[0]]-hf[i[0]-1])/(Ui[i[0]-1]-Ui[i[0]])*(u-Ui[i[0]])
                            self.h_rec = h
                            return h
                        else:
                            j = np.argmin(hi[hi > self.h_adv])
                            h = hi[j]+(self.h_adv-hi[j])/((self.u_adv-du)-Ui[j])*(u-(Ui[j]))
                            self.h_rec = h
                            return h
                        
                        
    def train_model(self, u_adv, y_adv, u_rec, y_rec, p0_adv, p0_rec, train=False):
        
        if train==True:
            
            popt_ad, pcov, d1,d2,d3 = curve_fit(np.vectorize(mod.fun_adv), u_adv, y_adv*1e-6,  p0=p0_adv,
                                                method='dogbox', full_output=True, gtol=1e-10)#, xtol=1e-8, ftol=1e-8)
            self.par_adv = popt_ad
            popt_rec, pcov, d1,d2,d3 = curve_fit(np.vectorize(mod.fun_rec), U_rec, y_rec*1e-6,  p0=p0_rec,
                                                 method='dogbox', full_output=True, gtol=1e-10)#, xtol=1e-12, ftol=1e-12)
            self.par_rec = popt_rec
        else:
            self.par_adv = p0_adv
            self.par_rec = p0_rec
            self.h_rec = p0_rec[0]
            self.h_0 = p0_adv[0]
            self.h_eq = mod.h_eq[(self.h_eq<self.h_0) & (self.h_eq> np.min(y_adv)*1e-6*0.98)]
            #h_min_dat = self.predict(np.max(u_adv), 0)
            #self.h_eq = mod.h_eq[(self.h_eq<self.h_0) & (self.h_eq>h_min_dat)]
            self.h_min = np.min(self.h_eq)
    
    def predict(self, u, u_act):
        ''' 
        u is the next value
        u_act is the actual value
        '''
        if u>=u_act:
            return self.fun_adv(u, *self.par_adv)
        elif u<u_act:
            return self.fun_rec(u, *self.par_rec)
    
    def reach_intervals(self):
        # TO continue some day
        # Advancing
        i_adv = []
        i_adv.append(np.array([self.h_0,self.h_eq[0]]))
        for h in self.h_eq:
            i_adv.append(np.array([h,h]))
        i_rec = []
        hi = self.par_rec[3::3]
        hf = self.par_rec[5::3]
        i_rec.append(np.array([self.h_0,hi[0]]))
        for j in range(len(hi)-1):
            i_rec.append(np.array([self.h_0,hi[0]]))
        
    def inverse(self, h_d):
        U_adv = np.arange(0,self.par_adv[-1],0.1)
        
        F = np.vectorize(mod.fun_adv)
        h = F(U_adv, *p0_adv)
        
        U_rec = np.arange(0,self.par_adv[-1],0.1)
        
        F_rec = np.vectorize(mod.fun_rec)
        h_rec = F_rec(U_rec, *p0_rec)
        h = np.stack
        return
        
    def calc_hpos(self,t0=60,tmax=120, hmin=None):
        '''
        It calculates the stables position (height)
        '''
        rmin = self.r_calc(tmax,0)*1e6
        #print(rmin)
        rmax = self.r_calc(t0,0)*1e6
        i0 = int((rmin-self.e_width)/(self.e_width+self.e_gap))
        im = int((rmax-self.e_width)/(self.e_width+self.e_gap))+1
        h_eq = []
        for i in np.arange(i0,im):
            r = (self.e_width+self.e_gap)*i+self.e_width
            #print(r)
            t = fsolve(self.r_calc, 100, args=(r*1e-6,))
            #print(t)
            h_eq.append(self.h_calc(t))
        if hmin is None:
            self.h_eq = np.array(h_eq).flatten()
        else: 
            self.h_eq = np.array(h_eq[h_eq>hmin]).flatten()
        return 
    
    def r_calc(self, theta, shift):
        ''' Given the volume V and the contact angle theta, this function returns
            the value of the contact radius
        '''
        #print(theta)
        
        return (3*self.V/np.pi*np.sin(theta*np.pi/180)**3/f_theta(theta))**(1/3)-shift
    
    def h_calc(self,theta):
        '''
            It returns the height of the droplet
        '''
        return (3*self.V/np.pi*(1-np.cos(theta*np.pi/180))**3/f_theta(theta))**(1/3)



#%% Advancing first guess

mod = Model(4.025e-9)

# Initialization
print(mod.h_eq)
h0 = np.max(y_adv)*1e-6
print(h0)
hmin = np.min(y_adv*1e-6)*0.98
N = len(mod.h_eq[(mod.h_eq<h0) & (mod.h_eq>hmin)])
print('N:', N)
print(mod.h_eq[(mod.h_eq<h0) & (mod.h_eq>hmin)])

u_arr = np.array([20,38,58])
p0_adv = np.concatenate(([h0],u_arr))

du = 5
uth = 7
u_arr = np.array([30,33,36,50])
print('u arr: ', u_arr)

#hi = mod.h_eq[(mod.h_eq<h0) & (mod.h_eq>hmin)] + 10e-6#.min()
#hf = mod.h_eq[(mod.h_eq<h0) & (mod.h_eq>hmin)] - 10e-6
#hi = np.array([1274, 1250,1220,1190])*1e-6
#hf = np.array([1263,1233,1202,1180])*1e-6

hi = np.array([1246,1228,1208,1190])*1e-6
hf = np.array([1234,1215,1202,1180])*1e-6
print(hi)
print(hf)
p0_rec = np.zeros(len(hi)+len(hf)+len(u_arr)+3)
p0_rec[:3] = [h0,du,uth]
p0_rec[3:len(hi)*3+3:3] = hi
p0_rec[4:len(u_arr)*3+4:3] = u_arr 
p0_rec[5::3] = hf

mod.train_model(x_adv, y_adv, x_rec, y_rec, p0_adv, p0_rec)

U_adv = x_adv

h_l = []


F = np.vectorize(mod.fun_adv)
h = F(U_adv, *p0_adv)
hmin = np.min(h)
fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(8,7))
axs.plot(U_adv,np.array(h)*1e6,'*')
axs.plot(U_adv,y_adv,'.')
#fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(8,7))
#axs.plot(U_adv,np.array(h)*1e6-y_adv,'.')

#Receding first guess
U_rec = x_rec
h_l = []
#h0 = 1280*1e-6

#fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(8,7))
    
F_rec = np.vectorize(mod.fun_rec)
h_rec = F_rec(U_rec, *p0_rec)
#fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(8,7))
axs.plot(U_rec,np.array(h_rec)*1e6,'*')
axs.plot(U_rec,y_rec,'.')

#% Minimization

mod.train_model(U_adv, y_adv, U_rec, y_rec, p0_adv, p0_rec)


#%%


U_adv = np.arange(0,65,0.001)

F_adv = np.vectorize(mod.fun_adv)
h_adv = F_adv(U_adv, *p0_adv)

U_rec = np.arange(65,0,-0.001)

F_rec = np.vectorize(mod.fun_rec)
h_rec = F_rec(U_rec, *p0_rec)

h = np.stack((h_adv,h_rec)).flatten()
U = np.stack((U_adv,U_rec)).flatten()


#%% Signal generation starts here

#h_d = 1260
h_sp = (mod.h_0 - mod.h_min )*np.random.random_sample(20) + mod.h_min
#%%
np.savetxt('randomvals-h_sp.txt',h_sp)
#%%
h_sp = np.loadtxt('randomvals-h_sp.txt')
fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(3.3,4))
axs.plot(U,h*1e6,'.')
fs = 30
Ts = 5
Tst = 1
U_signal = []
U_sig = np.zeros(len(h_sp)*fs*Ts)
h_sig = np.zeros(len(h_sp)*fs*Ts)
hd_sig = np.zeros(len(h_sp)*fs*Ts)
sp_sig = np.zeros(len(h_sp)*fs*Ts)
U_signal.append(0)
h_m = []
i=0
print(h_sp)
for h_d in h_sp.copy():
    print(h_d)
    if h_d > mod.h_eq[1]:
        e = np.abs(h[0:len(U_adv)]-h_d)
    else:
        e = np.abs(h-h_d)
    idx_min = np.argmin(e)
    
    
    U_sv = U[idx_min]
    U_act = U_sv
    U_sig[i*fs*Ts:fs*Ts*(i+1)] = U_sv
    sp_sig[i*fs*Ts:fs*Ts*(i+1)] = h_d
    h_sig[i*fs*Ts:fs*Ts*(i+1)] = h[idx_min]
    hd_sig[i*fs*Ts:fs*Ts*(i+1)] = h_d
    if U_signal[-1] < U_sv:
        j = 0
        while 1:
            U_s1 = mod.par_adv[j]
            if F_adv(U_s1,*p0_adv) <= h[idx_min]:
                break
            j = j+1
            if j == len(mod.par_adv):
                #U_s1 = 65
                break
        if U_s1 == np.max(mod.par_adv):
            U_s2 = 65
        else:
            U_s2 = mod.par_adv[mod.par_adv > U_s1].min()
        U_s = (U_s1+U_s2)/2
        U_signal.append(U_s)
        h_s = F_adv(U_s,*p0_adv)
        h_m.append(h_s)
        U_sig[i*fs*Ts:fs*Ts*i+fs*Tst] = U_s
        h_sig[i*fs*Ts:fs*Ts*i+fs*Tst] = h_s
        h_sp[i*fs*Ts:fs*Ts*i+fs*Tst] = h_s
        sp_sig[i*fs*Ts:fs*Ts*i+fs*Tst] = h_s
    else:
        if U_sv<mod.par_adv[2]:
            U_sig[i*fs*Ts:fs*Ts*i+fs*Tst] = 0
            h_sig[i*fs*Ts:fs*Ts*i+fs*Tst] = mod.h_0
            sp_sig[i*fs*Ts:fs*Ts*i+fs*Tst] = mod.h_0
    U_signal.append(U_sv)        
    i = i+1
    h_m.append(h[idx_min])
    axs.plot(U[idx_min],h[idx_min]*1e6,'r*')
    axs.hlines(h_d*1e6,0,65,linestyles='--')
    
    #%%
fig, axs = plt.subplots(nrows=3,ncols=1, sharex=True, figsize=(3.3,4))
t = np.arange(0,len(U_sig))/fs
axs[0].plot(t,h_sig*1e6,'--')

axs[0].plot(t,sp_sig*1e6)
axs[1].plot(t,h_sig*1e6-sp_sig*1e6)
axs[2].plot(t,U_sig,'--')

f = 1000
t = 10
N = len(U_sig)
sig = U_sig
signals = np.column_stack((sig,sig,sig))
np.savetxt('input.txt',signals)

#%%
fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(3.3,4))

for UM in [20,30,45,55,65]:
    u_test = np.arange(0,UM,0.1)
    y_predict = []
    for i in range(len(u_test)-1):
        y_predict.append(mod.predict(u_test[i+1], u_test[i]))
    l1, = axs.plot(u_test[1::],np.array(y_predict)*1e6,'--', zorder=10,color='r')
    u_test = u_test[::-1]
    y_predict = []
    for i in range(len(u_test)-1):
        y_predict.append(mod.predict(u_test[i+1], u_test[i]))
    l1, = axs.plot(u_test[1::],np.array(y_predict)*1e6,'--', zorder=10,color='r')
l2, = axs.plot(U_rec,y_rec,'.',color='k')
l2, = axs.plot(U_adv,y_adv,'.',color='k')
axs.legend([l2,l1],['data','model'])
axs.set_xlabel('Voltage (Vrms)')
axs.set_ylabel('Height (um)')
fig.savefig('Model-linear.pdf', bbox_inches = 'tight')
plt.show()


#%% Testing cicles
fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(3.3,4))


u_test = np.arange(0,65,0.1)
y_predict = []
for i in range(len(u_test)-1):
    y_predict.append(mod.predict(u_test[i+1], u_test[i]))
l1, = axs.plot(u_test[1::],np.array(y_predict)*1e6,'--', zorder=10,color='r')

U_m = 30
u_test = np.arange(65,U_m,-0.1)
y_predict = []
for i in range(len(u_test)-1):
    y_predict.append(mod.predict(u_test[i+1], u_test[i]))
l1, = axs.plot(u_test[1::],np.array(y_predict)*1e6,'--', zorder=10,color='r')

u_test = np.arange(U_m,65,0.1)
y_predict = []
for i in range(len(u_test)-1):
    y_predict.append(mod.predict(u_test[i+1], u_test[i]))
l1, = axs.plot(u_test[1::],np.array(y_predict)*1e6,'--', zorder=10,color='r')

u_test = np.arange(65,0,-0.1)
y_predict = []
for i in range(len(u_test)-1):
    y_predict.append(mod.predict(u_test[i+1], u_test[i]))
l1, = axs.plot(u_test[1::],np.array(y_predict)*1e6,'--', zorder=10,color='r')

l2, = axs.plot(U_rec,y_rec,'.',color='k')
l2, = axs.plot(U_adv,y_adv,'.',color='k')
axs.legend([l2,l1],['data','model'])
axs.set_xlabel('Voltage (Vrms)')
axs.set_ylabel('Height (um)')
fig.savefig('Model-linear.pdf', bbox_inches = 'tight')
plt.show()

#%%
# Advancing
#h0 = 1350*1e-6
#p0 = np.concatenate(([h0],u_arr))

popt_ad, pcov, d1,d2,d3 = curve_fit(np.vectorize(mod.fun_adv), U_adv, y_adv*1e-6,  p0=p0_adv,
                                 method='dogbox', full_output=True, gtol=1e-10)#, xtol=1e-8, ftol=1e-8)

print(popt_ad)
fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(8,7))
axs.plot(U_adv,y_adv, '.')
axs.plot(U_adv, np.vectorize(mod.fun_adv)(U_adv, *popt_ad)*1e6, '.')

# Receding

popt_rec, pcov, d1,d2,d3 = curve_fit(np.vectorize(mod.fun_rec), U_rec, y_rec*1e-6,  p0=p0_rec,
                                 method='dogbox', full_output=True, gtol=1e-10)#, xtol=1e-12, ftol=1e-12)

print(popt_rec)
#fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(8,7))
axs.plot(U_rec, y_rec, '.')
axs.plot(U_rec, np.vectorize(mod.fun_rec)(U_rec, *popt_rec)*1e6, '.')
#%% Testing
U = np.linspace(0,9,250)
h_l = []
print(mod.h_eq)
h0 = 1240*1e-6
hmin = 1100*1e-6
N = len(mod.h_eq[(mod.h_eq<h0) & (mod.h_eq>hmin)])
u_arr = np.arange(1,N+1)
p0 = np.concatenate(([h0],u_arr))
F = np.vectorize(mod.fun_adv)
h = F(U, *p0)
hmin = np.min(h)

#print(mod.h_eq[(mod.h_eq<h0) & (mod.h_eq>hmin)])

axs.plot(U,np.array(h)*1e6,'.-')
#%%
h_l = []
#h0 = 1280*1e-6
du = 0.5
uth = 0.4
u_arr = np.arange(1,N-1)-0.2
print('u arr: ', u_arr)
#hi = mod.h_eq[(mod.h_eq +10e-6<h0) & (mod.h_eq +10e-6>hmin)] + 10e-6#.min()
hi = mod.h_eq[(mod.h_eq<h0) & (mod.h_eq>hmin)] + 10e-6#.min()
#hf = mod.h_eq[(mod.h_eq -10e-6<h0) & (mod.h_eq>hmin +10e-6)] - 10e-6
hf = mod.h_eq[(mod.h_eq<h0) & (mod.h_eq>hmin)] - 10e-6
print(hi)
print(hf)
p0 = np.zeros(len(hi)+len(hf)+len(u_arr)+3)
p0[:3] = [h0,du,uth]
p0[3:len(hi)*3+3:3] = hi
p0[4:len(u_arr)*3+4:3] = u_arr 
p0[5::3] = hf
fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(8,7))
for u in U:
    haux=mod.fun_rec(u, *p0)
    #print(h)
    h_l.append(haux)
    
axs.plot(U,np.array(h_l)*1e6)
axs.plot(U,np.array(h)*1e6,'.-')
#%% Add noise to signal
hdata = np.array(h)
hdata = hdata + np.random.normal(0,0.5*np.max(hdata)/1000,len(hdata))
hd_rec = np.array(h_l)
hd_rec = hd_rec + np.random.normal(0,0.5*np.max(hd_rec)/1000,len(hd_rec))

fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(8,7))
axs.plot(U,hdata)
axs.plot(U,hd_rec)
#%% Minimization
# Advancing
u_arr = np.arange(1,N+1)
#h0 = 1350*1e-6
p0 = np.concatenate(([h0],u_arr))

popt_ad, pcov, d1,d2,d3 = curve_fit(np.vectorize(mod.fun_adv), U, hdata,  p0=p0*0.99,
                                 method='dogbox', full_output=True, gtol=1e-10)#, xtol=1e-8, ftol=1e-8)

print(popt_ad)
fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(8,7))
axs.plot(U,hdata)
axs.plot(U, np.vectorize(mod.fun_adv)(U, *popt_ad), 'g--')


# Receding
#u_arr = np.arange(1,N+1)
u_arr = np.arange(1,N-1)-0.2
p0 = np.zeros(len(hi)+len(hf)+len(u_arr)+3)
p0[:3] = [h0,du,uth]
p0[3:len(hi)*3+3:3] = hi.copy()
p0[4:len(u_arr)*3+4:3] = u_arr.copy() 
p0[5::3] = hf.copy()
print(p0)
popt_rec, pcov, d1,d2,d3 = curve_fit(np.vectorize(mod.fun_rec), U, hd_rec,  p0=p0*0.995,
                                 method='dogbox', full_output=True, gtol=1e-10)#, xtol=1e-12, ftol=1e-12)

print(popt_rec)
#fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(8,7))
axs.plot(U,hd_rec)
axs.plot(U, np.vectorize(mod.fun_rec)(U, *popt_rec), 'g--')