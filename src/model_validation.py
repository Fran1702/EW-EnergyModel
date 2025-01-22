import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import matplotlib.ticker as mtick
import imageio
from stabfunc import *
from funaux import *
from ene_model_faux import *
import matplotlib as mpl
import time

import scienceplots
plt.style.use(['science'])
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'figure.dpi':'100'})

mpl.rcParams['legend.frameon'] = 'True'
mpl.rcParams['legend.facecolor'] = 'w'

os.chdir("../data")

# Validation of the model using the electrowetting equation
# cos(theta) = cos(theta_0) + epsilon0*eps_d U^2/(2*gamma*d)

# Actuator parameters
V = 4.2e-9 # 0.8 uL in m3 3.1 (90)
gamma_lg = 64.8/1000 # g/s^2 from microkat.gr
d = 0.5e-6
eps_d = 2
eps_0 = 8.85418782*1e-12  # [F/m]
C_g = 1
C_g2 = C_g
theta0 = 100
theta0r = theta0
C1 = -0.000
cpin = 0.0
model = 1
d_theta = 5
dt = 0.001
theta_x = theta0
fname_csv = 'r_gf_3438-Annular-W20-G10-3490um.png.csv'
data_table = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
#print(data_table.shape)
U = np.linspace(0,40,100)

act = EH_Actuator(V, gamma_lg, theta0, d, eps_d, theta0r = theta0r, C1=C1
                 ,C_g = C_g, C_g2 = C_g2, cpin = cpin, cpin2= cpin, uth1 = 1
                 , uth2 = 120, model=model)    # Create actuator w/params
act.C1 = C1
act.load_table(data_table)
#act.uth1 = 0
#find_uth1_uth2(act, 65, model)

#%%
#C1 = 0.04
#act.C1 = .01
#act.uth1 = 0

U = np.linspace(20,50,50)
#U = np.linspace(0,0,2
# Theta calculus

theta_l = theta_func0(U, C_g, C_g2, theta0, theta0r, 0, 0, act=act, data=data_table)
theta_l = np.array(theta_l)
chi = 0.08e-3   # Coefficient of contact line friction [Kg/(s.m)] From paper not glyucerine
eta = 5e-4 # Kinematic viscosity of glycerine m2/s
# Y-L equation
theta_YL = np.arccos(np.cos(theta0*np.pi/180)+eps_0*eps_d*U**2/(2*d*gamma_lg))*180/np.pi
def calc_theoretical(U,gamma_lg, cpin, eps_0, eps_d,chi,eta, theta0):
    val = []
    for u in U:
        if u>=0:
            v = np.arccos(np.cos(theta0*np.pi/180)-cpin/gamma_lg-C1*eps_0*eps_d/(2*d)*(chi+6*eta)/(2*eta*gamma_lg)*np.abs(u)**2+eps_0*eps_d*np.abs(u)**2/(2*d*gamma_lg))*180/np.pi
        else:
            v = np.arccos(np.cos(theta0*np.pi/180)+cpin/gamma_lg+C1*eps_0*eps_d/(2*d)*(chi+6*eta)/(2*eta*gamma_lg)*np.abs(u)**2+eps_0*eps_d*np.abs(u)**2/(2*d*gamma_lg))*180/np.pi
        val.append(v)
    return np.array(val).flatten()

theoretical = calc_theoretical(U,gamma_lg, cpin, eps_0, eps_d,chi,eta, theta0)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(3.3,4))
#ax[0].plot(U, theta_YL, label='Young-Lipmann')
#ax[0].plot(U, theoretical, label='Theoretical')
ax[0].plot(U, theta_l, label='Model')
ax[1].plot(U, 100*(theoretical-theta_l)/theoretical, label='Error (\%)')
ax[0].set_ylabel('Contact angle (deg)')
ax[1].set_ylabel('Error (\%)')
ax[1].set_xlabel('Voltage (Vrms)')
ax[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
plt.show()
#%%
U = np.linspace(0,50,50)
theta_l = theta_func0(U, C_g, C_g2, theta0, theta0r, 0, 0, act=act, data=data_table)
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(3.3,4))

ax.plot(U, theta_l, label='Model')
ax.set_ylabel('Contact angle (deg)')
ax.set_xlabel('Voltage (Vrms)')
ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)

#%%
### Energy analysis

# Actuator parameters
V = 3.65e-9 # 0.8 uL in m3 3.1 (90)
gamma_lg = 64.8/1000 # g/s^2 from microkat.gr
d = 0.5e-6
eps_d = 2
eps_0 = 8.85418782*1e-12  # [F/m]
C_g = 0.89
C_g2 = 1.1
theta0 = 99.1
theta0r = theta0
C1 = 1e-3
cpin = 5e-3
model = 1
d_theta = 5
dt = 0.001
theta_x = theta0
fname_csv = 'r_gf_3438-Annular-W20-G10-3490um.png_d1130.csv'
data_table = np.genfromtxt(fname_csv, delimiter =',',dtype=None)

#print(data_table.shape)


act = EH_Actuator(V, gamma_lg, theta0, d, eps_d, theta0r = theta0r
                 ,C_g = C_g, C_g2 = C_g2, cpin = cpin, cpin2=cpin ,uth1 = 1
                 , uth2 = 120, model=model)    # Create actuator w/params
act.C1 = C1
act.load_table(data_table)
find_uth1_uth2(act, 65, model)

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
ax.plot(act.table[:,0],act.table[:,1])
plt.show()

#%%
U = np.linspace(35,35,1)

theta_arr = np.linspace(98,100,100)

act.C1 = 0.0000001
# 0.004183187133024244
#act.gamma_sl_calc_fmin(100,act.gamma_sl, er=1e-3, N=300)
#act.gamma_sl_calc_fmin(100,0.004183187133024244, er=1e-3, N=100)
data_d = {}
E_f_l = []
for u in U:
    l_aux = []
    for t_val in theta_arr:
        Esurf, Epin, Eelec, Efric, Etot = act.f_ene_ft_wpinning(t_val,u, split=True)
        #l_aux.append([t_val, Esurf, np.abs(Epin), np.abs(Eelec), np.abs(Efric), Etot])
        l_aux.append([t_val, Esurf, Epin, Eelec, Efric, Etot])
    data_d[u] = np.array(l_aux)
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
for key in list(data_d.keys()):
    #print(data_d[key][:,1])
    ax.plot(data_d[key][:,0], (data_d[key][:,1]-np.min(data_d[key][:,1]))/(np.max(data_d[key][:,1])-np.min(data_d[key][:,1])),'--', label='Surface energy')
    ax.plot(data_d[key][:,0], (data_d[key][:,2]-np.min(data_d[key][:,2]))/(np.max(data_d[key][:,2])-np.min(data_d[key][:,2])) ,'--', label='Pinning energy')
    ax.plot(data_d[key][:,0], (data_d[key][:,3]-np.min(data_d[key][:,3]))/(np.max(data_d[key][:,3])-np.min(data_d[key][:,3])),'-.', label='Elec energy')
    #ax.plot(data_d[key][:,0], data_d[key][:,4]-min(data_d[key][:,4]),'--', label='Friction energy')
    #ax.plot(data_d[key][:,0], data_d[key][:,4]/(np.max(data_d[key][:,1])-np.min(data_d[key][:,4]),'--', label='Friction energy')
    ax.plot(data_d[key][:,0], (data_d[key][:,5]-np.min(data_d[key][:,5]))/(np.max(data_d[key][:,5])-np.min(data_d[key][:,5]))-np.min(data_d[key][:,1]), label=f'Total {key} V')
    #E_aux =  data_d[key][:,1] + data_d[key][:,3]+ data_d[key][:,2]+ data_d[key][:,4]
    #ax.plot(data_d[key][:,0],E_aux-np.min(data_d[key][:,1]), label=key)
    arg = np.argmin(data_d[key][:,1]+data_d[key][:,3])
    E_f_l.append(data_d[key][arg,4])
    print(data_d[key][arg,0])
    #ax.scatter(data_d[key][arg,0], data_d[key][arg,1]+data_d[key][arg,3])
ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
plt.show()

#%%
fname_csv = 'r_gf_1000-rb2.png.csv'
data_table = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
theta_arr = np.linspace(80,100,100)
rl = []
#A_sl_l = []
#for t_val in theta_arr:
#    rl.append(act.r_calc(t_val))
#    A_sl_l.append(np.pi*rl[-1]**2)
    
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
ax.plot(data_table[:,0],data_table[:,1]*1e-12, label='Gf')
#ax.plot(np.array(rl)*1e6, A_sl_l, label='Droplet')
r = np.arange(1100,1450,100)*1e-6

ax.plot(r*1e6, , label='Droplet2')
ax.legend()

plt.show()
U = np.linspace(0,0.1,2)
theta_arr = np.linspace(94,102,100)
for u in U:
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    l_aux = []
    Etot = []
    for t_val in theta_arr:
        E = act.f_ene_ft_wpinning(t_val,u, split=False)
        Etot.append(E)
    Earr = np.array(Etot)
    ax.plot(theta_arr, Earr, label='Electrical energy')
    plt.show()
