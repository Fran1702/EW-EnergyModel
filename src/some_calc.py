import numpy as np
from ene_model_faux import *

V = 2e-9
t0 = 105
t1 = 80
r0 = (3*V/np.pi*np.sin(t0*np.pi/180)**3/(f_theta(t0)))**(1/3)
r1 = (3*V/np.pi*np.sin(t1*np.pi/180)**3/(f_theta(t1)))**(1/3)
print(r0*1e6)
print(r1*1e6)
print('Delta ', (r1-r0)*1e6)

