import os
import numpy as np
from scipy.optimize import fsolve

def file_unique(filename):

	i = 0
	file_name, file_extension = os.path.splitext(filename)
	while os.path.exists(f'{file_name}_{i}{file_extension}'):
		i +=1

	return f'{file_name}_{i}{file_extension}'


def EW_stairs(theta,Rb,V):
	return np.sin(theta*np.pi/180)*(3*V/(np.pi*(2+np.cos(theta*np.pi/180))*(1-np.cos(theta*np.pi/180))**2))**(1/3)-Rb


def EW_eq(U,theta_0 = 100,K=0.00001,U_th=1):
	y0 = theta_0
	y = np.arccos(np.cos(theta_0*np.pi/180) + K*(U-U_th)**2)*180/np.pi
	y[U<U_th] = y0
	y[U<41] = y0
	return y

def LinTh_eq(x, C = 0.1 , K=0.00001,B=1):
	y0 = C
	y = K*x+B
	return np.maximum(y0,y)

def findstairs(V = 0.8e-9 ,W = 10, G = 10, n0 = 15, nf = 50):
	Rb = [((n-1)*G+n*W)*1e-6 for n in range(n0,nf)]
	theta0 = 90
	theta_min = 50
	theta_max = 120
	f = 100
	sol = [fsolve(EW_stairs,theta0, args=(Rb_val,V), factor=f) for Rb_val in Rb]
	return sol

def window_rms(a, window_size):
  a2 = np.power(a,2)
  window = np.ones(window_size)/float(window_size)
  return np.sqrt(np.convolve(a2, window, 'same'))

def extractProp(fname):
    pArr = fname.split('-')
    prop = {
        'fluid':pArr[0],
        'Eshape':pArr[1]+'-'+pArr[2],
        'freq':pArr[3],
        'signal':pArr[4]
        }
    return prop
if __name__ == '__main__':

	fn = file_unique('AS.AD.txt')
	print(fn)
