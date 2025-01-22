# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:21:07 2022

@author: hector.ortiz
"""

#from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os
import funaux



class Signal:
	"""
	A class used to represent a signal

	Attributes
	----------
	Vmax : TYPE, optional
		Max tension value. The default is 1.
	slope : TYPE, optional
		Slope of the signal in volt/s. The default is 1.
	t_step : TYPE, optional
		Time of each step. The default is 1.
	N : TYPE, optional
		NUMBER OF STEPS FROM 0 TO VMAX, ADD 1 TO INCLUDE 0. The default is 1.
	freqSin : TYPE, optional
		SINE FREQUENCY. The default is 1.
	f_sampling : TYPE, optional
		Sampling frequency. The default is 1.
	sig : List with signal or signals
	t : Time array
	Methods
	----------

	"""
	def __init__(self, filename = 'default_params', vmax = 1, q_signals = 1, slope = 1, t_step = 1, N=10, freq_sin = 1, f_sampling = 10, phase = [0],Ncycles=1):
		"""
		Parameters
		----------
		Vmax : TYPE, optional
			Max tension value. The default is 1.
		q_signals: TYPE, optional
			Number of signals generated
		slope : TYPE, optional
			Slope of the signal in volt/s. The default is 1.
		t_step : TYPE, optional
			Time of each step. The default is 1.
		N : TYPE, optional
			NUMBER OF STEPS FROM 0 TO VMAX, ADD 1 TO INCLUDE 0. The default is 1.
		freqSin : TYPE, optional
			SINE FREQUENCY. The default is 1.
		f_sampling : TYPE, optional
			Sampling frequency. The default is 1.
		phase: signal phase shifting in degrees. The default is 0
		"""
		self.vmax = vmax
		self.q_signals = q_signals
		self.phase = phase
		self.slope = slope
		self.t_step = t_step
		self.N = N
		self.freq_sin = freq_sin
		self.f_sampling = f_sampling
		self.filename = filename
		self.Ncycles = Ncycles
        
		self.gen_filename()
#		self.write_params()
		self.signal = []
		self.t = None
		self.signal_generator_sine()	
	
	def gen_filename(self):
		filename = self.filename
		if os.path.splitext(filename)[1] != '.txt':
			filename = filename +'.txt'
		self.filename = funaux.file_unique(filename)

	def print_params(self):
		i = 0
		partxt = ['1- V max [V]','2- Number of signals' , '3- Phases', '4- Slope [V/s]', '5- Dead time [s]',
				'6- Number of steps', '7- Freq. sin', '8- Freq. samp','9- Signal params filename','10-Ncycles']
		for val,ptxt in zip(self.__dict__.values(),partxt):
			print(ptxt, '=', val)

	def signal_generator_sine(self):
		'''
		Signal generator. Generates a sine signal with variable amplitude

		Parameters
		----------
		Vmax : TYPE, optional
		Max tension value. The default is 1.
		slope : TYPE, optional
		Slope of the signal in volt/s. The default is 1.
		t_step : TYPE, optional
		Time of each step. The default is 1.
		N : TYPE, optional
		NUMBER OF STEPS FROM 0 TO VMAX, ADD 1 TO INCLUDE 0. The default is 1.
		freqSin : TYPE, optional
		SINE FREQUENCY. The default is 1.
		f_sampling : TYPE, optional
		Sampling frequency. The default is 1.

		Returns
		-------
		signal : numpy array
		signal vector.
		t : numpy array
		time vector.

		'''

		amp = self.amp_generator()
		self.rms_values = amp.copy()*2**0.5
		#Vmax = Vmax, slope = slope, t_step = t_step, N = N, f_sampling = f_sampling)
		n_samp = len(amp)
		t = np.linspace(0, n_samp/self.f_sampling, n_samp)
		self.t = t
		for i in range(int(self.q_signals)):
			values = amp*np.sin(2*np.pi*self.freq_sin*t + np.pi*int(self.phase[i])/180.0 )
			self.signal.append(values)
		return #values, t

	def amp_generator(self):
	#Vmax = 1, slope = 10, t_step = 1, N = 11, f_sampling = 1000):
		"""
		Amplitude generation for stair signal, ramp with a given slope and t
		Arguments: 
			  Vmax: Max tension value
			  slope: Slope of the signal in volt/s
			  t_step: Time of each step
			  N: Number of steps from 0  to Vmax, +1 to include 0
			  f_sampling: Sampling frequency
		Returns:
			  Amplitude generated
		"""
		steps = np.linspace(0, self.vmax, self.N)
		steps = np.append(steps,np.flip(steps[1:-1]))    
		steps = np.tile(steps,self.Ncycles)        
		steps = np.append(steps,0)
		amp = np.array([])
		for i in range(len(steps)-1):
			slope = np.linspace(0,steps[i+1]-steps[i],int(steps[1]*self.f_sampling/self.slope))
			amp = np.concatenate((amp,slope+steps[i]))
			amp = np.concatenate((amp,steps[i+1]*np.ones(int(self.t_step*self.f_sampling))))
		return amp

	def write_params(self):
		"""
		Write params to a .txt file
		"""
		with open(self.filename, 'w') as f:        
			for key, value in self.__dict__.items():
				if key not in ['signal','t','m', 'c', 'filenames']:
					if type(value) == int:
						valuestr = "%d" % value
					elif type(value) == float:
						valuestr = "%.2f" % value
					else:
						valuestr = "%s" % repr(value)
	 
					f.write("%s = %s\n" % (key, valuestr))
			f.close()
	@classmethod
	def fromparams(cls, filename):
		"""
		Create an instance with params from a file
		"""
		params = {}
		with open(filename) as f:
			exec(f.read(), params)
		del params['__builtins__']
		
		return cls(**params)

	def set_param(self, param, value):
		"""
		Set the param with value 
		Arguments:
			param: Number reference of the param to change
		               '1- V max [V]','2- Number of signals' , '3- Phases', '4- Slope [V/s]', '5- Dead time [s]', 6- Number of steps', '7- Freq. sin', '8- Freq. samp','9- Signal params filename']
		"""
		if param == 1:
			self.vmax = int(value)
		elif param == 4:
			self.slope = int(value)
		elif param == 5:
			self.t_step = int(value)
		elif param == 6:
			self.N = int(value)
		elif param == 7:
			self.freq_sin = int(value)
		elif param == 8:
			self.f_sampling = int(value)
		elif param == 9:
			self.filename = value
			self.gen_filename()
		
		self.write_params()
		self.signal_generator_sine()	

	def plot_signals(self):
		for i in range(len(self.signal)):
			plt.plot(self.t,np.array(self.signal[i]))
		plt.show()



if __name__ == '__main__':

	freq = 5
	f_sampling = 100
	#t_end = 2
	vmax = 300
	slope = 200 # V/s
	t_step = 1 # seg
	N = 11
	fname = 'ABC'
	#sig = Signal(filename = fname ,vmax = vmax, q_signals = 2,
	#		slope = slope,t_step = t_step, N = N, freq_sin=freq, 
	#		f_sampling = f_sampling, phase = [0,180])
	sig = Signal.fromparams('default_params.txt')
	sig.print_params()
	sig.write_params()
	t = sig.t
	signal = sig.signal
	for i in range(len(signal)):
		plt.plot(t,np.array(signal[i]))
	
	plt.show()
