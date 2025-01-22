import numpy as np
from multiprocessing.pool import ThreadPool
'''
name changes:
 fun: r_calc -> contact_rad_calc

'''


def min_scal_parallel(fun, value, ran, args,brute_delta=3, brute_step=0.01):
    t0 =  minimize_scalar(fun, ran, args=(value,args), tol = 1e-14).x
    t0 = brute_parallel(fun, t0-brute_delta, t0+brute_delta, brute_step, args=(value,args))
    return t0

def diff_central(fun, x_value, delta):
    ''' Returns derivative using central differences '''
    return (fun(x_value+delta)-fun(x_value-delta))/(2.0*delta)

def brute_parallel(func, ang_min, ang_max, ang_step, args):
    tx = np.arange(ang_min,ang_max,ang_step)
    res_l = []
    with ThreadPool() as pool:
        if np.size(args)==1:
            items = [(tx[i], args) for i in range(len(tx))]
        else: 
            items = [(tx[i], *args) for i in range(len(tx))]
        for result in pool.starmap(func,items):
            res_l.append(result)
    res_arr = np.vstack(res_l).flatten()

    return tx[np.argmin(res_arr)]

def contact_rad_calc(V,theta):
    ''' Given the volume V and the contact angle theta, this function returns
        the value of the contact radius
    '''
    return (3*V/np.pi*np.sin(theta*np.pi/180)**3/f_theta(theta))**(1/3)

def h_calc(V, theta):
        '''
            It returns the height of the droplet
        '''
    return (3*V/np.pi*(1-np.cos(theta*np.pi/180))**3/f_theta(theta))**(1/3)


def f_theta(theta):
    ''' Theta in degrees
    # This function returns the value of the function
    # f(theta) = (2+cos(theta))*(1-cos(theta))^2
    '''
    return (2+np.cos(theta*np.pi/180))*(1-np.cos(theta*np.pi/180))**2


def f1_theta(theta):
    ''' Theta in degrees
    This function returns the value of the function
    f1(theta) = (1-cos(theta))/f(theta)^(2/3)
    '''
    return (1-np.cos(theta*np.pi/180))/(f_theta(theta))**(2/3)


def f2_theta(theta):
    """Theta in degrees
    This function returns the value of the function
    f2(theta) = (sin^2(theta))/f(theta)^(2/3)
    """
    return (np.sin(theta*np.pi/180)**2)/(f_theta(theta))**(2/3)



