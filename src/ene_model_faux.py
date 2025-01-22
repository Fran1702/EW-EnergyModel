import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from scipy.optimize import minimize_scalar
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import cv2
from stabfunc import *
import time
import scipy.signal


#C_g = 1.8

def theta_func(U, C_g, C_g2, cpin, theta0, model = 1):
    global V, gamma_lg, eps_d, fname
    d_theta = 5
    dt = 0.001
    theta_x = theta_0
    fname_csv = 'r_gf_' +fname+'.csv'
    data = np.genfromtxt(fname_csv, delimiter =',',dtype=None)

    act = EH_Actuator(V, gamma_lg, theta0, d, eps_d
                     ,C_g = C_g, C_g2 = C_g2, cpin = cpin, uth1 = 1
                     , uth2 = 120, model=1)    # Create actuator w/params
    act.load_table(data)
    find_uth1_uth2(act, np.max(U),  model)
    theta_l = []
    with ThreadPool() as pool:
        items = [(U[i],act, model) for i in range(len(U))]
        for result in pool.starmap(calc_thetax_U,items):
            theta_l.append(result)
    theta_l = np.vstack(theta_l).flatten()
    return theta_l

def theta_func0(U, C_g, C_g2, theta0, theta0r, dt0a, dt0r, act = None, data=None):
    global V, gamma_lg, eps_d, fname
    model = 1
    d_theta = 5
    dt = 0.01
    theta_x = theta0
    if data is None:
        print('data None')
        fname_csv = 'r_gf_' +fname+'.csv'
        data = np.genfromtxt(fname_csv, delimiter =',',dtype=None)

    if act == None:
        print('act None')
        act = EH_Actuator(V, gamma_lg, theta0, d, eps_d, theta0r = theta0r
                         ,C_g = C_g, C_g2 = C_g2, cpin = 0, uth1 = 1
                         , uth2 = 120, model=1)    # Create actuator w/params
        act.load_table(data)
        find_uth1_uth2(act, np.max(U),  model)

    theta_l = []
    with ThreadPool() as pool:
        items = [(U[i],act, model,dt0a,dt0r) for i in range(len(U))]
        for result in pool.starmap(calc_thetax_U,items):
            theta_l.append(result)
    theta_l = np.vstack(theta_l).flatten()
    return theta_l

def inv_theta(act, x, advancing=None, data=None):
    Theta = x
    '''
    C_g = params['C_g'].value
    C_g2 = params['C_g2'].value
    theta0 = params['theta0'].value
    theta0r = params['theta0r'].value
    if 'dt0a' in params:    
        dt0a = params['dt0a'].value
    if 'dt0r' in params:    
        dt0r = params['dt0r'].value
    if 'C1' in params:    
        C1 = params['C1'].value
    cpin = params['cpin'].value
    print(params.valuesdict().values())
    d_theta = 5
    dt = 0.001
    theta_x = theta0
    fname_csv = 'r_gf_' + fname+'.csv'
    data_table = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
    print(data_table.shape)
    act = EH_Actuator(V, gamma_lg, theta0, d, eps_d, theta0r = theta0r
                     ,C_g = C_g, C_g2 = C_g2, cpin = cpin, uth1 = 1
                     , uth2 = 120, model=model)    # Create actuator w/params
    act.C1 = C1
    act.load_table(data_table)
    a,b, = find_uth1_uth2(act, 70.0,  model)
    print('utmin: ', a)
    print('utmax: ', b)
    '''
    if advancing is None:
        advarr = np.ones_like(Theta)
    else:
        advarr = advancing
    print('uth1', act.uth1)
    print('uth2', act.uth2)
    print('uthmax', act.umax)
    print('C1', act.C1)
    print('Cpin', act.cpin)
#    print('theta', theta)
    model = 1
    U_l = []
    ran_theta = (50,105)
    ranU = []
    for i in range(len(Theta)):
        if advarr[i] == 1:
            ran = (act.uth1, act.umax)
        elif advarr[i] == 0:
            ran = (-act.uth2,0)
        ranU.append(ran)

    with ThreadPool() as pool:
        items = [(ranU[i],Theta[i],act, ran_theta, model) for i in range(len(Theta))]
        for result in pool.starmap(min_error,items):
            U_l.append(np.abs(result))
    U_l = np.vstack(U_l).flatten()
    if data is None:
        return U_l
    print('data: ', data)
    return (U_l - data)


def h_func0_lmfit(params, x, V, d, gamma_lg, eps_d, fname, data=None):
    U = x
    C_g = params['C_g'].value
    C_g2 = params['C_g2'].value
    theta0 = params['theta0'].value
    theta0r = params['theta0r'].value
    if 'dt0a' in params:    
        dt0a = params['dt0a'].value
    if 'dt0r' in params:    
        dt0r = params['dt0r'].value
    if 'C1' in params:    
        C1 = params['C1'].value
    cpin = params['cpin'].value
    cpin2 = params['cpin2'].value
    print(params.valuesdict().values())
    model = 1
    d_theta = 5
    dt = 0.001
    theta_x = theta0
    fname_csv = 'r_gf_' + fname+'.csv'
    data_table = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
    #print(data_table.shape)
    act = EH_Actuator(V, gamma_lg, theta0, d, eps_d, theta0r = theta0r,C1=C1
                     ,C_g = C_g, C_g2 = C_g2, cpin = cpin, cpin2=cpin2,
                     uth1 = 1, uth2 = 120, model=model)    # Create actuator w/params
    act.C1 = C1
    act.load_table(data_table)
    a,b, = find_uth1_uth2(act, np.max(U),  model)
    print('utmin: ', a)
    print('utmax: ', b)
    theta_l = []
    with ThreadPool() as pool:
        if 'dt0a' and 'dtor' in params:    
            items = [(U[i],act, model,dt0a,dt0r) for i in range(len(U))]
        else:
            items = [(U[i],act, model) for i in range(len(U))]
        for result in pool.starmap(calc_thetax_U,items):
            theta_l.append(result)
    theta_l = np.vstack(theta_l).flatten()
    h_l = 1e+6*(3*V/np.pi/f_theta(theta_l))**(1/3)*(1-np.cos(theta_l*np.pi/180))
    if data is None:
        return h_l
    return (h_l - data)


def theta_func0_lmfit(params, x, V, d, gamma_lg, eps_d, fname, data=None):
    U = x
    C_g = params['C_g'].value
    C_g2 = params['C_g2'].value
    theta0 = params['theta0'].value
    theta0r = params['theta0r'].value
    if 'dt0a' in params:    
        dt0a = params['dt0a'].value
    if 'dt0r' in params:    
        dt0r = params['dt0r'].value
    if 'C1' in params:    
        C1 = params['C1'].value
    cpin = params['cpin'].value
    cpin2 = params['cpin2'].value
    print(params.valuesdict().values())
    model = 1
    d_theta = 5
    dt = 0.001
    theta_x = theta0
    fname_csv = 'r_gf_' + fname+'.csv'
    data_table = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
    #print(data_table.shape)
    act = EH_Actuator(V, gamma_lg, theta0, d, eps_d, theta0r = theta0r
                     ,C_g = C_g, C_g2 = C_g2, cpin = cpin, cpin2=cpin2,
                     uth1 = 1, uth2 = 120, model=model)    # Create actuator w/params
    act.C1 = C1
    act.load_table(data_table)
    a,b, = find_uth1_uth2(act, np.max(U),  model)
    print('utmin: ', a)
    print('utmax: ', b)
    theta_l = []
    with ThreadPool() as pool:
        if 'dt0a' and 'dtor' in params:    
            items = [(U[i],act, model,dt0a,dt0r) for i in range(len(U))]
        else:
            items = [(U[i],act, model) for i in range(len(U))]
        for result in pool.starmap(calc_thetax_U,items):
            theta_l.append(result)
    theta_l = np.vstack(theta_l).flatten()
    if data is None:
        return theta_l
    return (theta_l - data)

def calc_thetax_U(Uval, act, model, dt0a=0,dt0r=0):
    dt = 5
    ds = 0.02
    #print('U: ',Uval)
    
    if 0 <= float(Uval) and float(Uval) < float(act.uth1):
        #    Uth = minimize_scalar(error_uth, bounds=ran, args=(t0,act,ran_theta, model), method='bounded',
            #                        options={'xatol':1e-5,'disp':0}).x
        t_g = minimize_scalar(act.f_ene_ft_wpinning,bounds=(70,105), args=(-0.001,model),
                              options={'xatol':1e-5,'disp':0}).x
        
     #                                      args=(-0.001,model), tol = 1e-5, method = 'Golden').x
    #    t_g = minimize_scalar(act.f_ene_ft_wpinning,(75,110),
     #                                      args=(-0.001,model), tol = 1e-5, method = 'Golden').x
        
        theta_x = brute_parallel(act.f_ene_ft_wpinning,
                                 t_g-dt, t_g+dt, ds,args=(-0.001,model))

    elif float(Uval) < -1*float(act.uth2):
        t_g = minimize_scalar(act.f_ene_ft_wpinning,bounds=(70,105), args=(act.umax,model),
                              options={'xatol':1e-5,'disp':0}).x
        #t_g = minimize_scalar(act.f_ene_ft_wpinning,(75,110),
        #                                   args=(act.umax,model), tol = 1e-5, method = 'Golden').x
        theta_x = brute_parallel(act.f_ene_ft_wpinning,
                                 t_g-dt, t_g+dt, ds,args=(act.umax,model))

    else:
        t_g = minimize_scalar(act.f_ene_ft_wpinning,bounds=(70,105), args=(Uval,model),
                              options={'xatol':1e-6,'disp':0}).x
        #t_g = minimize_scalar(act.f_ene_ft_wpinning,(75,110),
        #                                   args=(Uval,model), tol = 1e-5, method = 'Golden').x
        theta_x = brute_parallel(act.f_ene_ft_wpinning, 
                                 t_g-dt, t_g+dt, ds,args=(Uval,model))
    if Uval >= 0:
        theta_x = theta_x #+ dt0a
    elif Uval < 0:    
        theta_x = theta_x #+ dt0r
    return theta_x

def brute_parallel(func, ang_min, ang_max, ang_step, args):
    tx = np.arange(ang_min,ang_max,ang_step)
    #print('ang_min: ', ang_min)
    #print('ang_max: ', ang_max)
    res_l = []
    with ThreadPool() as pool:
        if np.size(args)==1:
            items = [(tx[i], args) for i in range(len(tx))]
        else: 
            items = [(tx[i], *args) for i in range(len(tx))]

        for result in pool.starmap(func,items):
            res_l.append(result)
    res_arr = np.vstack(res_l).flatten()
    #fig, ax = plt.subplots(figsize=(3.3,2.5))
    #ax.plot(tx, res_arr)
    #plt.show()
#    print('resss: ', res_arr)    
    argm = np.argmin(res_arr)
    if argm == len(res_arr)-1 or argm == 0 :
        print('Warning min at the extreme')
    return tx[np.argmin(res_arr)]

def error_uth(U, real_value, act, ran_theta,  model=1):
#def error_uth(U, real_value, act, ran_theta1, ran_theta2,  model=1):
    #    ran_theta = (ran_theta1, ran_theta2)
    
    t =  minimize_scalar(act.f_ene_ft_wpinning,bounds=ran_theta, method='bounded',
                                args=(U,model), options={'xatol':1e-5,'disp':0}).x
    act_value = brute_parallel(act.f_ene_ft_wpinning, 
                             t-3.5, t+3.5, 0.01, args=(U,model))
    err = np.abs(act_value - real_value)
    return err

def error_uth2(U, real_value, act, ran_theta1, ran_theta2,  model=1):
    ran_theta = (ran_theta1, ran_theta2)
    t =  minimize_scalar(act.f_ene_ft_wpinning,bounds=ran_theta, method='bounded',
                                args=(U,model), options={'xatol':1e-5,'disp':0}).x
    act_value = brute_parallel(act.f_ene_ft_wpinning, 
                             t-3.5, t+3.5, 0.02, args=(U,model))
    err = np.abs(act_value - real_value)
    return err

data_d = {}
def min_error(ran, t0, act, ran_theta, model):
    #    t =  minimize(act.f_ene_ft_wpinning,x0=(ran_theta[0]+ran_theta[1])/2,bounds=[ran_theta],
    #                            args=(U,model), tol = 1e-5).x
    Uth = minimize_scalar(error_uth, bounds=ran, args=(t0,act,ran_theta, model), method='bounded',
                        options={'xatol':1e-5,'disp':0}).x
    Uth = brute_parallel(error_uth2, Uth-10,Uth+10,0.1,
                        args=(t0,act,ran_theta[0],ran_theta[1], model))
    
    return Uth

def find_uth1_uth2(act, U_max, model=1):
    t = time.time()
    act.umax = U_max
    
    t0 = minimize_scalar(act.f_ene_ft_wpinning,bounds=(85,105), args=(-0.001,model),
                          options={'xatol':1e-5,'disp':0}).x
    #t0 =  minimize_scalar(act.f_ene_ft_wpinning,(75,110),
    #                                args=(-0.001,model), tol = 1e-5).x
    
    t0 = brute_parallel(act.f_ene_ft_wpinning, 
                                 t0-3, t0+3, 0.01, args=(-0.001,model))
    
    t1 = minimize_scalar(act.f_ene_ft_wpinning,bounds=(70,85), args=(act.umax,model),
                          options={'xatol':1e-5,'disp':0}).x
    
    #    t1 =  minimize_scalar(act.f_ene_ft_wpinning,(75,110),
    #                      args=(U_max,model), tol = 1e-5, method='golden').x
    t1 = brute_parallel(act.f_ene_ft_wpinning, 
                                 t1-4, t1+4, 0.01, args= (U_max,model))
    
    #    print('t finding uth1: ', time.time()-t)
    t = time.time()
 #   act.theta_0r = t0
    t_l = [t0,t1]
    rans = [(0.0,30.0), (-70,-40)]
    ran_theta = (80,110)
    Uth = []
    with ThreadPool() as pool:
        items = [(rans[i],t_l[i],act, ran_theta, model) for i in range(2)]
        for result in pool.starmap(min_error,items):
            Uth.append(result)
    act.uth1 = Uth[0]
    act.uth2 = -Uth[1]
#    print('t finding uth2: ', time.time()-t)
    return Uth[0] , -Uth[1]

def H_vs_theta():
    theta = np.linspace(189,1,110)
    r = [h_calc(V,t)*1e6 for t in theta]

    plt.plot(theta,r)
    plt.xlabel('Contact Angle (deg)')
    plt.ylabel('Height (um)')
    plt.savefig('CA_vs_H.pdf', bbox_inches = 'tight')
    plt.show()

def A_calc(fname,ddir,V,theta=None):
    # Read the image of the electrode
    # dsize = int(input('Insert droplet diam. in um: '))
    ddir = ddir
    fname = fname
    Volume = V
    def calc(theta, plot = None):
        # Calculate the radius of the droplet
        dsize = 2.0*r_calc(Volume,theta)*1e6   # in um
        dens = int(fname.split('-')[0])    # pixels density in px/mm
        #dsize_px = int(dsize*dens/1000)    # diameter in px 
        pathImg = ddir + '/' +fname
        im = cv2.imread(pathImg)
        im_d, im_r = im2bin(im)
        #overlay = im.copy()
        name = 'output/'+fname[:-4]+'_drop'
        if plot:
            plotDroplet(int(dsize_px/2),overlay,name)
        geom_fact = geomFact2D(im_r, im_d, (dsize/2), displacement = 0, px_density = dens)  # in um^2
        return geom_fact[0]*1e-12
    return calc

def create_table(ddir,fname,Volume):
        # Calculate the radius of the droplet
        dens = int(fname.split('-')[0])    # pixels density in px/mm
        pathImg = ddir + '/' +fname
        im = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)
        im_d, im_r = im2bin(im)
        #plt.figure()
        #plt.imshow(im)
        #plt.imshow(im_r)
        #plt.show()
        #plt.imshow(im_d)
        #plt.show()
        rmin = r_calc(Volume,105)*1e6   # in um
        #rmin = 1000
        rmax = r_calc(Volume,100)*1e6   # in um
        #rmax = 1300
        N_th = 2
        res = 1/8.0                             # 1/8.0 #1/3.438 #um
        N_rad = int((rmax-rmin)/(N_th*res)+2)
        print('[Geometric Factor data creation] Rmin, Rmax ',rmin,rmax)
        print('Number of eval per thread: ',N_rad)
        input()
        r_extremes = np.linspace(rmin,rmax,N_th)
        print(' len r_extremes', len(r_extremes))
        data = []
        with ThreadPool() as pool:
            print('Start threadpool')
            items = [(im_r,im_d, (r_extremes[i], r_extremes[i+1]), N_rad, dens) for i in range(len(r_extremes)-1)]
            for result in pool.starmap(compute_gf,items):
                data.append(result)
        print(len(data))
        
        #r_items = np.arange(rmin, rmax, res)
        '''
        for r in r_items:
            print(len(data))
            data.append([r, np.pi*r**2])
        '''
        data = np.vstack(data)
        data = np.unique(data,axis=0)
        
        fname_csv = 'r_gf_' +fname+'.csv'
        print(fname_csv)
        with open(fname_csv,'wb') as f:
            np.savetxt(f,data,delimiter =",")
        ## ADD something to remove outliers before saving!
        data = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
        
        plt.plot(data[:,0],data[:,1])
        plt.xlabel('Rad')
        plt.ylabel('Cg')
        plt.savefig('Cg'+fname+'.pdf', bbox_inches = 'tight')
        plt.show()


class EH_Actuator:

    def __init__(self, V, gamma_lg, theta0, d, eps_d, theta0r = None
                 ,C_g = 1, C_g2 = 0, cpin = 0, cpin2 = 0,uth1 = None, uth2 = None
                 ,gamma_sl=None, model = None, C1 = 1e-4):
        self.V = V
        self.d = d
        self.gamma_lg = gamma_lg
        self.theta_0 = theta0
        self.theta_0r = theta0r
        self.C_g = C_g
        self.C1 = C1
        self.chi = 0.08e-3   # Coefficient of contact line friction [Kg/(s.m)] From paper not glyucerine
        self.eta = 5e-4 # Kinematic viscosity of glycerine m2/s
        if model == None:
            self.C_grec = self.C_g
        else:
            self.C_grec = C_g2
        self.eps_d = eps_d
        self.cpin = cpin
        self.cpin2 = cpin2
        self.uth1 = 0 if uth1 == None else uth1
        self.uth2 = np.Inf if uth2 == None else uth2
        if gamma_sl == None:
            (self.gamma_sl_adv, self.gamma_sl_rec) = self.gamma_sl_calc(self.theta_0, dtheta = 0.01)
            print(self.gamma_sl_adv)
            print(self.gamma_sl_rec)
        else:
            self.gamma_sl = gamma_sl

        #self.gamma_sl_rec = self.gamma_sl_calc(self.theta_0r, dtheta = 0.01) if theta0r is not None else None

    def gamma_sl_calc(self, theta, dtheta):
        ''' Theta in degrees
        This function returns the value of (gamma_sl-gamma_lg)
        gamma equiv.
        gamma_sl = -(gamma_lg*f1'(theta))/f2'(theta) (OLD)
            
        '''
        c1 = (self.chi+6*self.eta)*self.C1*self.gamma_lg/self.eta
        g_rec = (c1*np.sin(self.theta_0*np.pi/180)**2*np.cos(self.theta_0*np.pi/180)\
                +2*c1*np.sin(self.theta_0*np.pi/180)**2-2*self.gamma_lg*np.cos(self.theta_0*np.pi/180))/2
        c1 = -c1
        g_adv = (c1*np.sin(self.theta_0*np.pi/180)**2*np.cos(self.theta_0*np.pi/180)\
                +2*c1*np.sin(self.theta_0*np.pi/180)**2-2*self.gamma_lg*np.cos(self.theta_0*np.pi/180))/2
        self.gamma_sl_adv = g_adv
        self.gamma_sl_rec = g_rec
        return g_adv, g_rec
        
        #return -self.gamma_lg*2*diff_central(f1_theta, theta, dtheta) / \
        #    diff_central(f2_theta, theta, dtheta)


    def gamma_sl_calc_fmin(self, theta, gamma_sl_0, er=1e-6, N=100):
        ''' Theta in degrees
        This function returns the value of gamma_sl
        using fmin
        '''
        self.gamma_sl = gamma_sl_0
        #t_min = fmin(f_ene,theta,args=(None,0,V,gamma_sl_0,gamma_lg), disp=False)
        t_min = fmin(self.f_ene_ft_wpinning, theta,args=(0,1), disp=False)
        i = 0
        while abs(t_min-theta)>er:
            i=i+1
            if t_min < theta:
                self.gamma_sl = self.gamma_sl*1.0001**(1-0.1**N)
                t_min = fmin(self.f_ene_ft_wpinning, theta,args=(0,1), disp=False)
            if t_min > theta:
                self.gamma_sl = self.gamma_sl*0.9999*(1-0.1**N)
                t_min = fmin(self.f_ene_ft_wpinning, theta,args=(0,1), disp=False)
            if i>N:
                print('Max iter reached')
                break
        print('theta min: ', t_min)
        return self.gamma_sl


    def f_ene(self, theta, func_geom_fact, U, SplitEne=False):
        '''
            It calculate the energy of the system
        '''
        eps_0 = 8.85418782*1e-12  # [F/m]
        
        E_surf = self.gamma_sl*np.pi**(1/3)*(3*self.V)**(2/3)*f2_theta(theta) +\
                 self.gamma_lg*2*np.pi**(1/3)*(3*self.V)**(2/3)*f1_theta(theta)
     
        if func_geom_fact==None:
            return E_surf
        E_elec = eps_0*self.eps_d/(2*self.d)*self.C_g*func_geom_fact(theta)*U**2
        return E_surf-E_elec

    def load_table(self, table):
        self.table = table

    def inv_f_ene_ft_wpinning(self, U, theta, model = 1, split=False):
        '''
            It calculate the energy of the system
            The voltage is negative for the reciding part of the curve
        '''
        eps_0 = 8.85418782*1e-12  # [F/m]
        A_lg = 2*np.pi**(1/3)*(3*self.V)**(2/3)*f1_theta(theta)
        A_sl = np.pi**(1/3)*(3*self.V)**(2/3)*f2_theta(theta)
        print('U: ', U)
        print('Theta: ', theta)
        if model == 1:
            E_surf = self.gamma_sl*A_sl + self.gamma_lg*A_lg 
        elif model == 0:
            if U >= 0:
                E_surf = self.gamma_sl_adv*A_sl + self.gamma_lg*A_lg 
            else:
                E_surf = self.gamma_sl_rec*A_sl + self.gamma_lg*A_lg 
                #E_surf = self.gamma_sl_rec*A_sl + self.gamma_lg*A_lg 

        E_pinning = self.cpin*A_sl #if model == 1 else 0
        
        if self.table.any()==None:
            if split:
                return E_surf,0,0,0
            else:
                return E_surf

        r = self.r_calc(theta)*1e6   # in um
        # Check if the value is out of the table
        if r<np.min(self.table[:,0]) or r>np.max(self.table[:,0]):
            print(f'WARNING RADIUS IS OUT OF THE DATA, rad: {r}')
        gf = self.table[np.absolute(self.table[:,0]-r).argmin(),1] # in um^2
        gf = gf*1e-12


        if 0 <= U < self.uth1:
            if split:
                return E_surf,0,0,0
            else:
                return E_surf

        elif self.uth1 <= U:
            E_elec = eps_0*self.eps_d/(2*self.d)*self.C_g*gf*np.abs(U)**2
            E_fric = (self.chi+6*self.eta)*self.C1*(self.gamma_lg/eta)*(np.cos(self.theta_0*np.pi/180)+eps_0*self.eps_d/(2*self.d)*self.C_g*gf*np.abs(U)**2/self.gamma_lg-np.cos(theta*np.pi/180))*A_sl*0
            if split:
                return E_surf, E_pinning, -E_elec, E_fric
            else:
                return E_surf + E_pinning - E_elec + E_fric

        elif  U < -1*float(self.uth2):
            if split:
                return E_surf,0,0,0
            else:
                return E_surf

        elif  0 > U >= -1*float(self.uth2):
            E_elec = eps_0*self.eps_d/(2*self.d)*self.C_grec*gf*np.abs(U)**2
            E_fric = (self.chi+6*self.eta)*self.C1*(self.gamma_lg/eta)*(np.cos(self.theta_0*np.pi/180)+eps_0*self.eps_d/(2*self.d)*self.C_g*gf*np.abs(U)**2/self.gamma_lg-np.cos(theta*np.pi/180))*A_sl
            return E_surf - E_pinning - E_elec - E_fric
        print(U)
        print('uth1 ', self.uth1)
        print('uth2 ', self.uth2)
        print('OUT of ene value')
        return 0

    def f_ene_ft_wpinning(self, theta, U, model = 1, split=False):
        '''
            It calculate the energy of the system
            The voltage is negative for the reciding part of the curve
        '''
        eps_0 = 8.85418782*1e-12  # [F/m]
        #A_lg = 2*np.pi**(1/3)*(3*self.V)**(2/3)*f1_theta(theta)
        #A_sl = np.pi**(1/3)*(3*self.V)**(2/3)*f2_theta(theta)
        r = self.r_calc(theta)
        R = self.r_calc(theta)/np.sin(theta*np.pi/180)
        
        #print(R)
        A_lg = 2*np.pi*R**2*(1-np.cos(theta*np.pi/180))
        #A_sl = np.pi*R**2*np.sin(theta*np.pi/180)**2
        A_sl = np.pi*r**2
        
        if U >= 0:
            #E_surf = self.gamma_sl_adv*A_sl + self.gamma_lg*A_lg 
            E_surf = self.gamma_lg*(A_lg - A_sl*np.cos(self.theta_0*np.pi/180))
        else:
            #E_surf = self.gamma_sl_rec*A_sl + self.gamma_lg*A_lg 
            E_surf = self.gamma_lg*(A_lg - A_sl*np.cos(self.theta_0*np.pi/180))
            #E_surf = self.gamma_sl_rec*A_sl + self.gamma_lg*A_lg 
        if U<0:
            E_pinning = self.cpin2*A_sl #if model == 1 else 0
        else:
            E_pinning = self.cpin*A_sl #if model == 1 else 0
        
        if self.table.any()==None:
            if split:
                return E_surf,0,0,0, E_surf
            else:
                return E_surf

        r = self.r_calc(theta)*1e6   # in um
        if r<np.min(self.table[:,0]) or r>np.max(self.table[:,0]):
            print(f'WARNING RADIUS IS OUT OF THE DATA, theta: {r}; {theta}')
            print(f'Min, max table: {np.min(self.table[:,0])} , {np.max(self.table[:,0])} ')
            #input("Press Enter to continue...")
        # Instead of taking the closer value, I use interpolation
        #gf = self.table[np.absolute(self.table[:,0]-r).argmin(),1] # in um^2
        r_d = self.table[:,0]
        gf_d = self.table[:,1]
        f = interp1d(r_d,gf_d, bounds_error = False, fill_value='extrapolate')
        gf = f(r)
        gf = gf*1e-12

        chi = 0.08e-3   # Coefficient of contact line friction [Kg/(s.m)] From paper not glyucerine
        eta = 5e-4 # Kinematic viscosity of glycerine m2/s

        if 0 <= U < self.uth1:
            if split:
                return E_surf,0,0,0, E_surf
            else:
                return E_surf

        elif self.uth1 <= U:
            E_elec = eps_0*self.eps_d/(2*self.d)*self.C_g*gf*np.abs(U)**2
            #v_r = self.C1*(self.gamma_lg/eta)*(np.cos(self.theta_0*np.pi/180)+eps_0*self.eps_d*np.abs(U)**2/(2*self.d*self.gamma_lg)-np.cos(theta*np.pi/180))
            #E_fric = (chi+6*eta)*v_r**0.5*A_sl
            E_fric = self.C1*U*A_sl
            #E_fric = (chi+6*eta)*self.C1*(self.gamma_lg/eta)*eps_0*self.eps_d*np.abs(U)**2/(2*self.d*self.gamma_lg)*A_sl
            #E_fric = np.cos(self.theta_0*np.pi/180)-np.cos(theta*np.pi/180)

            if split:
                return E_surf, E_pinning, - E_elec, E_fric, E_surf + E_pinning - E_elec + E_fric
            else:
                return E_surf + E_pinning - E_elec + E_fric


        elif  U < -1*float(self.uth2):
            if split:
                return E_surf,0,0,0, E_surf
            else:
                return E_surf

        elif  0 > U >= -1*float(self.uth2):
            E_elec = eps_0*self.eps_d/(2*self.d)*self.C_grec*gf*np.abs(U)**2
            
            #E_fric = (chi+6*eta)*self.C1*(self.gamma_lg/eta)*(np.cos(self.theta_0*np.pi/180)-eps_0*self.eps_d*self.C_grec*gf*np.abs(U)**2/(2*self.d*self.gamma_lg)/A_sl-np.cos(theta*np.pi/180))*A_sl
            E_fric = self.C1*U*A_sl
            

            if split:
                return E_surf, -E_pinning, -E_elec, E_fric, E_surf - E_pinning - E_elec - E_fric
            else:
                return E_surf - E_pinning - E_elec - E_fric

        print(U)
        print('uth1 ', self.uth1)
        print('uth2 ', self.uth2)
        print('OUT of ene value')
        return 0


    def f_ene_from_table(self, theta, U,SplitEne=False):
        eps_0 = 8.85418782*1e-12  # [F/m]
        
        E_surf = self.gamma_sl*np.pi**(1/3)*(3*self.V)**(2/3)*f2_theta(theta) +\
                 self.gamma_lg*2*np.pi**(1/3)*(3*self.V)**(2/3)*f1_theta(theta)
     
        if self.table.any()==None:
            return E_surf
        r = self.r_calc(theta)*1e6   # in um
        gf = self.table[np.absolute(self.table[:,0]-r).argmin(),1] # in um^2
        gf = gf*1e-12
        E_elec = eps_0*self.eps_d/(2*self.d)*self.C_g*gf*U**2
        if SplitEne:
            return E_surf,E_elec
        return E_surf-E_elec


    def f_dEne(theta,func_geom_fact,U,V,eps_d=2):
        eps_0 = 8.85418782*1e-12  # [F/m]
        # And this equation to be sure if f2 multiply gamma_sl or gamma_lg
        E_surf = gamma_sl*np.pi**(1/3)*(3*V)**(2/3)*diff_central(f2_theta,theta,dtheta) +\
                 gamma_lg*2*np.pi**(1/3)*(3*V)**(2/3)*diff_central(f1_theta,theta,dtheta)
        E_elec = eps_0*eps_d/(2*d)*C_g*diff_central(func_geom_fact,theta,dtheta)*U**2
    #    E_elec = diff_central(func_geom_fact,theta,dtheta)
    #    print(E_elec)
        return E_surf-E_elec

    def r_calc(self, theta):
        ''' Given the volume V and the contact angle theta, this function returns
            the value of the contact radius
        '''
        return (3*self.V/np.pi*np.sin(theta*np.pi/180)**3/f_theta(theta))**(1/3)

    def h_calc(self,theta):
        '''
            It returns the height of the droplet
        '''
        return (3*self.V/np.pi*(1-np.cos(theta*np.pi/180))**3/f_theta(theta))**(1/3)
        

def r_calc(V,theta):
    ''' Given the volume V and the contact angle theta, this function returns
        the value of the contact radius
    '''
    return (3*V/np.pi*np.sin(theta*np.pi/180)**3/f_theta(theta))**(1/3)


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


def diff_central(fun, x_value, delta):
    ''' Returns derivative using central differences '''
    return (fun(x_value+delta)-fun(x_value-delta))/(2.0*delta)





