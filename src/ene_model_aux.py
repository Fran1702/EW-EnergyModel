import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from scipy.optimize import minimize_scalar
import cv2
from stabfunc import *

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

def theta_func0(U, C_g, C_g2, theta0, theta0r, dt0a, dt0r, act = None):
    global V, gamma_lg, eps_d, fname
    model = 0
    d_theta = 5
    dt = 0.001
    theta_x = theta_0
    fname_csv = 'r_gf_' +fname+'.csv'
    data = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
    if act == None:
        act = EH_Actuator(V, gamma_lg, theta0, d, eps_d, theta0r = theta0r
                         ,C_g = C_g, C_g2 = C_g2, cpin = 0, uth1 = 1
                         , uth2 = 120, model=0)    # Create actuator w/params
        act.load_table(data)
        find_uth1_uth2(act, np.max(U),  model)

    theta_l = []
    with ThreadPool() as pool:
        items = [(U[i],act, model,dt0a,dt0r) for i in range(len(U))]
        for result in pool.starmap(calc_thetax_U,items):
            theta_l.append(result)
    theta_l = np.vstack(theta_l).flatten()
    return theta_l

def theta_func0_lmfit(params, x, V, d, gamma_lg, eps_d, fname, data=None):
    print('ASDS')
    U = x
    C_g = params['C_g'].value
    C_g2 = params['C_g2'].value
    theta0 = params['theta0'].value
    theta0r = params['theta0r'].value
    if 'dt0a' in params:    
        dt0a = params['dt0a'].value
    if 'dt0r' in params:    
        dt0r = params['dt0r'].value
    cpin = params['cpin'].value
    print(params.valuesdict().values())
    model = 1
    d_theta = 5
    dt = 0.001
    theta_x = theta0
    fname_csv = 'r_gf_' + fname+'.csv'
    data_table = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
    act = EH_Actuator(V, gamma_lg, theta0, d, eps_d, theta0r = theta0r
                     ,C_g = C_g, C_g2 = C_g2, cpin = cpin, uth1 = 1
                     , uth2 = 120, model=model)    # Create actuator w/params
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

def calc_thetax_U(Uval, act, model, dt0a=0, dt0r=0):
'''
    return the value of theta for a given U and actuator, using the model with pinning
'''
    if 0 <= float(Uval) and float(Uval) < float(act.uth1):
        theta_x = min_scal_parallel(act.f_ene_ft_wpinning, -0.001, (75,110), model)
    elif float(Uval) < -1*float(act.uth2):
        theta_x = min_scal_parallel(act.f_ene_ft_wpinning, act.umax, (75,110), model)
    else:
        theta_x = min_scal_parallel(act.f_ene_ft_wpinning, Uval, (75,110), model)
    if Uval >= 0:
        theta_x = theta_x + dt0a
    elif Uval < 0:    
        theta_x = theta_x + dt0r
    return theta_x


def error_uth(U, real_value, act, ran_theta,  model=1):
    act_value = min_scal_parallel(act.f_ene_ft_wpinning, U, ran_theta, model)
    err = np.abs(act_value - real_value)
    return err

def min_error(ran, t0, act, ran_theta, model):
    Uth = minimize_scalar(error_uth, bounds=ran, args=(t0,act,ran_theta, model), method='bounded').x
    return Uth

def find_uth1_uth2(act, U_max, model=1, ran_theta=(75,105), ran_t0 = (1.0,60.0), ran_t1=(120.0,20.0)):
    act.umax = U_max
    t0 =  min_scal_parallel(act.f_ene_ft_wpinning, -0.001, ran_theta, model)
    t1 =  min_scal_parallel(act.f_ene_ft_wpinning, U_max, ran_theta, model)
    act.theta_0r = t0
    t_l = [t0,t1]
    rans = [ran_t0, -ran_t1]
    Uth = []
    with ThreadPool() as pool:
        items = [(rans[i],t_l[i],act, ran_theta, model) for i in range(2)]
        for result in pool.starmap(min_error,items):
            Uth.append(result)
    act.uth1 = Uth[0]
    act.uth2 = -Uth[1]
    return Uth[0] , -Uth[1]

def H_vs_theta(V):
    theta = np.linspace(189,1,150)
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
        dsize = 2.0*contact_rad_calc(Volume,theta)*1e6   # in um
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
        im = cv2.imread(pathImg)
        im_d, im_r = im2bin(im)
        rmin = contact_rad_calc(Volume,105)*1e6   # in um
        rmax = contact_rad_calc(Volume,70)*1e6   # in um
        N_th = 10
        res = 0.1 #um
        N_rad = int((rmax-rmin)/(N_th*res)+2)
        print(rmin,rmax)
        print(N_rad)
        r_extremes = np.linspace(rmin,rmax,N_th)
        data = []
        with ThreadPool() as pool:
            items = [(im_r,im_d, (r_extremes[i], r_extremes[i+1]), N_rad, dens) for i in range(len(r_extremes)-1)]
            for result in pool.starmap(compute_gf,items):
                data.append(result)
        data = np.vstack(data)
        data = np.unique(data,axis=0)
        fname_csv = 'r_gf_' +fname+'.csv'
        print(fname)
        with open(fname_csv,'wb') as f:
            np.savetxt(f,data,delimiter =",")
        data = np.genfromtxt(fname_csv, delimiter =',',dtype=None)
        plt.plot(data[:,0],data[:,1])
        plt.show()


class EH_Actuator:

    def __init__(self, V, gamma_lg, theta0, d, eps_d, theta0r = None
                 ,C_g = 1, C_g2 = 0, cpin = 0, uth1 = None, uth2 = None
                 ,gamma_sl=None, model = None):
        self.V = V
        self.d = d
        self.gamma_lg = gamma_lg
        self.theta_0 = theta0
        self.theta_0r = theta0r
        self.C_g = C_g
        if model == None:
            self.C_grec = self.C_g
        else:
            self.C_grec = C_g2
        self.eps_d = eps_d
        self.cpin = cpin
        self.uth1 = 0 if uth1 == None else uth1
        self.uth2 = np.Inf if uth2 == None else uth2
        if gamma_sl == None:
            self.gamma_sl = self.gamma_sl_calc(self.theta_0, dtheta = 0.01)
        else:
            self.gamma_sl = gamma_sl

        self.gamma_sl_rec = self.gamma_sl_calc(self.theta_0r, dtheta = 0.01) if theta0r is not None else None

    def gamma_sl_calc(self, theta, dtheta):
        ''' Theta in degrees
        This function returns the value of gamma_sl
        gamma_sl = -(gamma_lg*f1'(theta))/f2'(theta)
        '''
        return -self.gamma_lg*2*diff_central(f1_theta, theta, dtheta) / \
            diff_central(f2_theta, theta, dtheta)


    def gamma_sl_calc_fmin(theta, V, gamma_lg, gamma_sl_0, er=1e-6):
        ''' Theta in degrees
        This function returns the value of gamma_sl
        using fmin
        '''
        gamma_sl = gamma_sl_0
        t_min = fmin(f_ene,theta,args=(None,0,V,gamma_sl_0,gamma_lg), disp=False)
        while abs(t_min-theta)>er:
            if t_min < theta:
                gamma_sl = gamma_sl*1.001
                t_min = fmin(f_ene,theta,
                             args=(None,0,V,gamma_sl,gamma_lg),
                             disp=False)
            if t_min > theta:
                gamma_sl = gamma_sl*0.999
                t_min = fmin(f_ene,theta,
                             args=(None,0,V,gamma_sl,gamma_lg),
                             disp=False)
        return gamma_sl


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
    
    def f_ene_ft_wpinning(self, theta, U, model = 1):
        '''
            It calculate the energy of the system
            The voltage is negative for the reciding part of the curve
        '''
        eps_0 = 8.85418782*1e-12  # [F/m]
        A_lg = 2*np.pi**(1/3)*(3*self.V)**(2/3)*f1_theta(theta)
        A_sl = np.pi**(1/3)*(3*self.V)**(2/3)*f2_theta(theta)
        if model == 1:
            E_surf = self.gamma_sl*A_sl + self.gamma_lg*A_lg 
        elif model == 0:
            if U >= 0:
                E_surf = self.gamma_sl*A_sl + self.gamma_lg*A_lg 
            else:
                E_surf = self.gamma_sl*A_sl + self.gamma_lg*A_lg 
                #E_surf = self.gamma_sl_rec*A_sl + self.gamma_lg*A_lg 

        E_pinning = self.cpin*A_sl if model == 1 else 0
        
        if self.table.any()==None:
            return E_surf
        r = contact_rad_calc(self.V, theta)*1e6   # in um
        gf = self.table[np.absolute(self.table[:,0]-r).argmin(),1] # in um^2
        gf = gf*1e-12
        if 0 <= U < self.uth1:
            return E_surf 
        elif self.uth1 <= U:
            E_elec = eps_0*self.eps_d/(2*self.d)*self.C_g*gf*np.abs(U)**2
            return E_surf + E_pinning - E_elec
        elif  U < -1*float(self.uth2):
            return E_surf 
        elif  0 > U >= -1*float(self.uth2):
            E_elec = eps_0*self.eps_d/(2*self.d)*self.C_grec*gf*np.abs(U)**2
            return E_surf - E_pinning - E_elec
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
        r = contact_rad_calc(self.V, theta)*1e6   # in um
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


