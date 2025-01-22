import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from stabfunc import *
from matplotlib import cm

d1Calc = 0
plotd1 = 1

def plot2DgeomFact(X,Y,Z,contour=False):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax = plt.axes(projection = '3d')
    surf = ax.plot_surface(X, Y, (Z/np.amax(Z)), alpha = 0.8, cmap = cm.jet)
    if contour:
        cset = ax.contour(X, Y, Z/np.amax(Z)-0.02, 25, offset=np.min(Z/np.amax(Z))-0.02, cmap=cm.jet , linestyles = 'solid')
    ax.set_zlabel('Normalized Geometric Factor')
    ax.set_xlabel('Droplet displacement [px]')
    ax.set_ylabel('Droplet displacement [px]')
    plt.show()

#------------------------------------------------------
## Design1 Electrodes analisys
#------------------------------------------------------

file = 'design7'
pathImg2 = file+'.png'
## Leo las imagenes de los electrodos y las convierto a binario
imgDC = cv2.imread(pathImg2)
imgDC_D = (255-imgDC[:,:,2])
imgDC_R = (255-imgDC[:,:,0])

if d1Calc:

#    imgDC_D = np.delete(imgDC_D, slice(0,19), axis = 1)
#    imgDC_R = np.delete(imgDC_R, slice(0,19), axis = 1)

#    imgDC_D = np.delete(imgDC_D, slice(-22,-1), axis = 0)
#    imgDC_R = np.delete(imgDC_R, slice(-22,-1), axis = 0)

    if False:
        drop = create_Circle(imgDC,rad = 280 ,center_Drift=(0,0))
        im2 = np.zeros_like(imgDC)
        im2[:,:,1] = imgDC_D
        im2[drop[:,:,0]>0] = drop[drop[:,:,0]>0]
        cv2.imshow('image',im2)# + drop)#imgDC_D+drop[:,:,0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    X,Y,Z = geomFact2D(imgDC_R, imgDC_D, 40, displacement = 100)

    np.savetxt(file+'_GF_Z.out', Z)
    np.savetxt(file+'_GF_X.out', X)
    np.savetxt(file+'_GF_Y.out', Y)

if plotd1:
    X = np.loadtxt(file+'_GF_X.out')
    Y = np.loadtxt(file+'_GF_Y.out')
    Z = np.loadtxt(file+'_GF_Z.out')
    n = 1
    plot2DgeomFact(X[n:-n,n:-n],Y[n:-n,n:-n],Z[n:-n,n:-n],contour = False)



im2 = np.zeros_like(imgDC)
im2[:,:,0] = imgDC_R 
im2[:,:,1] = imgDC_D

drop = create_Circle(imgDC,rad = 70 ,center_Drift=(0,0))
im2 = im2 + drop
#im2[drop[:,:,0]>0] = drop[drop[:,:,0]>0]
cv2.imshow('image',im2)# + drop)#imgDC_D+drop[:,:,0])
cv2.waitKey(0)
cv2.destroyAllWindows()
