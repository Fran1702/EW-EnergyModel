import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def create_Circle(img,rad=100,color=(255,0,0), center_Drift = (0,0)):

  # Reading an image in default mode
  dim = (img.shape[0],img.shape[1],3)
  Img = np.zeros(dim, np.uint8)
  # Center coordinates
  center_coordinates = (int(img.shape[1]/2 + center_Drift[0]), int(img.shape[0]/2 + center_Drift[1])) 
  thickness = -1
  # Draw a circle of red color of thickness -1 px
  image = cv2.circle(Img, center_coordinates, rad, color, thickness)
  return image

def geomFactCalc(Elect_R, Elect_D, Rdroplet, cent_DriftX = 0, cent_DriftY = 0 ):
  """
        Calculate the geometric factor of the electrode shape for a droplet of Radious Rdroplet
    Arguments:
        Elect_R : Binary image represented by a numpy array nxm 
        Elect_D : Binary image represented by a numpy array nxm 
        Rdroplet : Radious of the droplet
        center_DriftX : np array, x position of the droplet, x distance to the center of the image
        center_DriftY : np array, y position of the droplet, y distance to the center of the image
    Returns:
        The value of the geometric factor (numpy array)
  """
  geomFact = []
  for i in range(len(cent_DriftX)):
    droplet = create_Circle(Elect_R, rad = Rdroplet ,center_Drift = (cent_DriftX[i], cent_DriftY[i]))
    droplet = (droplet[:,:,0])
    droplet = droplet/np.amax(droplet)
    Ar = np.sum(droplet*Elect_R)
    Ad = np.sum(droplet*Elect_D)
    geomFact.append((Ad*Ar**2+Ar*Ad**2)/(Ar+Ad)**2)

  return np.array(geomFact)


def geomFact2D(Elect_R, Elect_D, Rdroplet, displacement=50, px_density = 1000):
    """
            Creates a 2D surface with the geometric factor
    Arguments:
        Elect_R : Binary image represented by a numpy array nxm (Reference electrode)
        Elect_D : Binary image represented by a numpy array nxm (Driven electrode
        Rdroplet : Radious of the droplet in um
        displacement: Displacement of the droplet to creates the grid un um
        px_density: Pixel density in px/mm 
    Returns
        X : grid in um or px
        Y : grid in um or px
        Z : Geometric factor in um^2
    """
    rDrop_px = int(Rdroplet*px_density/1000)
    n,m = Elect_R.shape
    xCent = int(n/2)
    yCent = int(m/2)
    '''
    if displacement != 0:
        a = 2*int(displacement*px_density/1000) # Convert um to px
        b = 2*int(displacement*px_density/1000) # Convert um to px
        x = np.arange(a)
        y = np.arange(b)
        X, Y = np.meshgrid(x,y)
        r = np.array([X.flatten(), Y.flatten()])
    #    geomFact = geomFactCalc(Elect_R, Elect_D, Rdroplet, r[0,:]-int(a/2),r[1,:]-int(b/2) )
        #geomFact = geomFactMod2(Elect_R, Elect_D, rDrop_px, r[0,:]-int(a/2),r[1,:]-int(b/2), px_density = px_density )
        geomFact = geomFactMod2(Elect_R, Elect_D, rDrop_px, a,0, px_density = px_density )
        Z = geomFact.reshape(X.shape)
        return (X-int(a/2))*1000/px_density ,(Y-int(b/2))*1000/px_density,Z
    '''
    #else:
    # Here I compute to let the edge of the droplet fixed to one side (t degrees)
    t = 50
    dpx = int(displacement*px_density/1000)
    if dpx >= rDrop_px:
        cd_x = int((dpx - rDrop_px)*np.cos(t*np.pi/180))
        cd_y = int((dpx - rDrop_px)*np.sin(t*np.pi/180))
    else:
        cd_x = 0
        cd_y = 0
    geomFact = geomFactMod2(Elect_R, Elect_D, rDrop_px, cd_x, cd_y, px_density = px_density )
    return geomFact


def geomFactMod2(Elect_R, Elect_D, Rdroplet, cent_DriftX = 0, cent_DriftY = 0, px_density = 1000 ):
    """
        Calculate the geometric factor by the model 2
    Arguments:
        Elect_R : Binary image represented by a numpy array nxm
        Elect_D : Binary image represented by a numpy array nxm
        Rdroplet : Radious of the droplet
        center_DriftX : np array, x position of the droplet, x distance to the center of the image
        center_DriftY : np array, y position of the droplet, y distance to the center of the image
        px_density: Pixel density in px/mm 
    Returns:
        The value of the geometric factor (numpy array) in um2
    """
    geomFact = []
    #if cent_DriftX == 0:
#    x_d = int(cent_DriftX*px_density/1000)
#    print('cent_drift', cent_DriftX)
#    print('x_d', x_d)
    Elect_R = (Elect_R > 0).astype(np.uint8)
    Elect_D = (Elect_D > 0).astype(np.uint8)
    if isinstance(cent_DriftX,int):
        droplet = create_Circle(Elect_R, rad = Rdroplet ,
                                center_Drift = (cent_DriftX, cent_DriftY))
        ##droplet = (droplet[:,:,0])
        droplet_mask = (droplet[:, :, 0] > 0).astype(np.uint8)
        
        #droplet[droplet>0] = 1 
        #plt.imshow(droplet)
        droplet = (droplet >0)*1
        #Elect_R = (Elect_R>0)*1
        #Elect_D = (Elect_D>0)*1
        #droplet[droplet<0] = 0
        #Elect_R[Elect_R<0] = 0
        #Elect_D[Elect_D<0] = 0
##        Ar = np.sum(np.array(droplet)*np.array(Elect_R))*(1000/px_density)**2
##        Ad = np.sum(np.array(droplet)*np.array(Elect_D))*(1000/px_density)**2
        Ar = np.sum(droplet_mask * Elect_R) * (1000 / px_density) ** 2
        Ad = np.sum(droplet_mask * Elect_D) * (1000 / px_density) ** 2
        
        geomFact.append(1/(1/Ad+1/Ar))

    else:
        
        ##for i in range(len(cent_DriftX)):
        for x, y in zip(cent_DriftX, cent_DriftY):
            ##droplet = create_Circle(Elect_R, rad = Rdroplet ,center_Drift = (cent_DriftX[i], cent_DriftY[i]))
            ##droplet = (droplet[:,:,0])
            ##droplet = (droplet >0)*1
            ##Elect_R = (Elect_R>0)*1
            ##Elect_D = (Elect_D>0)*1
            #cv2.imshow('image',100*(Elect_R +Elect_D + droplet))#imgDC_D+drop[:,:,0])
            # if (i%100 == 0) :
              #  cv2.imshow('image',100*droplet+100*Elect_R)# + drop)#imgDC_D+drop[:,:,0])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #droplet = create_Circle(Elect_R, rad=Rdroplet, center_Drift=(x, y))
            droplet = create_Circle(Elect_R, rad=Rdroplet, center_Drift=(x, y))
            droplet_mask = (droplet[:, :, 0] > 0).astype(np.uint8)
            
            Ad = np.sum(droplet_mask * Elect_D) * (1000 / px_density) ** 2
##            Ar = np.sum(np.array(droplet)*np.array(Elect_R))*(1000/px_density)**2
  ##          Ad = np.sum(np.array(droplet)*np.array(Elect_D))*(1000/px_density)**2
            geomFact.append(1/(1/Ar+1/Ad))

    return np.array(geomFact)


def im2bin(img):
    img_D = (255-img[:,:,2])
    img_R = (255-img[:,:,0])
    img_D = np.where(img_D > 100, 1, 0)
    img_R = np.where(img_R > 100, 1, 0)
    return img_D, img_R

def plotDroplet(radius,img,nameFig):
    overlay = img.copy()
    dim = (img.shape[0],img.shape[1],3)
    # Center coordinates
    center_coordinates = (int(img.shape[1]/2),int(img.shape[0]/2)) 
    cv2.circle(overlay, center_coordinates, radius, (0,0,0),-1)
    alpha = 0.5
    img_new = cv2.addWeighted(overlay, alpha, img, 1-alpha,0)
    img_new_resized = cv2.resize(img_new,(400,400))
    cv2.imshow('image',img_new_resized)# + drop)#imgDC_D+drop[:,:,0])
    cv2.waitKey(0)
    #cv2.imwrite(nameFig+'_schem.png',img_new)
    cv2.destroyAllWindows()


def readfiles(datadir, ext = '.png'):
    files = []
    # Read the files and select those without a number at the beginning 
    for file in os.listdir(datadir):
        if file.endswith(ext):
            if file.startswith(tuple([str(x) for x in range(0,10)])):
                files.append(file)
    return datadir, files



def select_file(datadir):
    ddir, files = readfiles(datadir,'.png')
    for f in files:
        yn = input('Do you want to process file: ' + f+' (y/n)?')
        if yn == 'y':
            return f,ddir

def compute_gf(Elect_R, Elect_D, R_range, N, px_density):
    print('Computing')
    rmin, rmax = R_range
    gf_values = []
    i = 0
    for r in np.linspace(rmin,rmax,N):
        i = i+1
        gf = geomFact2D(Elect_R, Elect_D, r, displacement=0, px_density=px_density)
        gf_values.append([r, gf[0]])
        print(f'Computed: {i}/{N}')
    print('Computed ALL')
    return gf_values

