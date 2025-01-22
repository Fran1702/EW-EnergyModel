import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from stabfunc import *
from matplotlib import cm
import imageio
import os


def readfiles(datadir, ext = '.png'):
    files = []
    # Read the files and select those without a number at the beginning 
    for file in os.listdir(datadir):
        if file.endswith(ext):
            if file.startswith(tuple([str(x) for x in range(0,10)])):
                files.append(file)
    return datadir, files


os.chdir("../../data")
datadir = os.getcwd()

ddir, files = readfiles(datadir,'.png')

im = cv2.imread(pathImg)
im_d, im_r = im2bin(im)


# Read the image with Opencv
img = cv2.imread('lena.png')
# Change the color from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Orgird to store data
x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
# In Python3 matplotlib assumes rgbdata in range 0.0 to 1.0
img = img.astype('float32')/255
fig = plt.Figure()
# gca do not work thus use figure objects inbuilt function.
ax = fig.add_subplot(projection='3d')

# Plot data
ax.plot_surface(x, y, np.atleast_2d(0), rstride=10, cstride=10, facecolors=img)
# fig.show() # Throws a AttributeError
# As above did not work, save the figure instead.
fig.savefig("results.png")


#ddir, files = readfiles(datadir,'.npy')
#    for f in files:
#        X,Y,Z = np.load(ddir+'/'+f)
#        plot2DgeomFact_Img(X,Y,Z,contour = False, namefig = ddir+'/output/plot'+f[:-4]+'.png')

