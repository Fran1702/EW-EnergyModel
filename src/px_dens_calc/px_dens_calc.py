import os
import cv2
import numpy as np

os.chdir("../../data")
datadir = os.getcwd()
files = []

print(datadir)
# Read the files and select those without a number at the beginning 
for file in os.listdir(datadir):
    if file.endswith(".png"):
        if not file.startswith(tuple([str(x) for x in range(0,10)])):
            files.append(file)
            #pxdensity = file.split("-")[0]
print(datadir+'/'+files[0])


def imcrop(image,whitebackground = 1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if whitebackground:
        thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1] # For white background
    else:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1] # For black background
    # Find contour and sort by contour area
    nonzero_idx = np.nonzero(thresh)
    x,w = np.min(nonzero_idx[0]), np.max(nonzero_idx[0])   
    y,h = np.min(nonzero_idx[1]), np.max(nonzero_idx[1])   
    ROI = image[x:w,y:h]
    return ROI
#
def improcess(image,file):
    ROI = imcrop(image)
    cv2.imshow('image', ROI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    axis = input('Horizontal or vertical dimension (h or v)?')
    dpx = 0
#    print(ROI.shape)
    if axis == 'v':
        dpx = ROI.shape[0]
    elif axis == 'h':
        dpx = ROI.shape[1]
    else:
        print('Error')

    dum = input('Insert length in um:')
    dmm = int(dum)/1000
    newname = str(int(int(dpx)/dmm)) + '-' +file
    print(newname)

    cv2.imwrite(datadir+'/'+newname, ROI,[cv2.IMWRITE_JPEG_QUALITY, 9])


for f in files:
    pathImg = datadir +'/' + f
    yn = input('Do you want to process file: ' + f+' (y/n)?') 
    if yn == 'y':
        im = cv2.imread(pathImg)
        improcess(im,f)
    

