import cv2 as cv
import math
import numpy as np 

img = cv.imread('/home/likcos/Imágenes/mo1.png',0)
h,w = img.shape[:2]
img2 = np.zeros((h*3, w*3), dtype = "uint8")
for i in range(h):
    for j in range(w):
        img2[int(i*math.cos(math.radians(30))-j*math.sin(math.radians(30)))+200,
             int(i*(math.sin(math.radians(30)))+j*math.cos(math.radians(30)))+50]=img[i,j]
cv.imshow('imagen1', img)
cv.imshow('imagen2', img2)
cv.waitKey(0)
cv.destroyAllWindows()
