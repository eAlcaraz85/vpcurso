import cv2 as cv
import numpy as np
img = cv.imread('/home/likcos/Imágenes/mo1.png',0)
h,w = img.shape[:2]
img2 = np.zeros((h*2, w*2, 1) , dtype = "uint8")
print("Valores " + str(img.shape[:2]))
for i in range(h):
    for j in range(w):
        img2[int(i+20),int(j+10)]=img[i,j]

cv.imshow('imagen', img)
cv.imshow('imagen2', img2)
cv.waitKey(0)
cv.destroyAllWindows()
