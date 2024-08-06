import cv2 as cv 
img = cv.imread('/home/likcos/Imágenes/tr.png', 1)
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2RGB)        
cv.imshow('ejemplo', img)
cv.imshow('ejemploGris', imgHSV)
cv.waitKey(0)
cv.destroyAllWindows()
