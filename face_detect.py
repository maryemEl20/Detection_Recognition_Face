#pylint:disable=no-member

import cv2 as cv

img = cv.imread('./data/Tester c/cristiano 7.jpg')
#cv.imshow('cristiano ', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#cv.imshow('Gray cristiano', gray)

#le fichier Haarcascades contient un modèle de détection entraîné, 
# qui est l'élément clé pour effectuer des détections précises
# dans l'applications de vision par ordinateur.

haar_cascade = cv.CascadeClassifier('./data/Haarcascades/haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)



cv.waitKey(0)