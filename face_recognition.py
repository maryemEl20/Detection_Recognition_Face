import cv2 as cv
import numpy as np
import os

# Charger le classificateur Haar
haar_cascade = cv.CascadeClassifier('./data/Haarcascades/haar_face.xml')

# Liste des personnes à reconnaître
people = ['Cristiano Ronaldo', 'Lionel Messi']

# Entraînement simplifié des visages
faces, labels = [], []
for person in people:
    folder = f'./images/{person}'
    if not os.path.exists(folder):
        continue
    for image_name in os.listdir(folder):
        img = cv.imread(os.path.join(folder, image_name))
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in face_rects:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(people.index(person))

# Entraîner le modèle LBPH
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Reconnaissance faciale sur une image de test
def recognize_face(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in face_rects:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)
        name = people[label]
        
        # Dessiner le rectangle et afficher le nom
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(img, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return img

# Charger l'image de test
img = cv.imread('./data/Tester c/test 1.jpg')
if img is not None:
    result_img = recognize_face(img)
    cv.imshow("Reconnaissance Faciale", result_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Erreur : Impossible de charger l'image.")
