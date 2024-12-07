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
    folder = f'images/{person}'
    if not os.path.exists(folder):
        continue
    for image_name in os.listdir(folder):
        img = cv.imread(os.path.join(folder, image_name))
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Détection des visages avec Haar Cascade
        face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in face_rects:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(people.index(person))

# Entraîner le modèle LBPH
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Sauvegarder le modèle pour éviter un nouvel entraînement à chaque exécution
recognizer.save('./face_recognizer.yml')

# Charger le modèle préexistant (si disponible)
if os.path.exists('./face_recognizer.yml'):
    recognizer.read('./face_recognizer.yml')

# Fonction de reconnaissance faciale sur une image de test
def recognize_face(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Détection des visages dans l'image
    face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in face_rects:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)
        
        # Calculer un seuil de confiance pour reconnaître ou non le visage
        if confidence < 100:  # Seulement si la confiance est faible (valeur ajustable)
            name = people[label]
            color = (0, 255, 0)  # Vert pour une reconnaissance réussie
        else:
            name = 'Inconnu'
            color = (0, 0, 255)  # Rouge pour un visage inconnu
        
        # Dessiner le rectangle et afficher le nom
        cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv.putText(img, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return img

# Charger l'image de test
img = cv.imread('data/Tester c/messi_test.jpg')
if img is not None:
    result_img = recognize_face(img)
    cv.imshow("Reconnaissance Faciale", result_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Erreur : Impossible de charger l'image.")
