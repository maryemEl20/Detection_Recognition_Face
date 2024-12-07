import cv2 as cv

# Charger le classificateur Haar pour la détection du visage
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Charger le fichier vidéo
video_path = './vedios/5725960-uhd_3840_2160_30fps.mp4' 
cap = cv.VideoCapture(video_path)

frame_rate = 2 
frame_count = 0

while True:
    # Lire chaque image du fichier vidéo
    check, frame = cap.read()

    # Si nous n'avons pas d'image (fin du fichier vidéo), arrêter la boucle
    if not check:
        print("Fin du vidéo.")
        break

    frame_count += 1

    if frame_count % frame_rate != 0:
        continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dessiner un rectangle vert autour de chaque visage détecté
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Couleur verte et épaisseur de 2 pixels

    # Redimensionner l'image pour ajuster l'affichage
    frame_resized = cv.resize(frame, (640, 360))  # تصغير الفيديو إلى حجم 640x360

    # Afficher l'image avec les rectangles dessinés
    cv.imshow('Détection des visages', frame_resized)

    # Détection de la touche 'q' pour quitter
    key = cv.waitKey(1) 
    if key == ord('q'):
        break

# Libérer la vidéo et fermer les fenêtres
cap.release()
cv.destroyAllWindows()
