import cv2
import mediapipe as mp

# Initialisation des modules Mediapipe pour la détection des visages
mp_face_detection = mp.solutions.face_detection

# Ouverture de la webcam (le 0 indique généralement la webcam par défaut)
cam = cv2.VideoCapture(0)

# Vérification de la connexion de la caméra
if not cam.isOpened():
    print("Erreur: Impossible d'accéder à la caméra.")
    exit()

# Création de l'objet FaceDetection avec un seuil de confiance minimal
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        # Lecture d'une image depuis la caméra
        check, image = cam.read()
        if not check:
            print("Erreur: Impossible de lire l'image.")
            break

        # Conversion de l'image en format RGB (nécessaire pour Mediapipe)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Détection des visages dans l'image
        results = face_detection.process(rgb_image)

        # Vérification si des visages ont été détectés
        if results.detections:
            for detection in results.detections:
                # Récupérer les coordonnées du rectangle autour du visage
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Dessiner un rectangle vert autour du visage
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectangle vert

        # Affichage de l'image avec les visages détectés
        cv2.imshow('La webcam', image)

        # Détection de la touche 'q' pour quitter
        key = cv2.waitKey(1) 
        if key == ord('q'):
            break

# Libération des ressources de la caméra et fermeture des fenêtres
cam.release()
cv2.destroyAllWindows()
