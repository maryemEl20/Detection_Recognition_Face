import cv2 as cv
import mediapipe as mp

# Initialisation de Mediapipe pour la détection des visages
mp_face_detection = mp.solutions.face_detection

# Ouverture du fichier vidéo
video_path = './vedios/vedio 1.mp4'
cap = cv.VideoCapture(video_path)

# Initialisation de la détection de visages avec une confiance de détection minimale de 0.5
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionnement de l'image pour améliorer les performances
        frame = cv.resize(frame, (640, 360))

        # Conversion de l'image BGR en RGB (format requis par Mediapipe)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Traitement de l'image pour détecter les visages
        results = face_detection.process(frame_rgb)

        # Si des visages sont détectés, on dessine un rectangle autour de chaque visage
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                # Dessiner un rectangle vert autour du visage détecté
                cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Affichage de l'image avec la détection des visages
        cv.imshow('Mediapipe Face Detection', frame)

        # Appuyer sur 'q' pour quitter la boucle
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

# Libération de la vidéo et fermeture des fenêtres
cap.release()
cv.destroyAllWindows()
