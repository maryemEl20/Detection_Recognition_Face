import cv2 as cv
import mediapipe as mp

# Initialisation de Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Ouvrir la webcam
cap = cv.VideoCapture(0)  # '0' correspond à la webcam par défaut

# Configuration de Mediapipe pour la détection des visages
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Impossible d'accéder à la webcam.")
            break

        # Convertir l'image en RGB pour Mediapipe
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Effectuer la détection des visages
        results = face_detection.process(frame_rgb)

        # Si des visages sont détectés, dessiner des rectangles autour
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Dessiner un rectangle vert autour du visage
                cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Afficher la vidéo avec les rectangles
        cv.imshow('Détection des visages', frame)

        # Appuyez sur la touche 'q' pour quitter
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# Libérer la webcam et fermer les fenêtres
cap.release()
cv.destroyAllWindows()
