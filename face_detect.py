import cv2
import mediapipe as mp

# Initialisation de Mediapipe pour la détection du visage et l'extraction des points du visage
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Chargement de l'image
img = cv2.imread('data/Tester c/cristiano 4.jpg')


# Conversion de l'image en RGB, car Mediapipe nécessite ce format
rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Détection du visage avec FaceDetection
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    results = face_detection.process(rgb_img)

    if results.detections:
        # Si un visage est détecté, dessiner un rectangle autour du visage
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape #Longueur et largeur
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            # Dessiner le rectangle autour du visage
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 

# Extraction des points du visage avec FaceMesh
with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.2, min_tracking_confidence=0.2) as face_mesh:
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Récupération des points pour les yeux, le nez et la bouche
            #
            left_eye = face_landmarks.landmark[33]  # Point pour l'œil gauche
            nose = face_landmarks.landmark[1]  # Point pour le nez
            mouth = face_landmarks.landmark[13]  # Point pour la bouche

            ih, iw, _ = img.shape

            # Calcul des coordonnées pour l'œil gauche
            left_eye_points = [33, 133, 173, 153, 144, 163, 157, 158]  # Liste des indices pour l'œil gauche
            left_eye_x_min = min([int(face_landmarks.landmark[p].x * iw) for p in left_eye_points])
            left_eye_x_max = max([int(face_landmarks.landmark[p].x * iw) for p in left_eye_points])
            left_eye_y_min = min([int(face_landmarks.landmark[p].y * ih) for p in left_eye_points])
            left_eye_y_max = max([int(face_landmarks.landmark[p].y * ih) for p in left_eye_points])

            # Dessiner un rectangle autour de l'œil gauche
            #run 2 L'épaisseur des bords du rectangle en pixels.
            cv2.rectangle(img, (left_eye_x_min - 10, left_eye_y_min - 10),
                          (left_eye_x_max + 10, left_eye_y_max + 10), (255, 0, 0), 2)

            # Dessiner un rectangle autour du nez
            nose_x, nose_y = int(nose.x * iw), int(nose.y * ih)
            cv2.rectangle(img, (nose_x - 40, nose_y - 40), (nose_x + 40, nose_y + 40), (0, 255, 0), 2)

            # Dessiner un rectangle autour de la bouche
            mouth_x, mouth_y = int(mouth.x * iw), int(mouth.y * ih)
            cv2.rectangle(img, (mouth_x - 40, mouth_y - 40), (mouth_x + 40, mouth_y + 40), (0, 0, 255), 2)

# Affichage de l'image avec les rectangles dessinés
cv2.imshow('Face and Landmarks', img)
cv2.waitKey(0)
cv2.destroyAllWindows()