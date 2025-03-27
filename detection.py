import cv2
import numpy as np
import time
import mediapipe as mp
from tensorflow.keras.models import load_model
import pygame

# Initialize pygame mixer
pygame.mixer.init()
drowsy_sound = pygame.mixer.Sound('alert.wav')
gaze_sound = pygame.mixer.Sound('alert.wav')

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Load emotion model
emotion_model = load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Thresholds
EYE_AR_THRESH = 0.23  # Drowsiness threshold
GAZE_THRESH = 0.35  # Gaze threshold (0.35-0.65 is center)


def eye_aspect_ratio(eye):
    p = np.array([(lm.x, lm.y) for lm in eye])
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]))


cap = cv2.VideoCapture(0)

# State variables
drowsy_start = None
gaze_away_start = None
alert_active = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_time = time.time()
    emotion = "Neutral"
    condition_corrected = False

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark

        # 1. Emotion Detection
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        face_roi = frame[int(min(ys) * frame.shape[0]):int(max(ys) * frame.shape[0]),
                   int(min(xs) * frame.shape[1]):int(max(xs) * frame.shape[1])]
        if face_roi.size > 0:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))
            emotion = emotion_labels[np.argmax(emotion_model.predict(gray[np.newaxis, :, :, np.newaxis] / 255.))]

        # 2. Drowsiness Detection
        left_eye = [lms[i] for i in LEFT_EYE]
        right_eye = [lms[i] for i in RIGHT_EYE]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        # 3. Gaze Detection
        gaze_x = np.mean([lm.x for lm in left_eye])

        # Check if conditions are corrected
        if alert_active:
            if ear >= EYE_AR_THRESH and (GAZE_THRESH <= gaze_x <= 1 - GAZE_THRESH):
                condition_corrected = True

        # Update timers if conditions persist
        if ear < EYE_AR_THRESH or gaze_x < GAZE_THRESH or gaze_x > 1 - GAZE_THRESH:
            if drowsy_start is None and ear < EYE_AR_THRESH:
                drowsy_start = current_time
            if gaze_away_start is None and (gaze_x < GAZE_THRESH or gaze_x > 1 - GAZE_THRESH):
                gaze_away_start = current_time
        else:
            drowsy_start = None
            gaze_away_start = None

    # Trigger alert after 60 seconds
    if not alert_active:
        if drowsy_start and current_time - drowsy_start >= 4:
            drowsy_sound.play()
            alert_active = "drowsy"
        elif gaze_away_start and current_time - gaze_away_start >= 4:
            gaze_sound.play()
            alert_active = "gaze"

    # Reset if user corrects behavior
    if condition_corrected:
        drowsy_start = None
        gaze_away_start = None
        alert_active = False

    # Display
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show active alerts
    if alert_active == "drowsy":
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif alert_active == "gaze":
        cv2.putText(frame, "GAZE AWAY ALERT!", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show timers if counting
    if drowsy_start and not alert_active:
        elapsed = int(current_time - drowsy_start)
        cv2.putText(frame, f"Drowsy: {elapsed}s", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    if gaze_away_start and not alert_active:
        elapsed = int(current_time - gaze_away_start)
        cv2.putText(frame, f"Gaze Away: {elapsed}s", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    cv2.imshow('Smart Alert System', frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()