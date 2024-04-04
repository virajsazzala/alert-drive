import cv2
import dlib
import time
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('eye-status.log')])

def calc_eye(eye):
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    return (a + b) / (2 * c)


def draw_eyes(frame, eye):
    cv2.polylines(frame, [eye], isClosed=True, color=(0, 255, 0), thickness=1)


face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("data/shape-pred-face-landmarks.dat")

cap = cv2.VideoCapture(0)

eyes_detected = False
eyes_closed_timer = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if not eyes_detected:
        for face in faces:
            landmarks = landmark_predictor(gray, face)
            left_eye = np.array(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            )
            right_eye = np.array(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            )

            ear_avg = (calc_eye(left_eye) + calc_eye(right_eye)) / 2

            if ear_avg < 0.2:
                eyes_closed_timer += 1
                if eyes_closed_timer > 5:
                    logging.debug("Eyes Closed")
                    # To trigger chat model
            else:
                eyes_closed_timer = 0

            # DEBUG: eye lines display
            draw_eyes(frame, left_eye)
            draw_eyes(frame, right_eye)

    cv2.imshow("alert drive", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()