
import cv2
import numpy as np
import threading
import winsound
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode


MODEL_PATH      = r"C:\Users\ASUS\Downloads\drowsiness_final2.h5"
LANDMARKER_PATH = r"C:\Users\ASUS\Downloads\face_landmarker.task"
IMG_SIZE        = 64
CONSEC_FRAMES   = 6
EYE_PADDING     = 25

CLASS_NAMES = ["closed", "no_yawn", "open", "yawn"]

LEFT_EYE_IDX  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_IDX = [33,   7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

model = load_model(MODEL_PATH)


base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)
face_options = FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
landmarker = FaceLandmarker.create_from_options(face_options)

closed_counter = 0
yawn_counter   = 0
active_counter = 0
status         = "Active"
yawn_count     = 0
prev_yawn      = False
alert_playing  = False

#sound alert
def play_alert(alert_type="drowsy"):
    global alert_playing
    if alert_playing:
        return
    alert_playing = True

    def _beep():
        global alert_playing
        try:
            if alert_type == "drowsy":
                for _ in range(3):
                    winsound.Beep(1000, 400)
                    winsound.Beep(800, 200)
            else:
                winsound.Beep(600, 600)
        except Exception:
            pass
        alert_playing = False

    threading.Thread(target=_beep, daemon=True).start()


def get_pts(landmarks, indices, h, w):
    return np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
        dtype=np.int32
    )


def crop_region(frame, pts, padding, h, w):
    x1 = max(0, pts[:, 0].min() - padding)
    y1 = max(0, pts[:, 1].min() - padding)
    x2 = min(w, pts[:, 0].max() + padding)
    y2 = min(h, pts[:, 1].max() + padding)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess_roi(roi):
    if roi is None or roi.size == 0:
        return None
    gray       = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized    = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict_eye(tensor, threshold=0.6):
    if tensor is None:
        return "open", 0.0
    pred        = model.predict(tensor, verbose=0)[0]
    closed_prob = pred[0]
    open_prob   = pred[2]
    if max(closed_prob, open_prob) < threshold:
        return "open", open_prob * 100
    if closed_prob > open_prob:
        return "closed", closed_prob * 100
    else:
        return "open", open_prob * 100


def predict_yawn(tensor, threshold=0.5):
    if tensor is None:
        return "no_yawn", 0.0
    pred         = model.predict(tensor, verbose=0)[0]
    yawn_prob    = pred[3]
    no_yawn_prob = pred[1]
    if max(yawn_prob, no_yawn_prob) < threshold:
        return "no_yawn", no_yawn_prob * 100
    if yawn_prob > no_yawn_prob:
        return "yawn", yawn_prob * 100
    else:
        return "no_yawn", no_yawn_prob * 100


def main():
    global closed_counter, yawn_counter, active_counter
    global status, yawn_count, prev_yawn

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("NO Webcam Found")
        return

    print("press q to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = landmarker.detect(mp_image)

        eye_label   = "open"
        mouth_label = "no_yawn"
        left_label  = right_label = "open"
        left_conf   = right_conf  = mouth_conf = 0.0
        left_box    = right_box   = None

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_pts  = get_pts(landmarks, LEFT_EYE_IDX,  h, w)
            right_pts = get_pts(landmarks, RIGHT_EYE_IDX, h, w)

            left_roi,  left_box  = crop_region(frame, left_pts,  EYE_PADDING, h, w)
            right_roi, right_box = crop_region(frame, right_pts, EYE_PADDING, h, w)


            mouth_label, mouth_conf = predict_yawn(preprocess_roi(frame))


            if mouth_label != "yawn":
                left_label,  left_conf  = predict_eye(preprocess_roi(left_roi))
                right_label, right_conf = predict_eye(preprocess_roi(right_roi))

                eye_label = "closed" if (left_label == "closed" and right_label == "closed") else "open"
            else:
                eye_label = "open"


            if left_box is not None:
                bx1, by1, bx2, by2 = left_box
                eye_color = (0, 0, 255) if left_label == "closed" else (0, 255, 100)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), eye_color, 2)
                cv2.putText(frame, f"L: {left_label} {left_conf:.0f}%",
                            (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, eye_color, 1)

            if right_box is not None:
                bx1, by1, bx2, by2 = right_box
                eye_color = (0, 0, 255) if right_label == "closed" else (0, 255, 100)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), eye_color, 2)
                cv2.putText(frame, f"R: {right_label} {right_conf:.0f}%",
                            (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, eye_color, 1)


            f_color = (0, 165, 255) if mouth_label == "yawn" else (200, 200, 200)
            cv2.putText(frame, f"Yawn: {mouth_label} {mouth_conf:.0f}%",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, f_color, 2)

        else:
            cv2.putText(frame, "Face detect nahi hua",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 165, 255), 2)


        if mouth_label == "yawn":
            yawn_counter += 1
            if yawn_counter >= CONSEC_FRAMES:
                if not prev_yawn:
                    yawn_count += 1
                    play_alert("yawn")
                prev_yawn = True
                status = "YAWNING!"
        else:
            yawn_counter = 0
            prev_yawn    = False


        if eye_label == "closed":
            closed_counter += 1
            active_counter  = 0
            if closed_counter >= CONSEC_FRAMES:
                status = "SLEEPING!"
                play_alert("drowsy")
        else:
            active_counter += 1
            closed_counter  = 0
            if active_counter >= CONSEC_FRAMES and mouth_label != "yawn":
                status = "Active"


        if status == "SLEEPING!":
            status_color = (0, 0, 255)
        elif status == "YAWNING!":
            status_color = (0, 165, 255)
        else:
            status_color = (0, 255, 0)

        cv2.rectangle(frame, (0, 0), (420, 55), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {status}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, status_color, 2)
        cv2.putText(frame, f"Yawn Count: {yawn_count}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Eye: {eye_label}  |  Mouth: {mouth_label}",
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1)

        if status == "SLEEPING!":
            cv2.rectangle(frame, (0, 125), (w, 165), (0, 0, 180), -1)
            cv2.putText(frame, "DROWSINESS ALERT!",
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()