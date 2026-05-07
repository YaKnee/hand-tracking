import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 5
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

# Shared frame updated by callback
latest_frame = None

def draw_landmarks_on_image(rgb_image, result):
    annotated_image = np.copy(rgb_image)

    if result.hand_landmarks:
        for i in range(len(result.hand_landmarks)):
            hand_landmarks = result.hand_landmarks[i]
            handedness = result.handedness[i]

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            h, w, _ = annotated_image.shape
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]

            text_x = int(min(x_coords) * w)
            text_y = int(min(y_coords) * h) - MARGIN

            cv2.putText(
                annotated_image,
                handedness[0].category_name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA
            )

    return annotated_image

def result_callback(result, output_image, timestamp_ms):
    global latest_frame
    rgb_image = output_image.numpy_view() # mediapipe img -> numpy
    latest_frame = draw_landmarks_on_image(rgb_image, result)

def main():
    global latest_frame

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=result_callback
    )

    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        timestamp_ms = int(time.time() * 1000)

        detector.detect_async(mp_image, timestamp_ms)

        # Display latest processed frame
        if latest_frame is not None:
            cv2.imshow("Hands", cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR))
        else:
            cv2.imshow("Hands", frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()