import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

import cv2
import mediapipe as mp
import pyautogui
from math import sqrt

pyautogui.FAILSAFE = False

# Initialize variables
x1 = y1 = x2 = y2 = x3 = y3 = 0

# Set up the webcam
webcam = cv2.VideoCapture(1)
if not webcam.isOpened():
    print("Error: Webcam not accessible.")
    exit()

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Get screen size for mouse control
screen_width, screen_height = pyautogui.size()

# Define constants
control_area_scaling = 1.3
mouse_speed_factor = 1.4
zoom_threshold = 50
zoom_scale_factor = 0.05  # Adjust this for zoom sensitivity

def calculate_distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while True:
    ret, image = webcam.read()
    if not ret:
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)
    frame_height, frame_width, _ = image.shape

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            handedness = result.multi_handedness[idx].classification[0].label

            for id, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:  # Index finger tip
                    cv2.circle(image, (x, y), 8, (0, 255, 255), 3)
                    x1 = x
                    y1 = y
                if id == 4:  # Thumb tip
                    cv2.circle(image, (x, y), 8, (255, 255, 0), 3)
                    x2 = x
                    y2 = y
                if id == 12:  # Middle finger tip
                    cv2.circle(image, (x, y), 8, (0, 255, 0), 3)
                    x3 = x
                    y3 = y

            dist_thumb_index = calculate_distance(x2, y2, x1, y1)  # Distance between thumb and index finger
            dist_thumb_middle = calculate_distance(x2, y2, x3, y3)  # Distance between thumb and middle finger

            if handedness == "Right":
                if dist_thumb_middle > 70:
                    # Control mouse movement
                    screen_x = screen_width * (hand_landmarks.landmark[8].x - 0.5) * control_area_scaling + screen_width / 2
                    screen_y = screen_height * (hand_landmarks.landmark[8].y - 0.5) * control_area_scaling + screen_height / 2
                    current_mouse_x, current_mouse_y = pyautogui.position()
                    pyautogui.moveTo(current_mouse_x + (screen_x - current_mouse_x) * mouse_speed_factor, 
                                     current_mouse_y + (screen_y - current_mouse_y) * mouse_speed_factor)
                else:
                    # Adjust volume based on thumb and index finger distance
                    if dist_thumb_index > 15:
                        pyautogui.press("volumeup")
                    else:
                        pyautogui.press("volumedown")
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

            elif handedness == "Left":
                if dist_thumb_middle < 70:
                    # Zoom in/out based on thumb and index finger distance
                    zoom_level = (dist_thumb_index - zoom_threshold) * zoom_scale_factor
                    zoom_level = max(1, min(zoom_level, 2))  # Clamp zoom level between 1 and 2
                    zoomed_image = cv2.resize(image, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)
                    
                    # Display zoomed image
                    cv2.imshow("Zoomed Image", zoomed_image)
                    
                    # Draw line to indicate click action
                    cv2.line(image, (x1, y1), (x3, y3), (0, 0, 255), 5)

    cv2.imshow("Hand Volume Control and Mouse Tracker", image)

    if cv2.waitKey(10) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()
