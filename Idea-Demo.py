
# Import Packages
from autopy import screen, mouse
import math
# import time
import cv2
from numpy import interp
from mediapipe.python.solutions.hands import Hands


# Colors
# RED = (0, 0, 255)
# GREEN = (0, 255, 0)

"""
# Draw Rectangle Frame Of Mouse Portion
def mouse_frame(col):
    cv2.rectangle(img, (frameRX-offSet, frameRY-offSet), (width_frame+offSet, height_frame+offSet), col, 5)


# Draw Circle Around Mouse Pointer
def mouse_pointer(pt, col):
    cv2.circle(img, pt, 10, col, cv2.FILLED)
"""

# Screen & Display Resolutions
width_display, height_display = screen.size()
# print(width_display, height_display)

# Camera Capture
cap = cv2.VideoCapture(0)

# Set Display Resolutions
width_cam, height_cam = 640, 480
# print(width_cam, height_cam)
cap.set(3, width_cam)
cap.set(4, height_cam)

# Hand Detection Object
hands = Hands()

# Time Counter
present_time = 0

# Smoothening
smoothen = 6
previous_x, previous_y = 0, 0
current_x, current_y = 0, 0

# Main Loop
while True:

    # Image Read From Captured Video
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find Hand Landmarks
    # lmList, img = find_hands(img)
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w, c = img.shape
    my_lm_list = []
    handType = "Right"

    if results.multi_hand_landmarks:
        for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
            for lm in handLms.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                my_lm_list.append([px, py])

    # Get Tip Of Index & Middle Fingers
    if my_lm_list:
        x1, y1 = my_lm_list[8]
        x2, y2 = my_lm_list[12]
        # print(x1, y1, x2, y2)

        # Check which finger are up
        tipIds = [4, 8, 12, 16, 20]
        finger_up = []

        # Thumb
        if handType == "Right":
            if my_lm_list[tipIds[0]][0] > my_lm_list[tipIds[0] - 1][0]:
                finger_up.append(1)
            else:
                finger_up.append(0)
        else:
            if my_lm_list[tipIds[0]][0] < my_lm_list[tipIds[0] - 1][0]:
                finger_up.append(1)
            else:
                finger_up.append(0)

        # 4 Fingers
        for f in range(1, 5):
            if my_lm_list[tipIds[f]][1] < my_lm_list[tipIds[f] - 2][1]:
                finger_up.append(1)
            else:
                finger_up.append(0)

        # print(finger_up)

        # Frame Reduction
        frameRX, frameRY = 150, 150
        offSet = 20
        width_frame, height_frame = width_cam - frameRX, height_cam - frameRY
        # mouse_frame(RED)
        # mouse_pointer((x1, y1), RED)

        # Only index finger in moving mode
        if finger_up[1] == 1 and finger_up[2] == 0:

            # Convert Co-Ordinates
            x3 = interp(x1, (frameRX, width_frame), (0, width_display))
            y3 = interp(y1, (frameRY, height_frame), (0, height_display))

            # Smoothen Values
            current_x = previous_x + (x3 - previous_x) / smoothen
            current_y = previous_y + (y3 - previous_y) / smoothen

            # Move Mouse
            mouse.move(current_x, current_y)
            previous_x, previous_y = current_x, current_y

        # Both Index & Middle Fingers Up For Clicking Mode
        if finger_up[1] == 1 and finger_up[2] == 1:
            # Find Distance Between Index & Middle Fingers
            length = math.hypot(x2 - x1, y2 - y1)
            # mouse_frame(GREEN)
            # mouse_pointer(my_lm_list[8], GREEN)
            # mouse_pointer(my_lm_list[12], GREEN)

            # Distance Short Between Index & Middle Fingers
            if length < 30:
                # Click Mouse
                mouse.click()
                # mouse_frame(RED)
                # mouse_pointer(my_lm_list[8], RED)
                # mouse_pointer(my_lm_list[12], RED)

    # Frame Rate
    # current_time = time.time()
    # fps = 1 / (current_time - present_time)
    # present_time = current_time
    # fps = str(int(fps))
    # cv2.putText(img, fps, (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display
    # cv2.im show("AI Virtual Mouse", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
