from facial_emotion_recognition.facial_emotion_recognition import EmotionRecognition

import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()

er = EmotionRecognition(device='cpu')

webcam = cv2.VideoCapture(0)

a = []

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # get emotion
    frame, emotion = er.recognise_emotion(frame, return_type='BGR')

    # get gaze detection
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)
    a.append(emotion)

    if cv2.waitKey(1) == 27:
        print(a)
        break

