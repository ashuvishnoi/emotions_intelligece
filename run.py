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
    frame = gaze.get(frame)

    cv2.imshow("Demo", frame)
    a.append(emotion)

    if cv2.waitKey(1) == 27:
        print(a)
        break
