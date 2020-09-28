#Project Description

    Facial Emotion Recognition using PyTorch.

    It creates a bounding box around the face of the person present in the picture and put a text
    at the top of the bounding box representing the recognised emotion.

#



#Install

    pip install emotion_recognition
    
    
#

# Requirements

pytorch >= 1.2.0

torchvision >= 0.3.0
#

#Usage:

    from facial_emotion_recognition import EmotionRecognition
    
    import cv2 as cv
    
    er = EmotionRecognition(device='gpu', gpu_id=0)
    
    cam = cv.VideoCapture(0)
    
    success, frame = cam.read()
    
    frame = er.recognise_emotion(frame, return_type='BGR')
    
    cv.imshow('frame', frame)
    
    cv.waitkey(0)
    
#

#Arguments

    er = EmotionRecognition(device='gpu', gpu_id=0)
    
    device = 'gpu' or cpu'
    
    gpu_id will be effective only when more than two GPUs are detected or it will through error.
    
    frame = er.recognise_emotion(frame, return_type='BGR')
    
    return_type='BGR' or 'RGB'
#

#References

1. "Challenges in Representation Learning: A report on three machine learning
contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,
X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
Y. Bengio. arXiv 2013.

#
