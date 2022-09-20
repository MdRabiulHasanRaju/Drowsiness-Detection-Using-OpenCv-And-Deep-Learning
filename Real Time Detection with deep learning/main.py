from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from pygame import mixer
mixer.init()
sound = mixer.Sound('alarm.wav')
from keras.models import load_model
import tensorflow as tf
new_model = load_model('C:\\Users\\user\\Desktop\\Thesis\\Models\\full_model.h5')
model_ib = load_model('C:\\Users\\user\\Desktop\\Thesis\\Models\\model_IB.h5')
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 18
COUNTER = 0
thicc = 2

print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

test = 0
path = 'C:/Users/user/Desktop/PCIU/thesis paper/Drowsiness/datasets/Drowsiness-and-Yawn-Detection-with-voice-alert-using-Dlib-master/img'
counter = 0

classes = [
          "safe_driving",
          "texting-right",
          "talking_on_the_phone-right", 
          "texting-left", 
          "talking_on_the_phone-left",
          "operating_the_radio",
          "drinking",
          "reaching_behind",
          "hair-and-makeup",
          "talking_to_passenger"
          ]


while True:
    frame = vs.read()
    height, width = frame.shape[:2]
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    IB_Img = cv2.resize(frame, (224, 224))
    IB_Img = IB_Img/255
    IB_Img = np.expand_dims(IB_Img, 0)
    pred=(model_ib.predict(IB_Img)> 0.5)*1
    if np.array(pred[0][0]) == 1:
        print(classes[0])
    if np.array(pred[0][1]) == 1:
        print(classes[1])
    if np.array(pred[0][2]) == 1:
        print(classes[2])
    if np.array(pred[0][3]) == 1:
        print(classes[3])
    if np.array(pred[0][4]) == 1:
        print(classes[4])
    if np.array(pred[0][5]) == 1:
        print(classes[5])
    if np.array(pred[0][6]) == 1:
        print(classes[6])
    if np.array(pred[0][7]) == 1:
        print(classes[7])
    if np.array(pred[0][8]) == 1:
        print(classes[8])
    if np.array(pred[0][9]) == 1:
        print(classes[9])
        
    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        x2, y2 = x + w, y + h
        x = x
        y = y
        x2 = x2
        y2 = y2
        takeface = frame[y:y2,x:x2]
        if takeface.any():
            result = cv2.resize(takeface, (224, 224))
            result = result/255
            result = np.expand_dims(result, 0)
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            yhat = new_model.predict(result)
            if yhat > 0.5: 
                cv2.putText(frame, "Your Are Not Drowsy", (210, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                counter=0
            else:
                counter +=1
                if(counter>30):
                    cv2.putText(frame, "Your Are Drowsy", (210, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if(counter > 30):
                    try:
                        sound.play()
                    except:  
                        pass
                    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('Drowsy and Yawn', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
