import cv2
import numpy as np

FACE_CASCADE_PATH = './haarcascade_frontalcatface.xml'

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

capture = cv2.VideoCapture(0)


while True:
    ret, frame = capture.read()

    #cv2.imshow('frame', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (250, 0, 0), 2)
        face = frame[y: y+h, x:x+w]
        face_gray = gray[y: y+h, x: x+w]

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


# close window
capture.release()
cv2.destroyAllwindows()
