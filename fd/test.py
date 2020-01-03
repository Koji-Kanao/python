import cv2
import numpy as np

# face_cascade_path = './haarcascade_frontalface_default.xml'


# face_cascade = cv2.CascadeClassifier(face_cascade_path)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    cv2.imshow('video image', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = img[y: y + h, x: x + w]

    cv2.imshow('video image', img)
    key = cv2.waitKey(10)
    if key == 27:  # ESCキーで終了
        break

cap.release()
cv2.destroyAllWindows()
