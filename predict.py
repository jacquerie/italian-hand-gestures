# -*- coding: utf-8 -*-

from collections import Counter, deque

import cv2 as cv
from sklearn.utils._joblib import load

RECTANGLE_COLOR = (69, 53, 220)
TEXT_COLOR = (41, 37, 33)

NUMBER_OF_ROIS = 29

Q_KEY = ord('q')
W_KEY = ord('w')
E_KEY = ord('e')
ESC_KEY = 27


def putText(frame, text):
    cv.putText(frame, text, (580, 90), cv.FONT_HERSHEY_COMPLEX, 1, TEXT_COLOR, 4, cv.LINE_AA)


model = load('model.joblib')

capture = cv.VideoCapture(1)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

bg_subtractor = cv.createBackgroundSubtractorMOG2()

prediction = -1
rois = deque([], maxlen=NUMBER_OF_ROIS)
show_magic = False
show_prediction = False

while True:
    _, frame = capture.read()
    frame = cv.flip(frame, 1)
    cv.rectangle(frame, (520, 120), (1160, 600), RECTANGLE_COLOR, 8)
    if show_prediction and len(rois) == NUMBER_OF_ROIS:
        el, count = Counter(model.predict(rois)).most_common()[0]
        if count > NUMBER_OF_ROIS / 2:
            prediction = el
            if prediction == 0:
                putText(frame, 'I AM REALLY ANGRY WITH YOU!')
            elif prediction == 1:
                putText(frame, 'THAT WAS REALLY A NICE JOB!')
            elif prediction == 2:
                putText(frame, 'WHAT THE F*CK DO YOU WANT?')

    roi = frame[120:600, 520:1160]
    roi = bg_subtractor.apply(roi)
    rois.append(roi.flatten())
    if show_magic:
        frame[120:600, 520:1160] = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)

    cv.imshow('Italian Hand Gestures / Predict', frame)

    key = cv.waitKey(16) & 255
    if key == Q_KEY:
        show_prediction = not show_prediction
    elif key == W_KEY:
        prediction = -1
        rois = deque([], maxlen=NUMBER_OF_ROIS)
    elif key == E_KEY:
        show_magic = not show_magic
    elif key == ESC_KEY:
        break

capture.release()
cv.destroyAllWindows()
