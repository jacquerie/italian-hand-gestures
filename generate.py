# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import cv2 as cv

RECTANGLE_COLOR = (69, 53, 220)

CLASS = 'horn'
LIMIT = 1000

E_KEY = ord('e')
R_KEY = ord('r')
ESC_KEY = 27


capture = cv.VideoCapture(1)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

bg_subtractor = cv.createBackgroundSubtractorMOG2()

show_magic = False
i, start_writing = 0, False

while True:
    _, frame = capture.read()
    frame = cv.flip(frame, 1)
    cv.rectangle(frame, (520, 120), (1160, 600), RECTANGLE_COLOR, 8)

    roi = frame[120:600, 520:1160]
    roi = bg_subtractor.apply(roi)
    if show_magic:
        frame[120:600, 520:1160] = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)

    cv.imshow('Italian Hand Gestures / Generate', frame)
    if start_writing:
        cv.imwrite(f'data/train/{CLASS}s/{CLASS}{i:03}.jpg', roi)
        i += 1
        if i == LIMIT:
            break

    key = cv.waitKey(16) & 255
    if key == E_KEY:
        show_magic = not show_magic
    elif key == R_KEY:
        start_writing = not start_writing
    elif key == ESC_KEY:
        break

capture.release()
cv.destroyAllWindows()
