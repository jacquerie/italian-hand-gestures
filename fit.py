# -*- coding: utf-8 -*-

import glob

import cv2 as cv
from sklearn.svm import LinearSVC
from sklearn.utils._joblib import dump


def gray2arr(f):
    return cv.imread(f, cv.IMREAD_GRAYSCALE).flatten()


X_train = [gray2arr(f) for f in sorted(glob.glob('data/train/**/*.jpg'))]
y_train = [0] * 1000 + [1] * 1000 + [2] * 1000

model = LinearSVC(random_state=0, tol=1e-5)
model.fit(X_train, y_train)

dump(model, 'model.joblib')
