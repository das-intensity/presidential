#!/usr/bin/python2

import os, sys

import cv2
import numpy as np
from PIL import Image # Optional

fname = 'squirrel.png'

# Read the squirrel image
sq = cv2.imread(fname)
print sq.shape

cv2.imshow('squirrel', sq)

sq_gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
cv2.imshow('squirrel gray', sq_gray)

sq_gray_mask = sq_gray > 0
sq_mask = np.array([[[v, v, v] for v in row] for row in sq_gray_mask])
#cv2.waitKey()




# Optional - PIL conversions
pilsq = Image.open(fname)

# PIL -> cv2
sq2 = np.asarray(pilsq)
sq2 = cv2.cvtColor(sq2, cv2.COLOR_RGB2BGR)
# cv2 -> PIL
pilsq2 = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
pilsq2 = Image.fromarray(pilsq2)





# Load face detection cascade
print 'loading face cascade...'
cas_dir = '/usr/share/opencv/haarcascades/'
cas_fname = 'haarcascade_frontalface_default.xml'
#cas_fname = 'haarcascade_frontalface_alt.xml'
cas = cv2.CascadeClassifier(os.path.join(cas_dir, cas_fname))


# Webcam viewing
print 'starting webcam...'
cam = cv2.VideoCapture(0)
ret, frame = cam.read()
assert ret
print 'webcam res:', frame.shape

while True:
    ret, frame = cam.read()

    # overlay the squirrel
    np.copyto(frame[-sq.shape[0]:, 0:sq.shape[1], :], sq, where=sq_mask)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cas.detectMultiScale(
            frame_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
            # v2.4 -> cv2.cv.CV_HAAR_SCALE_IMAGE
            )
    #print faces

    #print 'frame: %s - %s' % (frame.shape, frame.dtype)

    for (x, y, w, h) in faces:
        # First augment it a little, because the original rect isn't very good
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        x0 = x
        y0 = y
        w0 = w
        h0 = h


        # Extend the box a bit
        #y -= int(h * 0.1)
        h += int(h * 0.3)
        x -= int(w * 0.1)
        w += int(w * 0.2)
        if y < 0 or y+h > frame.shape[0]:
            continue
        if x < 0 or x+w > frame.shape[1]:
            continue

        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Make more presidential colour
        # Get skin values
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        face_hsv = frame_hsv[y:y+h, x:x+w, :]

        hsv_min = np.array([0, 0, 40])
        hsv_max = np.array([200, 120, 220])
        face_inrange = cv2.inRange(face_hsv, hsv_min, hsv_max)
        face_inrange = np.array([[[v, v, v] for v in row] for row in face_inrange], dtype=bool)
        #print 'face_inrange: %s - %s' % (face_inrange.shape, face_inrange.dtype)

        pres_skin = [0.2, 0.65, 1.7]
        face_bgr = frame[y:y+h, x:x+w, :] * pres_skin
        face_bgr = np.minimum(face_bgr, 255)
        face_bgr = np.array(face_bgr, dtype='uint8')
        #print 'face_bgr: %s - %s' % (face_bgr.shape, face_bgr.dtype)
        np.copyto(frame[y:y+h, x:x+w, :], face_bgr, where=face_inrange)



    cv2.imshow('cam', frame)
    key = cv2.waitKey(1)
    if key != -1:
        break


