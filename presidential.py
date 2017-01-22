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
cv2.waitKey()




# Optional - PIL conversions
pilsq = Image.open(fname)

# PIL -> cv2
sq2 = np.asarray(pilsq)
sq2 = cv2.cvtColor(sq2, cv2.COLOR_RGB2BGR)
# cv2 -> PIL
pilsq2 = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
pilsq2 = Image.fromarray(pilsq2)


