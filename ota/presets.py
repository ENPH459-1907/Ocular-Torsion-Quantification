# ====================== #
# TORSION QUANTIFICATION #
# ====================== #

from math import pi

# Angular span of the correlation window
ANGULAR_WINDOW_SPAN = 45*pi/180

# Number of angular subdivisions of the correlation window
NUM_ANGULAR_SUBDIVS = 450

# Correlation start at the 1/4 mark
WINDOW_START = round(NUM_ANGULAR_SUBDIVS / 4 + 1)

# Correlation end at the 1/2 mark
WINDOW_END = WINDOW_START + (NUM_ANGULAR_SUBDIVS / 2)

WINDOW_LENGTH = WINDOW_END - WINDOW_START

# Number of shifts of the cross correlation window
WINDOW_SHIFTS = NUM_ANGULAR_SUBDIVS  - (NUM_ANGULAR_SUBDIVS  / 2)

REQUIRED_CORR_FIRST_FRAME = 0.4
REQUIRED_CORR_PREV_FRAME = 0.95


MAX_ANGLE = 25
