# For IRIS
# Presets.deltatheta = Presets.THETA / Presets.ntheta;        % Angular width of each subdivision of the correlation window (angular sampling)
# Presets.percentiriswidth = 50;                              % the percent of the Iris width covered by correlation window
# Presets.prw = Presets.percentiriswidth/200;                 % Convert above to half and transform from a percentage (50% to 0.25)
# Presets.nrad = 300;                                         % radius sub-divisions rmax/nrad

# For ANALYSIS
# remove frames with bad correlation
# Presets.MinCorrelationFirstFrame = 0.4;                     % Minimum correlation to include in first frame relative torsion measurement
# Presets.MinCorrelationPrevioustFrame = 0.95;                % Minimum correlation to include in previous frame torsion measurement

# PERCENT_IRIS_WIDTH = 50

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
