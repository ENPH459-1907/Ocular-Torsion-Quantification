'''
Method to calculate the 2D cross correlation between two images.
'''

import cv2
import numpy as np
import time
from math import pi
from scipy import interpolate, optimize
import matplotlib.pyplot as plt

MAX_ROTATION_ANGLE = 25

# ================ #
# == EXCEPTIONS == #
# ================ #

class InputParameterError(Exception):
    '''
    General error for correlation methods with incorrect parameters.
    '''
    def __init__(self, message):
        self.message = message

class CorrelationBelowThreshold(Exception):
    '''
    Correlation must be above a certain threshold.
    '''
    def __init__(self, message):
        self.message = message

class LackingInterpPoints(Exception):
    '''
    Need more then 3 points in order to interpolate.
    '''
    def __init__(self, message):
        self.message = message

class LengthMismatch(Exception):
    '''
    Arrays must be the same length.
    '''
    def __init__(self, message):
        self.message = message

def xcorr2d(
    iris_seg,
    reference_window,
    start=0,
    prev_deg=None,
    torsion_mode='interp',
    resolution=1,
    threshold=0,
    max_angle=25,
    verborose=False,
     **kw):
    '''
    Performs a pseduo 2D cross correlation method to calculate the relative shift
    between features in the given iris segment and a reference window. A moving
    window is taken from the current iris segment and the 2D correlation coefficient
    is calculated between the two matrices. Relative to other image convolution
    methods, in this function the window only moves along one axis (columns) but the
    correlation coefficient is calculated in two dimensions.

    There are two different methods of performing the cross correlation: 'full' or
    'subset'. In both methods, the maximum torsion is limited to +- 25 degrees to
    decrease the total number of comparisions required. The 'subset' method involves
    less comparisions but is less robust then the 'full' method as less points are
    used to calculate the correlation coefficient.

    'subset' - A smaller subset of the original segment is considered as the reference
        window. A window of the same size is taken from the current iris segement
        and the correlation coefficient is then computed.'

    'full' - The reference window is extended by the maximum torsion angle (25 deg)
        and the full iris segment is used as the moving window. The size of the
        moving window is equal to the size of the iris segment.

    There are two modes for quantifying the torsion: 'interp' and 'upsample'.

    'interp' - Interpolates the cross correlation coefficient values.
        See 'corr_interp' for more details.

    'upsample' - Used when passing an iris segment that has been upsampled.
        To accurately calculate the torsion, the resulotion must be equal to the
        upsampled resolution or the iris segment.

    See 'corr_upsample'
        for more details.

    Parameters
    ------------------------
    iris_seg : 2D array_like NxM
        Unwrapped iris segment, in polar coordinates.

    reference_window : 2D array_like NxK
        Intial window used as the reference for all rotation.

        Depending on the size of the reference window relative to the iris_seg,
        a different torsion_mode will be used to perform the cross correlation.

        'subset' - If K < M then a window of size NxK is taken from the iris
            segment and compared against the reference window. In total there
            is M - K possible windows available from the iris segment.

        'full' - If K >= M then the window will be the iris segement and the
            size of the window will be unmodified. The size of reference
            window should be K = M + 2 * max angle of torsion. The iris segement is
            then shifted 2 * max angle across the the reference window in order to
            perform the cross correlation.

    start : int, default = 0
        Starting index location of the window. Leave start as 0 for 'full' method.

    prev_deg : float
        Previous rotation angle. If an error occurs, the method will default
        to prev_deg.

    torsion_mode : optional, str {'interp', 'upsample'}
        Select the operation torsion_mode. 'interp' will interpolate the Correlation
        result. 'upsample' assumes the input iris_seg has been upsampled by a
        specific resolution.

    resolution : float, 0 < x < 1, default = 1
        'interp' - The resolution will define dx for the interpolated function.
        'upsample' - The resolution corresponds to the upsampled resolution.

    threshold : float, 0 =< x < 1, default = 0
        Minimum correlation value of correlation, all correlation values less
        then threshold will be ignored.

    max_angle : float 0 =< x < 360, default = 25
        The maximum angle of rotation of the eye. Only values within + and -
        max_angle will be considered.

    Returns
    ------------------------
    deg : float (in degrees)
        The amount of rotation of the iris relative to the reference window.
    '''

    corrs = []

    # Interpolation:
    # threshold = 0.4
    # resolution = 0.01

    # TODO kwargs?
    # WINDOW_LENGTH : int, default = len(first_window)
    #     The length of the window or the number of columns of the window.
    #
    # WINDOW_SHIFTS : int, default = len(iris_seg) - WINDOW_LENGTH
    #     The maximum number of shifts of the window to fully cover
    #     the iris segment.

    # constants
    WINDOW_SHIFTS = 0
    WINDOW_LENGTH = 0
    WINDOW_OFFSET = 0

    # hold the number of shifts here
    shifts = None

    # factor to update index locations for upsample methods
    upsample_factor = 1

    if torsion_mode == 'interp':
        pass
    elif torsion_mode == 'upsample':
        upsample_factor = resolution
    else:
        raise InputParameterError("Mode {} is not supported. torsion_mode={'interp', 'upsample'}".format(torsion_mode))

    max_angle = int(max_angle / upsample_factor)

    # determine the method given the number of columns on the input matrices
    # set the window length, shifts and start given the method
    method = ''

    # FULL METHOD
    if reference_window.shape[1] > iris_seg.shape[1]:
        method = 'full'
        WINDOW_LENGTH = iris_seg.shape[1]
        WINDOW_SHIFTS = reference_window.shape[1] - iris_seg.shape[1]

        # update start to account for the extension of the frame size by max_angle
        start = int(WINDOW_SHIFTS / 2)

        # force max_angle to be
        if max_angle != abs(start):
            print('WARNING: 2 * Max angle ({}) does not equal the total amount of extended degres to the reference window ({})'.format(2 * max_angle,  WINDOW_SHIFTS))
            max_angle = start
    # SUBSET METHOD
    else:
        method = 'subset'
        # raise an error if start is less then max_angle (this will break array-access later)
        if start < max_angle:
            raise InputParameterError('The start index ({}) must be greater then the max_angle / upsample resolution ({}/{} = {}).'.format(start, max_angle * upsample_factor, upsample_factor, max_angle))

        WINDOW_LENGTH = reference_window.shape[1]
        WINDOW_SHIFTS = max_angle * 2

        # to go through every possible shift
        # WINDOW_SHIFTS = iris_seg.shape[1] - WINDOW_LENGTH

    # defaults for upper and lower bound
    lb = abs(start) - max_angle
    ub = abs(start) + max_angle
    shifts = range(lb, ub)

    # override values with kw args
    WINDOW_LENGTH = kw.get('WINDOW_LENGTH', WINDOW_LENGTH)
    if kw.get('WINDOW_SHIFTS'):
        shifts = range(kw.get('WINDOW_SHIFTS'))

    for j in shifts:
        corr = 0
        # calculate the correlation coefficient between the current window and
        # reference window
        if method == 'full':
            # take a subset of the extended reference window
            corr = corr2_coeff(iris_seg, reference_window[:, j:j + WINDOW_LENGTH])
        else:
            # take a window from the current iris polar segment
            corr = corr2_coeff(iris_seg[:, j:j + WINDOW_LENGTH], reference_window)

        # save the correlation result
        corrs.append(corr)

    # TODO
    # in the result of an error, return prev_deg
    deg = prev_deg

    if len(corrs) <= 2 * max_angle:
        x, y = reduced_corr(corrs, threshold, offset=lb)
    else:
        x, y = reduced_corr(corrs, threshold, lb, ub)

    # calculate the torsion (in degrees) depending on the method
    if torsion_mode == 'interp':
        deg = corr_interp(x, y, start, resolution)
    elif torsion_mode == 'upsample':
        deg = corr_upsample(x, y, start, resolution)

    if method == 'full':
        deg = -1*deg

    return deg

def corr_interp(x, y, start, interp_resolution, kind='quadratic'):
    '''
    Interpolate a list of correlation values and return the amount of rotation
    on a finer grid.

    Parameters
    ------------------------
    x : array_like
        Indices location of the accepted correlation values relative to the
        original location in corrs.

    y : array_like
        Correlation values that are between lb and ub that are above the threshold.

    start : int
        Starting index location of the window.

    interp_resolution : float
        Resultion of the interpolated correlation function.

    kind : optional, str
        Type of piecewise interpolation function. See scipy.interpolate.interp1d
        for more details.

    Returns
    ------------------------
    deg : float
        The amount of rotation after interpolation.
    '''

    # get a list of correlation values between the maximum allowed rotation
    # only include values above the threshold value
    # lb = 0
    # ub = 0
    # if start < max_angle:
    #     lb = start - max_angle
    #     ub = start + max_angle
    # else:
    #     lb = 2 * max_angle
    #     ub = 0
    # x, y = reduced_corr(corrs, start - max_angle, start + max_angle, threshold)

    if len(y) <= 3:
        raise LackingInterpPoints('Only have {} points. Need at least 4.'.format(len(y)))

    try:
        f = interpolate.interp1d(x, y, kind=kind)
    except ValueError as e:
        print('Unhandled error in the scipy interp1d method.')
        raise

    # create a list of fine x values with the step size (resolution)
    xnew = np.arange(x[0], x[-1], interp_resolution)

    # calculate the list of interpolated y values
    ynew = f(xnew)

    # determine the location of the interpolated maximum
    interp_max = np.max(ynew)
    interp_argmax = xnew[np.argmax(ynew)]
    deg = interp_argmax - start

    return deg

def corr_upsample(x, y, start, upsample_resolution):
    '''
    Determine the maximum correlation using the upsampling method.

    Parameters
    ------------------------
    x : array_like
        Indices location of the accepted correlation values relative to the
        original location in corrs.

    y : array_like
        Correlation values that are between lb and ub that are above the threshold.

    start : int
        Starting index location of the window.

    upsample_resolution : float
        Resultion of the upsampled iris segment.

    Returns
    ------------------------
    deg : float
        The amount of rotation using an upsampled image.
    '''

    # x, y = reduced_corr(corrs, lb, ub, threshold)

    upsample_max = np.max(y)
    upsample_argmax = x[np.argmax(y)]
    deg = (upsample_argmax - start) * upsample_resolution

    return deg

def reduced_corr(corrs, threshold, lb=0, ub=None, offset=0):
    '''
    Returns a list of correlation values between a lower bound and upper bound
    that only includes the values above the threshold. If there is no correlation
    found greater then the threshold within the bounds then CorrelationBelowThreshold
    exception is raised. If there is not enough points founds for interpolation
    then LackingInterpPoints is raised. See scipy.interpolate.interp1d for more
    details.

    Parameters
    ------------------------
    corrs : float list
        List of correlation values.

    threshold : float, 0 =< x < 1
        Minimum correlation value of correlation, all correlation values less
        then threshold will be ignored.

    lb : int, 0 <= x < ub, default = 0
        Lower bound.

    ub : int lb < x < len(corrs), default = len(corrs) - 1
        Upper bound.

    Returns
    ------------------------
    x : array_like
        Indices location of the accepted correlation values relative to the
        original location in corrs.

    y : array_like
        Correlation values that are between lb and ub that are above the threshold.
    '''

    # set upper bound
    if ub is None:
        ub = len(corrs) - 1

    # only consider correlation values within +- 25 deg
    y = np.array(corrs[lb:ub])
    I = np.where(y > threshold)[0]

    if len(I) == 0:
        #print(lb,ub,corrs)
        raise CorrelationBelowThreshold('All frame correlation below threshold value: {}'.format(threshold))

    # index locations of correlation values above threshold
    x = np.add(I, lb)
    # correlation values above threshold
    y = y[I]
    # account for an offset
    x = np.add(x, offset)

    return x, y

def corr2_coeff(a,b):
    '''
    Caclulates the 2D correlation coefficient of two matrices. Raise an exception
    if a and b are not the same size.

    The coefficient is calculated using the following formula:
    https://www.mathworks.com/help/images/ref/corr2.html.

    Parameters
    ------------------------
        a : array_like, NxM
        b : array_like, NxM

    Returns
    ------------------------
        coeff : float, 0 =< x =< 1
            2D correlation coefficient.
    '''

    if a.shape != b.shape:
        raise Exception('Must be same shape')

    # Subtract each element by mean
    a_m = a - a.mean()
    b_m = b - b.mean()

    # Sum of squares across rows
    a_ss = np.multiply(a_m, a_m).sum();
    b_ss = np.multiply(b_m, b_m).sum();

    # if each element of a or b is equal then subtracting from the means gives a zero matrix
    if not a_ss or not b_ss:
        return 0

    # calculate the 2D coefficient
    return np.multiply(a_m,b_m).sum() / np.sqrt(a_ss * b_ss)