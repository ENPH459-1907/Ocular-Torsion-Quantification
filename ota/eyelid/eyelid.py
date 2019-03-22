import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('.'), os.path.pardir)))

import cv2
import cProfile
import numpy as np

# import settings
# from ota.helpers import plotting as plot

from ota.video import video as vid
from ota import presets
from ota.pupil import pupil
from ota.iris import iris


from functools import reduce

def detect_eyelid(image, pupil, **kw):
    """
    Detect the upper and lower eyelids within a video image

    Parameters
    ------------------------
    image : array_like
        Grayscale video image containing eyelid to be detected
    pupil: pupil object
        Object representing the pupil within the given image

    Returns
    ------------------------
    eyelids_removed : array_like
        A video image with the parts above the upper eyelid and below the lower eyelid blocked out.
    """

    # Define parameters to be used in eyelid detection
    ROI_STRIP_WIDTH = kw.get('ROI_STRIP_WIDTH', 200) # Width of Region of Interest
    ROI_BUFFER = kw.get('ROI_BUFFER', 20) # Buffer between pupil edge and start of Region of Interest
    LOWER_CANNY = kw.get('LOWER_CANNY', 50) # Lower threshold for canny edge detection
    UPPER_CANNY = kw.get('UPPER_CANNY', 60) # Upper threshold for canny edge detection
    min_theta = kw.get('min_theta', 70 * np.pi/180) # Minimum angle for hough line search
    max_theta = kw.get('max_theta', 110 * np.pi/180) # Maximum angle for hough line search
    POLY_DEG = kw.get('POLY_DEG', 2) # Degree of polynomial to fit to eyelid points
    UPPER_LID_POLY_TRANS = kw.get('UPPER_LID_POLY_TRANS', 40) # Amount by which to translate upper lid down to more conservatively ensure coverage of entire lid
    LOWER_LID_POLY_TRANS = kw.get('LOWER_LID_POLY_TRANS', 0) # Amount by which to translate lower lid up to more conservatively ensure coverage of entire lid

    # Defined the image indices representing four regions of interest about the eye
    l_cols = (int(pupil.center_col - (pupil.radius + ROI_STRIP_WIDTH)), int(pupil.center_col - (pupil.radius + ROI_BUFFER)) )
    r_cols = (int(pupil.center_col + (pupil.radius + ROI_BUFFER)), int(pupil.center_col + (pupil.radius + ROI_STRIP_WIDTH)))

    u_rows = (0, int(pupil.center_row - ROI_BUFFER))
    l_rows = (int(pupil.center_row + ROI_BUFFER), image.shape[0])

    # Create the 4 regions of interest: Upper left, Lower left, Upper right, Lower Right
    ul_img = image[u_rows[0]:u_rows[1], l_cols[0]:l_cols[1]]
    ll_img = image[l_rows[0]:l_rows[1], l_cols[0]:l_cols[1]]
    ur_img = image[u_rows[0]:u_rows[1], r_cols[0]:r_cols[1]]
    lr_img = image[l_rows[0]:l_rows[1], r_cols[0]:r_cols[1]]

    # Apply canny edge detection to the ROIs
    ul_canny = cv2.Canny(ul_img, LOWER_CANNY, UPPER_CANNY)
    ll_canny = cv2.Canny(ll_img, LOWER_CANNY, UPPER_CANNY)
    ur_canny = cv2.Canny(ur_img, LOWER_CANNY, UPPER_CANNY)
    lr_canny = cv2.Canny(lr_img, LOWER_CANNY, UPPER_CANNY)

    # Apply a hough transform to each ROI
    ul_lines = cv2.HoughLines(ul_canny, 1, np.pi/180, 1, min_theta=min_theta, max_theta=max_theta)
    ll_lines = cv2.HoughLines(ll_canny, 1, np.pi/180, 1, min_theta=min_theta, max_theta=max_theta)
    ur_lines = cv2.HoughLines(ur_canny, 1, np.pi/180, 1, min_theta=min_theta, max_theta=max_theta)
    lr_lines = cv2.HoughLines(lr_canny, 1, np.pi/180, 1, min_theta=min_theta, max_theta=max_theta)

    # Construct the detected line for each ROI individually
    lspace = np.arange(-1000,1000, 10)

    rho = ul_lines[0][0][0]
    theta = ul_lines[0][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho * a
    y0 = rho * b
    ul_x = np.array(x0 + lspace*(-b), dtype='int')
    ul_y = np.array(y0 + lspace*(a), dtype='int')

    rho = ur_lines[0][0][0]
    theta = ur_lines[0][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho * a
    y0 = rho * b
    ur_x = np.array(x0 + lspace*(-b), dtype='int')
    ur_y = np.array(y0 + lspace*(a), dtype='int')

    rho = lr_lines[0][0][0]
    theta = lr_lines[0][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho * a
    y0 = rho * b
    lr_x = np.array(x0 + lspace*(-b), dtype='int')
    lr_y = np.array(y0 + lspace*(a), dtype='int')

    rho = ll_lines[0][0][0]
    theta = ll_lines[0][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho * a
    y0 = rho * b
    ll_x = np.array(x0 + lspace*(-b), dtype='int')
    ll_y = np.array(y0 + lspace*(a), dtype='int')


    # Only keep points of the detected lines that lie within the ROIs
    inds = reduce(np.intersect1d, ( np.where(ul_x >= 0), np.where(ul_y >= 0),
                                    np.where(ul_x <= ul_img.shape[1]-1), np.where(ul_y <= ul_img.shape[0]-1)))
    ul_x = ul_x[inds]
    ul_y = ul_y[inds]

    inds = reduce(np.intersect1d, ( np.where(ur_x >= 0), np.where(ur_y >= 0),
                                    np.where(ur_x <= ur_img.shape[1]-1), np.where(ur_y <= ur_img.shape[0]-1)))
    ur_x = ur_x[inds]
    ur_y = ur_y[inds]

    inds = reduce(np.intersect1d, ( np.where(lr_x >= 0), np.where(lr_y >= 0),
                                    np.where(lr_x <= lr_img.shape[1]-1), np.where(lr_y <= lr_img.shape[0]-1)))
    lr_x = lr_x[inds]
    lr_y = lr_y[inds]

    inds = reduce(np.intersect1d, ( np.where(ll_x >= 0), np.where(ll_y >= 0),
                                    np.where(ll_x <= ll_img.shape[1]-1), np.where(ll_y <= ll_img.shape[0]-1)))
    ll_x = ll_x[inds]
    ll_y = ll_y[inds]

    # Put the coordinates of points on the lines back into the row,column space of the global image image
    # Upper eyelid first
    ul_x = ul_x + l_cols[0]
    ul_y = ul_y + u_rows[0]
    ur_x = ur_x + r_cols[0]
    ur_y = ur_y + u_rows[0]
    ulid_x = np.append(ul_x, ur_x)
    ulid_y = np.append(ul_y, ur_y)

    # Lower eyelid
    ll_x = ll_x + l_cols[0]
    ll_y = ll_y + l_rows[0]
    lr_x = lr_x + r_cols[0]
    lr_y = lr_y + l_rows[0]
    llid_x = np.append(ll_x, lr_x)
    llid_y = np.append(ll_y, lr_y)

    # Fit a polynomial to the upper and lower lids individually
    X = np.arange(0, image.shape[1], 1)

    z = np.polyfit(ulid_x, ulid_y, POLY_DEG)
    z[POLY_DEG] = z[POLY_DEG] + UPPER_LID_POLY_TRANS # Translate the estimated eyelid down.
    ulid_poly = np.poly1d(z)
    ulid = np.array(ulid_poly(X), dtype='int')

    z = np.polyfit(llid_x, llid_y, POLY_DEG)
    z[POLY_DEG] = z[POLY_DEG] - LOWER_LID_POLY_TRANS# Translate the estimated eyelid up.
    llid_poly = np.poly1d(z)
    llid = np.array(llid_poly(X), dtype='int')

    eyelids_removed = image.copy()

    # for each index value in ulid and llid
    for i in range(len(ulid)):
        # set the parts of the image above and below the eyelid to 0
        if ulid[i] >= 0:
            eyelids_removed[0:ulid[i],i] = 0
        if llid[i] <= image.shape[1]:
            eyelids_removed[llid[i]:image.shape[1],i] = 0

    return eyelids_removed

def pupil_obstruct(eyelid_mat, contour):
    """
    Determine if the pupil is obstructed (ie. blinks)

    Parameters
    ------------------------
    eyelid_mat : array_like
        An image with the eyelids removed
        
    contour: array_like
        The contour of the pupil of the eye.

    Returns
    ------------------------
    pupil_obstructed : int
        1 if the pupil is obstructed. 0 if not.
    """
    # If things are None, abort mission
    if eyelid_mat is None or contour is None:
        return None

     # Find locations where eyelid exists
    indices_zero = np.nonzero(eyelid_mat == 0)

     # If there is no eyelid, just abort
    if indices_zero[0].size == 0 or indices_zero[1].size == 0:
        return True
    eyelid_locs = [(indices_zero[0][i], indices_zero[1][i]) for i in range(0, len(indices_zero))]

     # Get the contour into proper form: set of tuples (row, col)
    contour = np.squeeze(contour)
    contour = [tuple(x) for x in contour]

     # Find the intersection between pupil contour and eyelid locations (intersection of 4 pixels)
    intersection = np.array([x for x in contour if x in eyelid_locs])
    if len(intersection) == 0:
        return 0
    return 1
