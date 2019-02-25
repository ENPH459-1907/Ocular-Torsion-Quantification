import numpy as np
import cv2, pdb, os
from matplotlib import pyplot as plt

import scipy as sp
from scipy import ndimage
from math import *
from ota.eyelid.eyelid import pupil_obstruct

class EmptyAreas(Exception):
    def __init__(self):
        Exception.__init__(self,'No distinct pupil area detected following thresholding, frame may be overexposed or eye might be closed.')

class Pupil:
    """
    Object to represent a pupil within a specific frame of the video.
    """

    def __init__(self, frame, threshold=10, skip_init=False):
        """
        Initialize pupil object and find it's center, and radius within frame

        Parameters
        ------------------------
        frame : array_like
            Grayscale video frame containing pupil to be detected
        threshold: Uint8
            Integer representing the value to use for image binary thresholding.

        Attributes
        ------------------------
        center : Dictionary
            A dictionary object containing the column and row indexes of the pupil center within the frame
            'c' : Center column index
            'r' : Center row index
        radius : float
            Value representing the radius of the pupil in frame (distance measured in pixels)
        pupil_cnt : array_like
            Vector type object containing a list of points contained in the contour of the pupil.
                0-index of point corresponds to column index
                1-index of point corresponds to row index
        blink : boolean
            True : frame records a blink
            False : frame does not record a blink
        """

        if skip_init is False:
            self.center_col, self.center_row, self.radius, self.contour = self.calc_pupil_properties_fit_ellipse(frame, threshold=threshold)
            #mat = np.ones((frame.shape[0]-100, frame.shape[1]))
            #eyelid_mat = np.zeros((100, frame.shape[1]))
            #mat_eye = np.vstack((eyelid_mat, mat))
            #self.blink = pupil_obstruct(mat_eye, self.contour)
            #print(self.blink)

        else:
            self.center_col = None
            self.center_row = None
            self.radius = None
            self.contour = None
            #self.blink = None

    def calc_pupil_properties_fit_ellipse(self, frame, threshold=10):
        """
        Find the location of the pupil center and the radius of the pupil within frame using a best fit ellipse.

        Parameters
        -----------------------
        frame : array_like
            Grayscale video frame containing pupil to be detected
        threshold: Uint8
            Integer representing the value to use for image binary thresholding.

        Returns
        -----------------------
        center : Dictionary
            A dictionary object containing the column and row indexes of the pupil center within the frame
            'c' : Center column index
            'r' : Center row index
        radius : float
            Value representing the radius of the pupil in frame (distance measured in pixels)
        pupil_cnt : array_like
            Vector type object containing a list of points contained in the contour of the pupil.
                0-index of point corresponds to column index
                1-index of point corresponds to row index
        """

        # Threshold the image
        ret, I = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)

        # Get a list of contours within the image
        img, contours, heighrarchy =  cv2.findContours(I, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Get the index corresponding to the contour with the maximum enclosed area
        areas = []
        if len(contours) == 1:
            areas.append(cv2.contourArea(contours[0]))
        else:
            for i in range(0, len(contours)):
                areas.append(cv2.contourArea(contours[i]))

        if not areas:
            raise EmptyAreas

        max_area_ind = areas.index(max(areas))

        # Get the list of contour points that enclose the max area
        pupil_cnt = contours[max_area_ind]

        # Fit an ellipse to the pupil contour in a least squares sense
        ellipse = cv2.fitEllipse(pupil_cnt)

        # Obtain the relevant pupil information from the best fit ellipse
        col = ellipse[0][0]
        row = ellipse[0][1]
        minor_axis_length = ellipse[1][0]
        major_axis_length = ellipse[1][1]

        # Obtain a rough estimate of the radius by averaging the major and minor axis lengths
        radius = (major_axis_length + minor_axis_length)/4

        return col, row, radius, pupil_cnt


    def calc_pupil_properties_min_enclosing_circle(self, frame, threshold=10):
        """
        Find the location of the pupil center and the radius of the pupil within frame using min enclosing circle.

        Parameters
        -----------------------
        frame : array_like
            Grayscale video frame containing pupil to be detected
        threshold: Uint8
            Integer representing the value to use for image binary thresholding.

        Returns
        -----------------------
        center : Dictionary
            A dictionary object containing the column and row indexes of the pupil center within the frame
            'c' : Center column index
            'r' : Center row index
        radius : float
            Value representing the radius of the pupil in frame (distance measured in pixels)
        pupil_cnt : array_like
            Vector type object containing a list of points contained in the contour of the pupil.
                0-index of point corresponds to column index
                1-index of point corresponds to row index
        """

        # Threshold the image
        ret, I = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)

        # Get a list of contours within the image
        img, contours, heighrarchy =  cv2.findContours(I, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Get the index corresponding to the contour with the maximum enclosed area
        areas = []
        if len(contours) == 1:
            areas.append(cv2.contourArea(contours[0]))
        else:
            for i in range(0, len(contours)):
                areas.append(cv2.contourArea(contours[i]))

        if not areas:
            raise EmptyAreas

        max_area_ind = areas.index(max(areas))

        # Get the list of contour points that enclose the max area
        pupil_cnt = contours[max_area_ind]

        # Fit a circle to the contour and find the center and radius
        (col, row), radius = cv2.minEnclosingCircle(pupil_cnt)

        return col, row, radius, pupil_cnt
