import numpy as np

import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
from math import *
from cv2 import remap

# TODO instead of ret_cartesian use mode='polar' or cartesian
# https://github.com/scipy/scipy/blob/v0.19.1/scipy/signal/signaltools.py#L111-L269
# https://github.com/uber/pyro/blob/dev/pyro/distributions/distribution.py
def iris_transform(
    frame,
    pupil,
    iris_thickness,
    theta_window = (-90, 270),
    theta_resolution=1,
    r_resolution=1,
    mode='polar',
    reference_pupil=None,
    ):
    '''
    Transforms the iris in the given frame into polar representation where the vertical
    axis is 'r' and horizontal axis is theta.

    Optionally returns iris segment in cartesian coordinates

    Inputs:
        frame - opencv video frame (numpy array of intensities)
        pupil - a dictionary containing information about the pupil within the frame
        iris_thickness - pixel width of the iris
        theta_window - Range of theta values over which to sample the cartesian image
        theta_resolution - sampling interval for theta in degrees. Default is 1 degree.
        r_resolution - sampling interval for radius. Default is 1 pixel length
        ret_cartesian - boolean value which allows the return of only the iris in
                        cartesian coordinates. By default this is set to false
        reference_pupil - the reference pupil used for geometric correction

    Outputs:
        polar_iris - opencv image (numpy array) of extracted iris in polar coordinates
        cartesian_iris - opencv image (numpy array) of extracted iris in cartesian coordinates
    '''
    inner_radius_buffer = 5
    min_radius = int(pupil.radius) + inner_radius_buffer
    max_radius = min_radius + int(iris_thickness)
    pupil_row_loc = int(pupil.center_row)
    pupil_col_loc = int(pupil.center_col)

    if mode == 'cartesian':
        iris_size = int(2*max_radius)

        cartesian_iris = np.zeros((iris_size,iris_size))
        row_range = np.linspace(pupil_row_loc-max_radius,pupil_row_loc+max_radius,iris_size,dtype=int)
        col_range = np.linspace(pupil_col_loc-max_radius,pupil_col_loc+max_radius,iris_size,dtype=int)

        # extract pixels that are within a square bounding iris
        for i in range(iris_size):
            for j in range(iris_size):
                pixel_rad, pixel_theta = get_polar_coord(row_range[i], col_range[j], pupil)
                # if pixel is outside iris domain do not extract information
                if (pixel_rad > min_radius and pixel_rad < max_radius) and (pixel_theta >= theta_window[0] and pixel_theta <= theta_window[1]):
                    cartesian_iris[i,j] = frame[row_range[i],col_range[j]]
        return cartesian_iris

    elif mode == 'polar':
        # determine number of radial and theta increments
        n_radius = int((max_radius - min_radius)/r_resolution)
        n_theta = int((theta_window[1] - theta_window[0])/theta_resolution)

        coordinates = np.mgrid[min_radius:max_radius:n_radius * 1j, theta_window[0]:theta_window[1]:n_theta * 1j]
        radii = coordinates[0,:]
        angles = np.radians(coordinates[1,:])

        if reference_pupil == None or pupil.width >= reference_pupil.width or pupil.height >= reference_pupil.height:
            # Using scipy's map_coordinates(), we map the input array into polar
            # space centered about the detected pupil center location.
            polar_iris = ndimage.interpolation.map_coordinates(frame,
                                                    (-1*radii*sp.sin(angles) + pupil.center_row,
                                                    radii*sp.cos(angles) + pupil.center_col),
                                                    order=3, mode='constant')

            return polar_iris
        else:
            map_x = {}
            map_y = {}

            for r in range(min_radius,max_radius,r_resolution):
                for a in range(theta_window[0],theta_window[1],theta_resolution):
                    print(pupil.center_row, pupil.center_col)
                    print(reference_pupil.center_row, reference_pupil.center_col)
                    h_pupil_movement = pupil.center_row - reference_pupil.center_row
                    v_pupil_movement = pupil.center_col - reference_pupil.center_col
                    h_dist_from_center = r * sp.sin(a)
                    v_dist_from_center = r * sp.cos(a)
                    print(h_pupil_movement)
                    print(pupil.width/reference_pupil.width)
                    print((pupil.width/reference_pupil.width)**2)
                    print(sqrt(1 - (pupil.width/reference_pupil.width)**2))
                    r_eye = h_pupil_movement / sqrt(1 - (pupil.width/reference_pupil.width)**2)
                    print(r_eye, v_pupil_movement * reference_pupil.height / sqrt(reference_pupil.height**2 - pupil.height**2))
                    map_x[(a, r)] = reference_pupil.center_row + h_pupil_movement * sqrt(1 - (h_dist_from_center/r_eye)**2) + sqrt(1-(h_pupil_movement/r_eye)**2) * h_dist_from_center
                    map_y[(a, r)] = reference_pupil.center_col + v_pupil_movement * sqrt(1 - (v_dist_from_center/r_eye)**2) + sqrt(1-(v_pupil_movement/r_eye)**2) * v_dist_from_center

            geometric_corrected_iris = cv.remap(frame, map_x, map_y)

            plt.imshow(geometric_corrected_iris)
            plt.show()

            return geometric_corrected_iris

    else:
        # TODO throw exception
        print('Mode not supported')
        return None

def get_polar_coord(r, c, pupil):
    """
    Calculates the polar coordinates of the location specified by cartesian
        point (c,r). The origin of the polar coordinate frame is the center
        of the pupil.
    Inputs:
        c - Column index of the feature
        r - Row index of the feature
        pupil - A dictionary containing information regarding the pupil in the image
    Outputs:
        radius - The distance of the (c,r) location from the pupil center
        theta - The angular coordinate of the (c,r) location in polar space
    """
    delta_c = c - pupil.center_col
    delta_r = -1 * (r - pupil.center_row) # multiply by negative one to account for increasing y correpsonding to decreasing r
    radius = np.sqrt( delta_c**2 + delta_r**2 )

    if delta_c >= 0:
        theta = np.arcsin(delta_r / radius) * (180/np.pi)
    elif delta_r >= 0:
        theta = 180 - np.arcsin(delta_r / radius) * (180/np.pi)
    else:
        theta = 180 + np.arctan(delta_r / delta_c) * (180/np.pi)

    return radius, theta

def get_cartesian_coord(radius, theta, pupil):
    """
    Calculates the cartesian coordinates of the location specified by polar
        coordinate point (radius, theta). The origin of the polar coordinate frame is the center
        of the pupil.
    Inputs:
        radius - Distance of the location from the pupil center
        theta - The angular coordinate of the location in polar space
        pupil - A dictionary containing information regarding the pupil in the image
    Outputs:
        location - dictionatry containing the following:
            row - The row index of the location in cartesian image space
            col - The column index of the location in cartesian image space
    """
    col = pupil.center_col + radius * np.cos(theta * np.pi/180)
    row = pupil.center_row - radius * np.sin(theta * np.pi/180)

    # TODO return a tuple instead
    # ex, return c, r
    location = {'r': row, 'c': col}
    return location

def calculate_func_of_theta(polar_image):
    """
    Input: polar_image - A transformed (to polar coordinates) and masked image of the iris
    Output: f - A function that relates an angle theta to the sum of the intensity as the radius is varied for that fixed theta
    """
    n = len(polar_image[1, :])
    f = np.zeros(n)

    for i in range(n):
        f[i] = np.sum(polar_image[:, i])

    return f
