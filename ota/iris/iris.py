import numpy as np

import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
from math import *
from cv2 import remap, INTER_LINEAR

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
    eye_radius=None,
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
        eye_radius - the radius of the eyeball in pixels

    Outputs:
        polar_iris - opencv image (numpy array) of extracted iris in polar coordinates
        cartesian_iris - opencv image (numpy array) of extracted iris in cartesian coordinates
    '''
    # If no pupil can be found, then just skip everything
    if pupil is None:
        return None

    inner_radius_buffer = 5
    min_radius = int(pupil.major/2) + inner_radius_buffer
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

        major_minor_ratio = pupil.minor/pupil.major

        if reference_pupil == None or major_minor_ratio >= 0.9:
            # Using scipy's map_coordinates(), we map the input array into polar
            # space centered about the detected pupil center location.
            polar_iris = ndimage.interpolation.map_coordinates(frame,
                                                    (-1*radii*sp.sin(angles) + pupil.center_row,
                                                    radii*sp.cos(angles) + pupil.center_col),
                                                    order=3, mode='constant')

            return polar_iris
        else:
            h_pupil_movement = pupil.center_col - reference_pupil.center_col
            v_pupil_movement = pupil.center_row - reference_pupil.center_row

            if eye_radius == None:
                lateral_angle = get_lateral_angle(major_minor_ratio)
                r_eye = sqrt(h_pupil_movement**2+v_pupil_movement**2)/sp.sin(lateral_angle)
            else:
                r_eye = eye_radius

            phi0 = asin(h_pupil_movement/r_eye)
            theta0 = asin(v_pupil_movement/r_eye)
            map_x = np.zeros((n_radius, n_theta), dtype=np.float32)
            map_y = np.zeros((n_radius, n_theta), dtype=np.float32)

            # TODO: r * cos(a * pi / 180) / r_eye sometimes give a value outside [-1,1], asin is invalid
            try:
                for r in range(min_radius,max_radius,r_resolution):
                    for a in range(theta_window[0],theta_window[1],theta_resolution):
                        phi = phi0 + asin(r * cos(a * pi / 180) / r_eye)
                        theta = theta0 - asin(r * sin(a * pi / 180) / r_eye)
                        x_loc = reference_pupil.center_col + r_eye * sin(phi)
                        y_loc = reference_pupil.center_row + r_eye * sin(theta)
                        #frame[min(y_loc, frame.shape[0]-1)][min(x_loc, frame.shape[1]-1)] = 0
                        map_x[((r - min_radius)/r_resolution, (a - theta_window[0])/theta_resolution)] = x_loc
                        map_y[((r - min_radius)/r_resolution, (a - theta_window[0])/theta_resolution)] = y_loc
                geometric_corrected_iris = remap(frame, map_x, map_y, INTER_LINEAR)
                return geometric_corrected_iris
            except:
                print('Something bad happened in angle calculations')
                return None
    else:
        # TODO throw exception
        print('Mode not supported')
        return None

def get_lateral_angle(ratio, ref_ratio=0.98):
    """
    Calculates the angle of lateral motion from the ratio of the major and
    minor axes of the pupil using the formula from Atchison-Smith.
    Inputs:
        ratio - the minor:major ratio
    Outputs:
        angle - the lateral rotation angle
    """
    a = 1.8698*10**-9
    b = -1.0947*10**-4
    c = 1 - ratio
    return sqrt((-b - sqrt(b**2-4*a*c))/(2*a))*pi/180

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
