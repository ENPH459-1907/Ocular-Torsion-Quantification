import numpy as np

from ota.video import video as vid
from ota.torsion import xcorr2d
from ota.pupil import pupil
from ota.iris import iris, eyelid_removal
from ota import presets as pre
from tqdm import tqdm
from math import *
import cv2

def quantify_torsion(
    WINDOW_RADIUS,
    RESOLUTION,
    torsion_mode,
    transform_mode,
    video,
    start_frame,
    reference_frame,
    end_frame,
    pupil_list,
    eyelid_list,
    blink_list,
    threshold,
    WINDOW_THETA = None,
    SEGMENT_THETA = None,
    upper_iris = None,
    lower_iris = None,
    feature_coords = None,
    calibration_frame = None,
    calibration_angle = None,
    noise_replace = 0):

    '''
    Utilizes the 2D cross correlation algorithm xcorr2d to measure and return torsion using the settings given.

    Inputs:
        WINDOW_RADIUS:
            Integer
            Mandatory input which sets the radial thickness of the iris transform.

        RESOLUTION:
            Double
            Mandatory input with sets the upsampling factor or the interpolation resolution depending on settings below.

        torsion_mode:
            String
            Mandatory input which determines whether interpolation or upsampling is used.
            if torsion_mode = 'interp', then interpolation is used and RESOLUTION is assumed to be interpolation resolution. Consequently an upsampling factor of 1 is used in the transform.
            if torsion_mode = 'upsample', then upsampling is used and RESOLUTION is assumed to be the upsampling factor.

        transform_mode:
            String
            Mandatory input which determines whether a subset of the iris is used or the full iris is used during correlation.
            if transform_mode = 'subset', a subset of the iris is used.
            if transform_mode = 'full', the full iris us used.

        video:
            Video object
            Mandatory input

        start_frame:
            Integer
            Mandatory input which is the index of the first frame to analyze.

        reference_frame:
            Integer
            Mandatory input which is the index of the reference frame.

        end_frame:
            Integer
            Mandatory input which is the index of the last frame to analyze.

        pupil_list:
            dictionary of pupil objects
            key: (int) video frame
            value: pupil object

        eyelid_list:
            dictionary of eyelid masks
            key: (int) video frame
            value: eyelid masks arrays

        blink_list:
            dictionary of whether or not a frame captures a blink
            key: (int) video frame
            value: 1 - blink
                   0 - no blink
                   None - None

        threshold:
            Integer
            The pupil colour threshold

        WINDOW_THETA:
            Integer
            Angle bounds above/below the feature that define the portion of the iris that is to be included in the reference iris window. This window should be smaller than the segment.
            Mandatory input if transform_mode = 'subset'.

        SEGMENT_THETA:
            Integer
            Angle bounds above/below the feature that define the portion of the iris that is to be included in each segment, for which the window is to be located in.
            Mandatory input if transform_mode = 'subset'.

        upper_iris_occ:
            dictionary, {'c': column index, 'r': row index}
            Holds the [row,column] coordinates of the upper boundary of the iris that is not occluded by eyelids or eyelashes.

        lower_iris_occ:
            dictionary, {'c': column index, 'r': row index}
            Holds the [row,column] coordinates of the lower boundary of the iris that is not occluded by eyelids or eyelashes.

        feature_coords:
            dictionary, {'c': column index, 'r': row index}
            Holds the dictionary of feature coordinates tracked during subset correlation.
            Mandatory input if transform_mode = 'subset'.

        calibration_frame:
            Integer
            The frame used for calibration

        calibration_angle:
            Double
            The angle the eye is rotated in the calibration frame
    Returns:
        torsion:
            Dictionary
            key = frame number
            value = rotation from reference frame
        torsion_deriative:
            Dictionary
            key = frame number
            value = rotation from previous frame
        transformed_iris
            Dictionary
            key = frame number
            value = image of polar transformed iris
    '''

    upsample_factor = 1
    if noise_replace == 1:
        noise_replace = True
    if noise_replace == 0:
        noise_replace = False

    if torsion_mode == 'interp':
        pass
    elif torsion_mode == 'upsample':
        upsample_factor = RESOLUTION

    if transform_mode == 'subset':
        feature_r, feature_theta = iris.get_polar_coord(feature_coords['r'], feature_coords['c'], pupil_list[start_frame])
        reference_bounds = (feature_theta - WINDOW_THETA, feature_theta + WINDOW_THETA)
        comparison_bounds = (feature_theta - SEGMENT_THETA, feature_theta + SEGMENT_THETA)
        start = int((SEGMENT_THETA - WINDOW_THETA)/upsample_factor)
    elif transform_mode == 'full':
        if upper_iris and lower_iris:
            noise_replace = True
        start = 0
        reference_bounds = (0,360)
        comparison_bounds = (0,360)
    elif transform_mode == 'alternate':
        # Get the aspects as if you were doing a full iris analysis
        if upper_iris and lower_iris:
            noise_replace = True
        start = 0
        reference_bounds = (0, 360) # what are these? 360 degrees?
        comparison_bounds = (0, 360)

         # Get the aspects as if you are doing the subset
        feature_r, feature_theta = iris.get_polar_coord(feature_coords['r'], feature_coords['c'], pupil_list[start_frame])
        reference_bounds_sr = (feature_theta - WINDOW_THETA, feature_theta + WINDOW_THETA)
        comparison_bounds_sr = (feature_theta - SEGMENT_THETA, feature_theta + SEGMENT_THETA)
        start_sr = int((SEGMENT_THETA - WINDOW_THETA)/upsample_factor)

    # get the reference window from the first frame of the video
    # this will be the base for all torsion ie. all rotation is relative to this window
    ref_pupil = pupil.Pupil(video[reference_frame], threshold)
    if transform_mode == 'alternate':
        first_window_sr = iris.iris_transform(video[reference_frame],
            ref_pupil,
            WINDOW_RADIUS,
            theta_resolution=upsample_factor,
            theta_window=reference_bounds_sr)

    first_window = iris.iris_transform(video[reference_frame],
        ref_pupil,
        WINDOW_RADIUS,
        theta_resolution = upsample_factor,
        theta_window = reference_bounds)

    if calibration_frame != None:
        h_dist = ref_pupil.center_col - pupil_list[calibration_frame].center_col
        v_dist = ref_pupil.center_row - pupil_list[calibration_frame].center_row
        eyeball_radius = sqrt(h_dist**2 + v_dist**2)/sin(calibration_angle*pi/180)
    else:
        eyeball_radius = None

    if noise_replace:
        # replace occluded sections with noise
        first_window = iris.iris_transform(mask_img(eyelid_list[reference_frame], video[reference_frame]),
                                           ref_pupil,
                                           WINDOW_RADIUS,
                                           theta_resolution=upsample_factor,
                                           theta_window=reference_bounds)
        first_window = eyelid_removal.noise_replace_eyelid(first_window)
        # Find mean iris intensity
        normalized_magnitude = calculate_iris_mean(first_window)
        # replace occluded sections with noise
        first_window = iris.iris_transform(mask_img(eyelid_list[reference_frame], video[reference_frame], normalized_magnitude=normalized_magnitude),
                                           ref_pupil,
                                           WINDOW_RADIUS,
                                           theta_resolution=upsample_factor,
                                           theta_window=reference_bounds)
        first_window = eyelid_removal.noise_replace_eyelid(first_window)

    if transform_mode == 'full' or transform_mode == 'alternate' or noise_replace:
        # extend iris window
        first_window = eyelid_removal.iris_extension(
            first_window,
            theta_resolution = upsample_factor,
            lower_theta = -pre.MAX_ANGLE,
            upper_theta=pre.MAX_ANGLE)

    torsion = {}
    torsion_derivative = {}
    transformed_iris = {}
    # find torsion between start_frame+1:last_frame
    for i, frame in tqdm(enumerate(video[start_frame:end_frame])):
        frame_loc = i + start_frame
        if frame_loc == start_frame:
            deg = 0
            previous_deg = None
            current_frame = first_window # This is true right?
            '''
            current_frame = iris.iris_transform(frame,
                pupil_list[frame_loc],
                WINDOW_RADIUS,
                theta_resolution=upsample_factor,
                theta_window=comparison_bounds,
                reference_pupil=ref_pupil,
                eye_radius=eyeball_radius)
            '''
        # check if a pupil exists , or if there is a blink
        elif not pupil_list[frame_loc] or blink_list[frame_loc] is None:
            # if there is no pupil, torsion cannot be calculated
            deg = None
            previous_deg = None
            current_frame = None
            print('WARNING: No pupil in frame: %d \n Torsion cannot be calculated' % (frame_loc))
        else:
            if transform_mode == 'alternate' and blink_list[frame_loc] == 1 or blink_list[frame_loc] == None:
                current_frame = iris.iris_transform(frame,
                    pupil_list[frame_loc],
                    WINDOW_RADIUS,
                    theta_resolution=upsample_factor,
                    theta_window=comparison_bounds_sr,
                    reference_pupil=ref_pupil,
                    eye_radius=eyeball_radius)
            elif noise_replace:
                current_frame = iris.iris_transform(mask_img(eyelid_list[frame_loc], frame, normalized_magnitude=normalized_magnitude),
                    pupil_list[frame_loc],
                    WINDOW_RADIUS,
                    theta_resolution=upsample_factor,
                    theta_window=comparison_bounds,
                    reference_pupil=ref_pupil,
                    eye_radius=eyeball_radius)
                try:
                    current_frame = eyelid_removal.noise_replace_eyelid(current_frame)
                except:
                    current_frame = None
            else:
                current_frame = iris.iris_transform(frame,
                    pupil_list[frame_loc],
                    WINDOW_RADIUS,
                    theta_resolution=upsample_factor,
                    theta_window=comparison_bounds,
                    reference_pupil=ref_pupil,
                    eye_radius=eyeball_radius)

            try:
                if current_frame is None:
                    deg = None
                    previous_deg = None
                else:

                    '''
                    if noise_replace:
                        print('hi')
                        # replace occluded sections with noise
                        current_frame = eyelid_removal.noise_replace_eyelid(current_frame)
                    # get the degree of rotation of the current frame based on reference fram
                    '''
                    deg = xcorr2d.xcorr2d(current_frame,
                        first_window,
                        start=start,
                        prev_deg=None,
                        torsion_mode=torsion_mode,
                        resolution=RESOLUTION,
                        threshold=0,
                        max_angle=pre.MAX_ANGLE)

                    previous_window = transformed_iris[frame_loc - 1]
                    if previous_window is None:
                        previous_deg = None
                        continue
                    if not (transform_mode == 'alternate' and blink_list[frame_loc] == 1 or blink_list[frame_loc] == None):
                        previous_window = eyelid_removal.iris_extension(previous_window,
                            theta_resolution=upsample_factor,
                            lower_theta=-pre.MAX_ANGLE,
                            upper_theta=pre.MAX_ANGLE)
                    # get the degree of rotation of the current frame based on previous frame
                    previous_deg = xcorr2d.xcorr2d(current_frame,
                        previous_window,
                        start=start,
                        prev_deg=None,
                        torsion_mode=torsion_mode,
                        resolution=RESOLUTION,
                        threshold=0,
                        max_angle=pre.MAX_ANGLE)
            except:
                deg = None
        torsion[frame_loc] = deg
        torsion_derivative[frame_loc] = previous_deg
        transformed_iris[frame_loc] = current_frame
    return torsion, torsion_derivative, transformed_iris

def mask_img(mask, frame, normalized_magnitude=None):
    '''
    Bitwise masking of a frame with a given mask
    '''
    maskedImg = cv2.bitwise_and(frame, mask)
    if normalized_magnitude:
        maskedImg[maskedImg == 0] = normalized_magnitude
    return maskedImg

def calculate_iris_mean(transformed_masked_img):
    '''
    Calculates the mean intesity of the iris for all frames
    '''
    iris = np.array(transformed_masked_img)
    nonzero = np.nonzero(iris)
    normalized_magnitude = (np.sum(iris[nonzero])/iris[nonzero].size)
    return normalized_magnitude