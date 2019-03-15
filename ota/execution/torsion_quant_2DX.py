import numpy as np

from ota.video import video as vid
from ota import presets
from ota.torsion import xcorr2d
from ota.pupil import pupil
from ota.iris import iris, eyelid_removal
from ota import presets as pre
from tqdm import tqdm
from math import *

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
    blink_list,
    threshold,
    alternate = None,
    WINDOW_THETA = None,
    SEGMENT_THETA = None,
    upper_iris = None,
    lower_iris = None,
    feature_coords = None,
    calibration_frame = None,
    calibration_angle = None,):

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
            Mandatory input which is the index of the reference frame.

        end_frame:
            Integer
            Mandatory input which is the index of the last frame to analyze.

        pupil_list:
            dictionary of pupil objects
            key: (int) video frame
            value: pupil object

        blink_list:
            dictionary of whether or not a frame captures a blink
            key: (int) video frame
            value: 1 - blink
                   0 - no blink
                   None - None

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

        torsion_deriative:
            Dictionary
            key = frame number
            value = rotation from previous frame

    Returns:

        torsion:
            Dictionary
            key = frame number
            value = rotation from reference frame
    '''

    upsample_factor = 1
    noise_replace = False

    if torsion_mode == 'interp':
        pass
    elif torsion_mode == 'upsample':
        upsample_factor = RESOLUTION

    if transform_mode == 'subset':
        print(transform_mode)
        feature_r, feature_theta = iris.get_polar_coord(feature_coords['r'], feature_coords['c'], pupil_list[start_frame])
        reference_bounds = (feature_theta - WINDOW_THETA, feature_theta + WINDOW_THETA)
        comparison_bounds = (feature_theta - SEGMENT_THETA, feature_theta + SEGMENT_THETA)
        start = int((SEGMENT_THETA - WINDOW_THETA)/upsample_factor)
    elif transform_mode == 'full':
        print(transform_mode)
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

    if calibration_frame != None:
        h_dist = ref_pupil.center_col - pupil_list[calibration_frame].center_col
        v_dist = ref_pupil.center_row - pupil_list[calibration_frame].center_row
        eyeball_radius = sqrt(h_dist**2 + v_dist**2)/sin(calibration_angle*pi/180)
    else:
        eyeball_radius = None

    # get the reference window from the first frame of the video
    # this will be the base for all torsion ie. all rotation is relative to this window
    if start_frame == reference_frame:
        if alternate:
            first_window_sr = iris.iris_transform(video[start_frame],
                                                  pupil_list[start_frame],
                                                  WINDOW_RADIUS,
                                                  theta_resolution=upsample_factor,
                                                  theta_window=reference_bounds_sr)
        first_window = iris.iris_transform(video[start_frame],
                                           pupil_list[start_frame],
                                           WINDOW_RADIUS,
                                           theta_resolution = upsample_factor,
                                           theta_window = reference_bounds)


        ref_pupil = pupil_list[start_frame]
    else:
        ref_pupil = pupil.Pupil(video[reference_frame], threshold)
        if alternate:
            first_window_sr = iris.iris_transform(video[reference_frame],
                                                  ref_pupil,
                                                  WINDOW_RADIUS,
                                                  theta_resolution=upsample_factor,
                                                  theta_window=reference_bounds_sr)

        first_window = iris.iris_transform(video[reference_frame], ref_pupil, WINDOW_RADIUS, theta_resolution = upsample_factor, theta_window = reference_bounds)


    if transform_mode == 'full' or transform_mode == 'alternate':
        # extend iris window
        first_window = eyelid_removal.iris_extension(first_window, theta_resolution = upsample_factor, lower_theta = -pre.MAX_ANGLE, upper_theta=pre.MAX_ANGLE)
    # TODO: If noise replace is selected, cannot select segment removal
    if noise_replace:
        # transform (colum,row) into (theta,r) space about pupil centre
        # get the boundaries of usable iris in polar
        upper_iris_r, upper_iris_theta = iris.get_polar_coord(upper_iris['r'], upper_iris['c'], pupil_list[start_frame])
        lower_iris_r, lower_iris_theta = iris.get_polar_coord(lower_iris['r'], lower_iris['c'], pupil_list[start_frame])

        # mirrors the upper angular boundary across the vertical axis
        upper_occlusion_theta = (90 - np.absolute(upper_iris_theta - 90), 90 + np.absolute(upper_iris_theta - 90))

        # mirrors the lower angular boundary across the vertical axis
        # deal with the branch cut at 270
        if lower_iris_theta < 0:
            lower_occlusion_theta = (-90 - np.absolute(lower_iris_theta + 90), -90 + np.absolute(lower_iris_theta + 90))
        else:
            lower_occlusion_theta = (-90 - np.absolute(lower_iris_theta - 270), -90 + np.absolute(lower_iris_theta - 270))

        # replace occluded sections with noise
        first_window = eyelid_removal.noise_replace(first_window, upper_occlusion_theta, lower_occlusion_theta)
    # TODO: Add a button to show iris segments

    torsion = {}
    torsion_derivative = {}
    transformed_iris = {}
    # find torsion between start_frame+1:last_frame
    for i, frame in tqdm(enumerate(video[start_frame:end_frame])):
        frame_loc = i + start_frame
        if frame_loc == start_frame:
            deg = 0
            previous_deg = None
            current_frame = iris.iris_transform(frame, pupil_list[frame_loc], WINDOW_RADIUS,
                                                theta_resolution=upsample_factor, theta_window=comparison_bounds,
                                                reference_pupil=ref_pupil, eye_radius=eyeball_radius)
        # check if a pupil exists , or if there is a blink
        elif not pupil_list[frame_loc] or blink_list[frame_loc] is None:
            # if there is no pupil, torsion cannot be calculated
            deg = None
            previous_deg = None
            current_frame = iris.iris_transform(frame, pupil_list[frame_loc], WINDOW_RADIUS,
                                                theta_resolution=upsample_factor, theta_window=comparison_bounds,
                                                reference_pupil=ref_pupil, eye_radius=eyeball_radius)
            print('WARNING: No pupil in frame: %d \n Torsion cannot be calculated' % (frame_loc))
        else:
            if noise_replace:
                current_frame = iris.iris_transform(frame, pupil_list[frame_loc], WINDOW_RADIUS,
                                                    theta_resolution=upsample_factor, theta_window=comparison_bounds,
                                                    reference_pupil=ref_pupil, eye_radius=eyeball_radius)
                try:
                    if current_frame is None:
                        deg = None
                        previous_deg = None
                    else:
                        # replace occluded sections with noise
                        current_frame = eyelid_removal.noise_replace(current_frame, upper_occlusion_theta, lower_occlusion_theta)
                        # get the degree of rotation of the current frame based on reference frame
                        deg = xcorr2d.xcorr2d(current_frame, first_window, start=start, prev_deg=None,
                                              torsion_mode=torsion_mode, resolution=RESOLUTION, threshold=0,
                                              max_angle=pre.MAX_ANGLE)
                        if i > 0:
                            # get the previous frame
                            previous_window = iris.iris_transform(video[frame_loc - 1],
                                                                  pupil_list[frame_loc - 1],
                                                                  WINDOW_RADIUS,
                                                                  theta_resolution=upsample_factor,
                                                                  theta_window=reference_bounds,
                                                                  reference_pupil=ref_pupil,
                                                                  eye_radius=eyeball_radius)
                            if previous_window is None:
                                previous_deg = None
                                continue
                            previous_window = eyelid_removal.iris_extension(previous_window,
                                                                            theta_resolution=upsample_factor,
                                                                            lower_theta=-pre.MAX_ANGLE,
                                                                            upper_theta=pre.MAX_ANGLE)
                            # get the degree of rotation of the current frame based on previous frame
                            previous_deg = xcorr2d.xcorr2d(current_frame, previous_window, start=start, prev_deg=None,
                                                           torsion_mode=torsion_mode, resolution=RESOLUTION,
                                                           threshold=0,
                                                           max_angle=pre.MAX_ANGLE)
                        else:
                            previous_deg = None
                except:
                    deg = None
                    previous_deg = None

            # unwrap the iris (convert into polar)
            elif alternate and blink_list[frame_loc] == 1:
                current_frame = iris.iris_transform(frame, pupil_list[frame_loc],
                                                    WINDOW_RADIUS,
                                                    theta_resolution=upsample_factor,
                                                    theta_window=comparison_bounds_sr, reference_pupil=ref_pupil, eye_radius=eyeball_radius)
                # get the degree of rotation of the current frame based on reference frame
                deg = xcorr2d.xcorr2d(current_frame,
                                      first_window_sr,
                                      start=start_sr,
                                      prev_deg=None,
                                      torsion_mode=torsion_mode,
                                      resolution=RESOLUTION,
                                      threshold=0,
                                      max_angle=pre.MAX_ANGLE)
                if i > 0:
                    # Calculate torsion based off of previous window
                    previous_window_sr = iris.iris_transform(video[frame_loc-1],
                                                          pupil_list[frame_loc-1],
                                                          WINDOW_RADIUS,
                                                          theta_resolution=upsample_factor,
                                                          theta_window=reference_bounds_sr, reference_pupil=ref_pupil, eye_radius=eyeball_radius)

                    # get the degree of rotation of the current frame based on previous frame
                    previous_deg = xcorr2d.xcorr2d(current_frame,
                                                    previous_window_sr,
                                                  start=start_sr,
                                                  prev_deg=None,
                                                  torsion_mode=torsion_mode,
                                                  resolution=RESOLUTION,
                                                  threshold=0,
                                                  max_angle=pre.MAX_ANGLE)
                else:
                    previous_deg = None


            else:
                current_frame = iris.iris_transform(frame, pupil_list[frame_loc], WINDOW_RADIUS,
                                                    theta_resolution=upsample_factor, theta_window=comparison_bounds, reference_pupil=ref_pupil, eye_radius=eyeball_radius)
                try:
                    if current_frame is None:
                        deg = None
                        previous_deg = None
                    else:
                        # get the degree of rotation of the current frame based on reference frame
                        deg = xcorr2d.xcorr2d(current_frame, first_window, start=start, prev_deg=None,
                                          torsion_mode=torsion_mode, resolution=RESOLUTION, threshold=0,
                                          max_angle=pre.MAX_ANGLE)
                        if i > 0:
                            '''
                            # get the previous frame
                            previous_window = iris.iris_transform(video[frame_loc-1],
                                                               pupil_list[frame_loc-1],
                                                               WINDOW_RADIUS,
                                                               theta_resolution=upsample_factor,
                                                               theta_window=reference_bounds, reference_pupil=ref_pupil, eye_radius=eyeball_radius)
                            '''
                            previous_window = transformed_iris[frame_loc - 1]
                            if previous_window is None:
                                previous_deg = None
                                continue
                            previous_window = eyelid_removal.iris_extension(previous_window,
                                                                            theta_resolution=upsample_factor,
                                                                            lower_theta=-pre.MAX_ANGLE,
                                                                            upper_theta=pre.MAX_ANGLE)
                            # get the degree of rotation of the current frame based on previous frame
                            previous_deg = xcorr2d.xcorr2d(current_frame, previous_window, start=start, prev_deg=None,
                                                           torsion_mode=torsion_mode, resolution=RESOLUTION,
                                                           threshold=0,
                                                           max_angle=pre.MAX_ANGLE)
                        else:
                            previous_deg = None
                except:
                    deg = None
                    previous_deg = None

        torsion[frame_loc] = deg
        torsion_derivative[frame_loc] = previous_deg
        transformed_iris[frame_loc] = current_frame
    return torsion, torsion_derivative, transformed_iris