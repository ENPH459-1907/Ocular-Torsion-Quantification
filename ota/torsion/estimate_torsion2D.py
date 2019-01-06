# import cProfile
# import os
# import sys
# from math import pi
#
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
#
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname('.'), os.path.pardir)))
#
# import ota.settings as settings
# from ota.gui import coord_click as clk
# from ota.helpers import plotting as plot
# from ota import presets
# from ota.torsion import xcorr2d
# from ota.video import video as vid
# from ota.pupil import pupil
# from ota.iris import iris
# from ota.data import data as dat
# from ota.gui import frame_scroll as scroll
#
# from time import time
#
# def estimate_torsion2D(video, first_frame=0, last_frame=len(video)):
#     # PRESETS
#     WINDOW_THETA = 50 # plus/minus angle range from feature angle which will define window
#     WINDOW_RADIUS = 20 # how thick the window is in the radial direction
#     THETA_RESOLUTION = 0.1 # sampling resolution in transform
#     FRAME_THETA = 100 # plus/minus angle range from feature angle which defines search window
#
#     # create a frame index list
#     first_frame = 0
#     last_frame = len(video) - 1
#
#     # np.arange goes from [first_frame,last_frame)
#     frame_index_list = np.arange(first_frame, last_frame+1)
#
#     # dictionaries to store results
#     pupil_list = {} #
#     offset_first_frame = {} # difference in angle between frame 1 and frame n
#     feature_location_list = {} # (column,row) coordinates of maximum correlation
#
#     # try to find pupil at index first_frame
#     try:
#         pupil_list[first_frame] = pupil.Pupil(video[first_frame])
#     except:
#         print('Error analyzing frame: %d, Select alterative starting frame' % first_frame)
#
#     # get user input on the feature to track in (column,row) format
#     feature = clk.click_coordinates(video[first_frame],'Click on a distinct iris feature to track')
#     print('Selected Column: %d, Selected Row: %d' % (feature['c'], feature['r']))
#
#     # # get user input on the approximate iris thickness
#     # outer_iris = clk.click_coordinates(video[first_frame],'Click on the iris periphery')
#     # print('Selected Column: %d, Selected Row: %d' % (outer_iris['c'], outer_iris['r']))
#
#     # transform (colum,row) format of feature postion into (theta,r) space about pupil centre
#     feature_r, feature_theta = iris.get_polar_coord(feature['r'], feature['c'], pupil_list[first_frame])
#     theta_window = (feature_theta - WINDOW_THETA, feature_theta + WINDOW_THETA)
#     theta_frame = (feature_theta - FRAME_THETA, feature_theta + FRAME_THETA)
#
#     # # find thickness of iris
#     # outer_iris_r, outer_iris_theta = iris.get_polar_coord(outer_iris['r'], outer_iris['c'], pupil_list[first_frame])
#     # IRIS_THICKNESS = np.absolute(outer_iris_r - pupil_list[first_frame].radius)
#
#
#     t_start = time()
#
#     # locate pupils in frames between first_frame+1:last_frame
#     for idx,frame in enumerate(video[first_frame:last_frame]):
#         try:
#             pupil_at_idx = pupil.Pupil(frame)
#         except pupil.EmptyAreas:
#             print('Pupil not found in frame: %d \n None type object used inplace' % frame_index_list[idx])
#             pupil_list[frame_index_list[idx]] = None
#         else:
#             pupil_list[frame_index_list[idx]] = pupil_at_idx
#
#     pupil_time = time() - t_start
#     scroll.pupil_scroll(video,pupil_list)
#
#     # transform frame at index first_frame
#     # beginning_frame = iris.iris_transform(video[first_frame], pupil_list[first_frame], WINDOW_RADIUS, theta_resolution = THETA_RESOLUTION, theta_window=theta_frame)
#     first_window = iris.iris_transform(video[first_frame], pupil_list[first_frame], WINDOW_RADIUS, theta_resolution = THETA_RESOLUTION, theta_window=theta_window)
#     window_length = int(np.absolute(2*WINDOW_THETA/THETA_RESOLUTION))
#     window_shifts = int(np.absolute(2*FRAME_THETA/THETA_RESOLUTION - window_length))
#
#     # find torsion between first_frame+1:last_frame
#     for idx, frame in enumerate(video[first_frame:last_frame]):
#         # check if a pupil exists
#         if not (pupil_list[frame_index_list[idx]] or pupil_list[frame_index_list[idx-1]]):
#             # if there is no pupil, torsion cannot be calculated
#             offset_first_frame[frame_index_list[idx]] = None
#             print('No pupil in frame: %d \n Torsion cannot be calculated' % (frame_index_list[idx]))
#             # offset between first frame and first frame is 0
#         elif idx == 0:
#             offset_first_frame[frame_index_list[idx]] = 0
#             previous_window = first_window
#         else:
#             current_frame = iris.iris_transform(video[frame_index_list[idx]], pupil_list[frame_index_list[idx]], WINDOW_RADIUS, theta_resolution = THETA_RESOLUTION, theta_window=theta_frame)
#             shift_prev, max_prev_corr, shift_first, max_first_corr = xcorr2d.xcorr2d(current_frame,previous_window,frame_index_list[idx-1],first_window,first_frame, WINDOW_LENGTH = window_length, WINDOW_SHIFTS = window_shifts)
#             offset_first_frame[frame_index_list[idx]] = shift_first * THETA_RESOLUTION - (FRAME_THETA-WINDOW_THETA)
#             previous_window = iris.iris_transform(video[frame_index_list[idx]], pupil_list[frame_index_list[idx]], WINDOW_RADIUS, theta_resolution = THETA_RESOLUTION, theta_window=theta_window)
#
#     x, y = zip(*offset_first_frame.items())
#     plt.plot(x,y)
#     plt.show()
#
#     # Create dictionary of time values corresponding to the frames in frame_index_list
#     frame_time = {}
#     dt = float(1/video.fps)
#     for idx in frame_index_list:
#         frame_time[idx] = idx * dt
#
#     # Initialize data object
#     data = dat.Data()
#     data.frame_index_list = frame_index_list
#     data.frame_time = frame_time
#     data.torsion = offset_first_frame
#     data.video_str = settings.VIDEO_PATH
#     data.video_fps = video.fps
#
#     data.save_data('test2D_anthony_provided_vid', settings.SAVE_PATH)
#
#     scroll.torsion_scroll(video,pupil_list,offset_first_frame)
#     scroll.window_scroll(video,pupil_list,offset_first_frame,theta_window,WINDOW_RADIUS)
