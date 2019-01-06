import os
import sys

# UI
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfile
import matplotlib
matplotlib.use("TkAgg")
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import time
import datetime
from plotly.offline import plot
import plotly.graph_objs as go

# OTA tools
from ota.gui import coord_click as clk
from ota.gui import frame_scroll as scroll
from ota.video import video as vid
from ota.execution import pupil_locate as pl
from ota.execution import torsion_quant_2DX as tq2dx
from ota.data import data as dat
from ota.iris import iris, eyelid_removal


LARGE_FONT= ("Verdana", 18)

class OcularTorsionApplication(tk.Tk):
    '''
    Object that allows the user to interface with multiple torsion quantification methods.

    Attributes:
        video: video object
        start_frame: integer
        reference_frame: reference frame from which torsion is measured
        end_frame: integer
        save_path: string, local directory location to save results to
        pupil_list: dictionary of pupil objects
                    key: (int) video frame
                    value: pupil object
        torsion: a list that holds iris rotations results
        frame: dictionary that holds the pages of the GUI
               key: (str) name of the frame
               value: GUI page
    '''

    def __init__(self, *args, **kwargs):
        '''
        Create a torsion application object
        '''

        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(10, weight=1)
        container.grid_columnconfigure(10, weight=1)

        # Values that are common to all torsion methods
        self.video_path = tk.StringVar()
        self.video = None

        self.start_frame = tk.IntVar()
        self.end_frame = tk.IntVar()
        self.reference_frame = tk.IntVar()

        self.save_path = tk.StringVar()

        self.pupil_list = None
        self.pupil_threshold = tk.IntVar()
        self.data = []

        self.torsion = []

        # Dictionary to store all the frames (pages) in the UI
        self.frames = {}

        for F in (StartPage, MeasureTorsion):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def run(self, measure_state):
        '''
        Runs the 2D correlation algorithm based on GUI state and common values.

        Input:
            measure_state: current state of measurement page of the GUI
        '''

        # Extract required values from GUI page
        RADIUS = measure_state.radial_thickness.get()
        RESOLUTION = measure_state.resolution.get()

        # Determine whether interpolation or upsampling should be used
        if measure_state.Interpolation.get():
            torsion_mode = 'interp'
        elif measure_state.Upsampling.get():
            torsion_mode = 'upsample'

        # Determine if the user desires locations of the reference frame to be replaced by noise
        if measure_state.NoiseReplacement.get():
            replace_status = 'noise_replace'
            upper_iris = measure_state.upper_iris_occ
            lower_iris = measure_state.lower_iris_occ
        else:
            replace_status = 'no_noise_replace'
            upper_iris = None
            lower_iris = None

        # Determine if the user wants to run 2D correlation on the whole iris
        if measure_state.Fulliris.get():
            # Set the transform mode and quantify torsion
            transform_mode = 'full'
            torsion = tq2dx.quantify_torsion(RADIUS, RESOLUTION, torsion_mode, transform_mode, self.video, self.start_frame.get(), self.reference_frame.get(), self.end_frame.get(), self.pupil_list, self.pupil_threshold.get(), upper_iris = upper_iris, lower_iris = lower_iris)
            # Construct metadata
            metadata = 'Mode: %(torsion_mode)s, Iris: %(transform_mode)s, %(replace_status)s, Radial Thickness (pix): %(radial_thickness)d, Video Path: %(video_path)s, Video FPS: %(video_fps)s' % \
                            {"torsion_mode": torsion_mode, "transform_mode": transform_mode, "replace_status": replace_status, "radial_thickness": measure_state.radial_thickness.get(), "video_path": self.video_path.get(),"video_fps": self.video.fps}
            metadata_dict = {'Mode': torsion_mode,
                             'Iris': transform_mode,
                             'Replace': replace_status,
                             'Thickness': measure_state.radial_thickness.get(),
                             'Video': self.video_path.get(),
                             'VIDEO_FPS': self.video.fps,
                             'REFERENCE_FRAME': self.reference_frame.get()}
            # Construct legend entry, which is a subset of the metadata
            legend_entry = 'Mode-%(torsion_mode)s_Iris-%(transform_mode)s_%(replace_status)s' % \
                            {"torsion_mode": torsion_mode, "transform_mode": transform_mode, "replace_status": replace_status}
            # Append torsion to the list as a tuple with the first element the results, second element as the metadata, third element as the legend entry
            self.torsion.append((torsion, metadata, legend_entry))

            # Initialize data object and append it to session list
            data = dat.Data(name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),path=self.save_path.get())
            torsion_data = [torsion_data[1] for torsion_data in torsion.items()]
            data.set(torsion = torsion_data, start_frame = self.start_frame.get(), pupil_list = self.pupil_list, metadata = metadata_dict)
            self.data.append(data)


        # Determine if the user wants to run 2D correlation on a subset of the iris
        elif measure_state.Subset.get():
            # Set the transform mode to subset
            transform_mode = 'subset'
            # Extract gui state values required for the subset method
            feature_coordinates = measure_state.feature_coordinates
            # Run the algorithm for each set of recorded feature coordinates
            for i, coords in enumerate(feature_coordinates):
                torsion_i = tq2dx.quantify_torsion(RADIUS, RESOLUTION, torsion_mode, transform_mode, self.video, self.start_frame.get(), self.reference_frame.get(), self.end_frame.get(), self.pupil_list, self.pupil_threshold.get(), WINDOW_THETA = measure_state.window_theta.get(), SEGMENT_THETA = measure_state.segment_theta.get(),feature_coords = coords)
                # Construct metadata
                metadata = 'Mode: %(torsion_mode)s, Iris: %(transform_mode)s, Window Theta (deg): %(window_theta)d, Segment Theta (deg): %(segment_theta)d, Radial Thickness (pix): %(radial_thickness)d, Feature Number: %(feature_num)d, Video Path: %(video_path)s, Video FPS: %(video_fps)d' % \
                            {"torsion_mode": torsion_mode, "transform_mode": transform_mode,"window_theta": measure_state.window_theta.get(),"segment_theta": measure_state.segment_theta.get(),"radial_thickness": measure_state.radial_thickness.get(),"feature_num": (i+1),"video_path": self.video_path.get(),"video_fps": self.video.fps}
                metadata_dict = {'Mode': torsion_mode,
                                 'Iris': transform_mode,
                                 'Window(deg)': measure_state.window_theta.get(),
                                 'Segment(deg)': measure_state.segment_theta.get(),
                                 'Feature Number': (i+1),
                                 'Thickness': measure_state.radial_thickness.get(),
                                 'Video': self.video_path.get(),
                                 'VIDEO_FPS': self.video.fps,
                                 'REFERENCE_FRAME': self.reference_frame.get()}
                # Construct legend entry, which is a subset of the metadata
                legend_entry = 'Mode: %(torsion_mode)s, Iris: %(transform_mode)s, Feature Number: %(feature_num)d' % \
                            {"torsion_mode": torsion_mode, "transform_mode": transform_mode, "feature_num": (i+1)}
                # Append torsion to the list as a tuple with the first element the results, second element as the metadata, third element as the legend entry
                self.torsion.append((torsion_i, metadata_dict, legend_entry))

                # Initialize data object and append it to session list
                data = dat.Data(name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),path=self.save_path.get())
                torsion_data = [torsion_data[1] for torsion_data in torsion_i.items()]
                data.set(torsion = torsion_data, start_frame = self.start_frame.get(), pupil_list = self.pupil_list, metadata = metadata_dict)
                self.data.append(data)


    def show_frame(self, cont):
        '''
        Display a frame.

        Inputs:
            cont - index of frame to display
        '''

        frame = self.frames[cont]
        frame.tkraise()

    def set_video_path(self):
        '''
        Set the path of the video.
        '''
        video_path = askopenfilename(initialdir = "/",title = "Select Video file",filetypes = (("AVI files","*.avi"),("all files","*.*")))
        if video_path:
            self.video_path.set(video_path)
            self.video = vid.Video(self.video_path.get())
            self.end_frame.set(len(self.video))

    def set_save_path(self):
        '''
        Set desired path to save the torsion results to.
        '''
        save_path = askdirectory(title="Select A Folder")
        self.save_path.set(save_path)

        if self.data:
            for dat in self.data:
                dat.path = save_path


    def save_results(self):
        '''
        Save the output of the torsion results to a CSV

        Inputs:
            data - data object storing the torsion results
        '''
        if self.data:
            for data in self.data:
                data.save()

    def scroll_frames(self):
        '''
        Scroll through video frames.
        '''
        scroll.frame_scroll(self.video)

    def scroll_pupil(self):
        '''
        Scroll through video frames while overlaying the pupil.
        '''
        if self.pupil_list:
            scroll.pupil_scroll(self.video, self.pupil_list)

    def view_axis_rotation(self):
        '''
        Scroll through the video while overlaying the pupil and a set of axis that rotate with torsion results.
        '''
        if self.torsion:
            last_run = self.torsion[0]
            scroll.torsion_scroll(self.video,self.pupil_list,last_run[0])

    def view_window_rotation(self, measure_state):
        '''
        Scroll through the video while overlaying the pupil and the window that rotates with the torsion results.

        Inputs:
            theta_window - tuple that stores the minimum and maximum angle above/below the feature
            WINDOW_RADIUS - radial thickness of the window
        '''
        if self.torsion:
            last_run = self.torsion[0]
            feature_coords = measure_state.feature_coordinates[0]
            feature_r, feature_theta = iris.get_polar_coord(feature_coords['r'], feature_coords['c'], self.pupil_list[self.start_frame.get()])
            reference_bounds = (feature_theta - measure_state.window_theta.get(), feature_theta + measure_state.window_theta.get())
            scroll.window_scroll(self.video,self.pupil_list,last_run[0],reference_bounds, measure_state.radial_thickness.get())

    def plot_torsion(self):
        '''
        plots all series within the torsion list in a plotly window. The legend shows all the legend entry relating to each series.
        '''
        data = []
        for (result, metadata, legend_entry) in self.torsion:

            x_i, y_i = zip(*result.items())
            trace = go.Scatter(x=x_i, y=y_i, name=legend_entry)
            data.append(trace)

        layout = go.Layout(title='Iris Rotation History',
                           xaxis=dict(title='Frame Number'),
                           yaxis=dict(title='Rotation (deg)'))

        plot({"data":data,"layout":layout})

    def construct_pupil_list(self, measure_torsion_button):
        '''
        Constructs a list of pupils.
        '''
        self.pupil_list = pl.construct_pupil_list(self.video, self.start_frame.get(), self.end_frame.get(), self.pupil_threshold.get())


class StartPage(tk.Frame):
    '''
    Main menu of the torsion application. Allows users to set the video path, set the save path, preview the video and select a torsion quantification
    method.
    '''

    def __init__(self, parent, controller):
        '''
        Creates a main menu.
        '''

        tk.Frame.__init__(self,parent)
        title_label = tk.Label(self, text="Main Menu", font=LARGE_FONT)
        title_label.grid(row=0,column=0,columnspan=3)

        self.measure_torsion_button = tk.Button(self, text="Measure Torsion", command=lambda: controller.show_frame(MeasureTorsion))
        self.measure_torsion_button.grid(row=8,column=2,sticky=tk.W)

        vid_button = tk.Button(self, text="Set Video Path", command=lambda:  controller.set_video_path())
        vid_button.grid(row=1,column=0,sticky=tk.W)

        video_path_label = tk.Label(self, textvariable=controller.video_path)
        video_path_label.grid(row=1,column=1,columnspan=2)

        scroll_vid_button = tk.Button(self, text="Preview Video", command=lambda: controller.scroll_frames())
        scroll_vid_button.grid(row=2,column=0,sticky=tk.W)

        save_path_button = tk.Button(self, text="Set Results Save Path", command=lambda: controller.set_save_path())
        save_path_button.grid(row=3,column=0,sticky=tk.W)

        save_path_label = tk.Label(self, textvariable=controller.save_path)
        save_path_label.grid(row=3,column=1,columnspan=2, sticky=tk.W)

        start_frame_label = tk.Label(self, text="Start Frame Number:")
        start_frame_label.grid(row=4, column=0,sticky=tk.W)

        start_frame_entry = tk.Entry(self, textvariable = controller.start_frame)
        start_frame_entry.grid(row=4, column=1)

        reference_frame_label = tk.Label(self, text='Reference Frame Number:')
        reference_frame_label.grid(row=5, column=0, sticky=tk.W)

        reference_frame_entry = tk.Entry(self, textvariable = controller.reference_frame)
        reference_frame_entry.grid(row=5, column=1)

        end_frame_label = tk.Label(self, text="End Frame Number:")
        end_frame_label.grid(row=6, column=0,sticky=tk.W)

        end_frame_entry = tk.Entry(self, textvariable = controller.end_frame)
        end_frame_entry.grid(row=6, column=1)

        pupil_threshold_label = tk.Label(self, text="Pupil Detection Threshold:")
        pupil_threshold_label.grid(row=7, column=0)

        pupil_threshold_entry = tk.Entry(self, textvariable = controller.pupil_threshold)
        pupil_threshold_entry.grid(row=7, column=1)

        pupil_loc_button = tk.Button(self, text="Construct Pupil List", command=lambda: controller.construct_pupil_list(self.measure_torsion_button))
        pupil_loc_button.grid(row=8,column=0,sticky=tk.W)

        pupil_scroll_button = tk.Button(self, text="Preview Pupil Locations", command=lambda: controller.scroll_pupil())
        pupil_scroll_button.grid(row=8,column=1)


class MeasureTorsion(tk.Frame):
    '''
    Measurement page of the torsion application.

    Attributes:
        Interpolation:
            Integer
            A value of 1 indicates interpolation will be used.
            A value of 0 indicates interpolation will not be used.

        Upsampling:
            Integer
            A value of 1 indicates upsampling will be used.
            A value of 0 indicates upsampling will not be used.

        Fulliris:
            Integer
            A value of 1 indicates that correlation will be performed on the whole iris.
            A value of 0 indicates that correlation will be performed on a subset of the iris.

        Subset:
            Integer
            A value of 1 indicates that correlation will be performed on a subset of the iris.
            A value of 0 indicates that correlation will be performed on the whole iris.

        NoiseReplacement:
            Integer
            A value of 1 indicates that portions of the iris will be replaced with noise.
            A value of 0 indicates that no replacement on the iris will be performed

        radial_thickness:
            Integer
            Radial thickenss of the iris

        upper_iris_occ:
            dictionary, {'c': column index, 'r': row index}
            Holds the [row,column] coordinates of the upper boundary of the iris that is not occluded by eyelids or eyelashes.

        lower_iris_occ:
            dictionary, {'c': column index, 'r': row index}
            Holds the [row,column] coordinates of the lower boundary of the iris that is not occluded by eyelids or eyelashes.

        resolution:
            Double
            If upsampling is used, resolution gives the degree of upsampling used during the iris transform.
            If interpolation is used, iris transform is performed using a transform resolution of 1deg/pixel and interpolated at increments given by resolution.

        feature_coordinates:
            List of dictionaries, {'c': column index, 'r': row index}
            Holds the dictionaries of feature coordinates tracked during subset correlation.

        num_features:
            Integer
            size of the feature_coordinates list.

        window_theta:
            Integer
            Angle bounds above/below the feature that define the portion of the iris that is to be included in the reference iris window. This window should be smaller than the segment.

        segment_theta:
            Integer
            Angle bounds above/below the feature that define the portion of the iris that is to be included in each segment, for which the window is to be located in.



    '''
    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)

        self.Interpolation = tk.IntVar()

        self.Upsampling = tk.IntVar()

        self.Fulliris = tk.IntVar()

        self.Subset = tk.IntVar()

        self.NoiseReplacement = tk.IntVar()

        # Thickness, beyond the pupil edge of both the iris segment and window.
        self.radial_thickness= tk.IntVar()

        # The [row,column] coordinates of point on the boundary of visible iris not occluded by upper eyelid or upper eyelashes.
        self.upper_iris_occ = {}

        # The [row,column] coordinates of point on the boundary of visible iris not occluded by lower eyelid or lower eyelashes.
        self.lower_iris_occ = {}

        # The degree of upsampling or interpolation used in the theta axis of transform. [deg/pixel].
        self.resolution= tk.DoubleVar()

        # List used to store the dictionary that holds the row and column indicies of the feature coordinates.
        self.feature_coordinates = []

        # Number of features in the list of feature coordinates.
        self.num_features = tk.IntVar()

        # Portion of the iris that is smaller than the segment.
        # Window sized portions of the segment are extracted and compared to this window.
        self.window_theta = tk.IntVar()

        # Portion of the iris in which the window is searched for.
        self.segment_theta = tk.IntVar()

        measure_torsion_label = tk.Label(self, text="Measure Torsion", font=LARGE_FONT)
        measure_torsion_label.grid(row=0,column=1, sticky=tk.W)

        main_button = tk.Button(self, text="Back to Main Menu", command=lambda: controller.show_frame(StartPage))
        main_button.grid(row=0,column=0, sticky=tk.W)

        measurement_options_label = tk.Label(self, text="Measurement Options", font=LARGE_FONT)
        measurement_options_label.grid(row=1,column=0, sticky=tk.W)

        interpolation_check = tk.Checkbutton(self, text="Interpolate", variable = self.Interpolation, command=lambda: [self.Upsampling.set(not(self.Interpolation.get())), self.update()])
        interpolation_check.grid(row=2,column=0, sticky=tk.W)

        upsampling_check = tk.Checkbutton(self, text="Upsample", variable = self.Upsampling, command=lambda: [self.Interpolation.set(not(self.Upsampling.get())), self.update()])
        upsampling_check.grid(row=3,column=0, sticky=tk.W)

        full_iris_check = tk.Checkbutton(self, text="Full Iris", variable = self.Fulliris, command=lambda: [self.Subset.set(not(self.Fulliris.get())), self.update()])
        full_iris_check.grid(row=2,column=1, sticky=tk.W)

        subset_check = tk.Checkbutton(self, text="Subset Iris", variable = self.Subset, command=lambda: [self.Fulliris.set(not(self.Subset.get())), self.update()])
        subset_check.grid(row=3,column=1, sticky=tk.W)

        noise_replacement_check = tk.Checkbutton(self, text="Noise Replacement", variable = self.NoiseReplacement, command=lambda: [self.update()])
        noise_replacement_check.grid(row=2,column=2, sticky=tk.W)

        radial_thickness_label = tk.Label(self, text="Radial Thickness (pixels):")
        radial_thickness_label.grid(row=4, column=0, sticky=tk.W)

        radial_thickness = tk.Entry(self, textvariable=self.radial_thickness)
        radial_thickness.grid(row=4, column=1, sticky=tk.W)

        resolution_label = tk.Label(self, text="Resolution (degree):")
        resolution_label.grid(row=5, column=0, sticky=tk.W)

        resolution = tk.Entry(self, textvariable=self.resolution)
        resolution.grid(row=5, column=1, sticky=tk.W)

        measurement_options_label = tk.Label(self, text="Measurement Settings", font=LARGE_FONT)
        measurement_options_label.grid(row=6,column=0, sticky=tk.W)

        self.upper_occ_get_button = tk.Button(self, text="Select Upper Occlusion Limit", command=lambda: self.get_occlusion_coordinates(controller))
        self.upper_occ_get_button.grid(row=7,column=0, sticky=tk.W)

        self.upper_occ_rec_button = tk.Button(self, text="Record Upper Occlusion Limit", command=lambda: self.record_upper_occ())
        self.upper_occ_rec_button.grid(row=7,column=1, sticky=tk.E)

        self.upper_set_check = tk.StringVar()
        self.upper_set_check.set('Not Set')
        upper_check_label = tk.Label(self, textvariable=self.upper_set_check)
        upper_check_label.grid(row=7,column=2, sticky=tk.W)

        self.lower_occ_get_button = tk.Button(self, text="Select lower Occlusion Limit", command=lambda: self.get_occlusion_coordinates(controller))
        self.lower_occ_get_button.grid(row=8,column=0, sticky=tk.W)

        self.lower_occ_rec_button = tk.Button(self, text="Record Lower Occlusion Limit", command=lambda: self.record_lower_occ())
        self.lower_occ_rec_button.grid(row=8,column=1, sticky=tk.E)

        self.lower_set_check = tk.StringVar()
        self.lower_set_check.set('Not Set')
        lower_check_label = tk.Label(self, textvariable=self.lower_set_check)
        lower_check_label.grid(row=8,column=2, sticky=tk.W)

        segment_theta_label = tk.Label(self, text="Iris Segment Bounds (deg):")
        segment_theta_label.grid(row=9, column=0, sticky=tk.W)

        self.segment_theta_entry = tk.Entry(self, textvariable=self.segment_theta)
        self.segment_theta_entry.grid(row=9, column=1, sticky=tk.E)

        window_theta_label = tk.Label(self, text="Iris Window Bounds (deg):")
        window_theta_label.grid(row=10, column=0, sticky=tk.W)

        self.window_theta_entry = tk.Entry(self, textvariable=self.window_theta)
        self.window_theta_entry.grid(row=10, column=1, sticky=tk.E)

        num_features_label = tk.Label(self, text="Number of features to track:")
        num_features_label.grid(row=11, column=0, sticky=tk.W)

        num_features_act_label = tk.Label(self, textvariable=self.num_features)
        num_features_act_label.grid(row=11, column=1)

        self.feature_loc_button = tk.Button(self, text="Select Feature", command=lambda: self.get_feature_coordinates(controller))
        self.feature_loc_button.grid(row=12,column=0, sticky=tk.W)

        self.feature_rec_button = tk.Button(self, text="Record Coordinates", command=lambda: self.record_feature_coordinates())
        self.feature_rec_button.grid(row=12,column=1, sticky=tk.W)


        self.remove_features_button = tk.Button(self, text="Clear Clicked Values", command=lambda: self.clear_coordinates())
        self.remove_features_button.grid(row=11,column=2, sticky=tk.W)

        measurement_options_label = tk.Label(self, text="Run and Save", font=LARGE_FONT)
        measurement_options_label.grid(row=13,column=0, sticky=tk.W)


        self.run_button = tk.Button(self, text="Run", command=lambda: controller.run(self))
        self.run_button.grid(row=14,column=0,  sticky=tk.W)

        self.save_button = tk.Button(self, text='Save to CSV', command=lambda: controller.save_results())
        self.save_button.grid(row=14,column=1, sticky=tk.W)

        measurement_options_label = tk.Label(self, text="Animate and Plot", font=LARGE_FONT)
        measurement_options_label.grid(row=15,column=0, sticky=tk.W)

        self.view_axis_button = tk.Button(self, text="Animate Axis Rotation", command=lambda: controller.view_axis_rotation())
        self.view_axis_button.grid(row=16,column=0, sticky=tk.W)

        self.view_window_button = tk.Button(self, text="Animate Window Location", command=lambda: controller.view_window_rotation(self))
        self.view_window_button.grid(row=16,column=1, sticky=tk.W)

        self.view_torsion_button = tk.Button(self, text="Plot Results", command=lambda: controller.plot_torsion())
        self.view_torsion_button.grid(row=16,column=2,sticky=tk.W)

        self.update()


    def get_occlusion_coordinates(self, controller):
        '''
        Opens the starting video frame in a separate window so that the user can click on the upper region of the iris not occluded by upper eyelid or upper eyelashes.
        Requires the user to separately 'record upper occlusion coordinates'
        '''
        clk.click_coordinates(controller.video[controller.start_frame.get()], 'Click On Undisturbed Iris Closest To Eyelid')

    def record_upper_occ(self):
        '''
        Stores upper occlustion coordinates clicked on to be used in the torsion quantification method.
        '''
        self.upper_iris_occ = clk.get_click_coordinates()
        self.upper_set_check.set('Set')

    def record_lower_occ(self):
        '''
        Stores lower occlustion coordinates clicked on to be used in the torsion quantification method.
        '''
        self.lower_iris_occ = clk.get_click_coordinates()
        self.lower_set_check.set('Set')

    def get_feature_coordinates(self, controller):
        '''
        Opens the starting video frame in a separate window so that the user can click on a prominent feature.
        Requires the user to separately 'record feature coordinates'
        '''

        clk.click_coordinates(controller.video[controller.start_frame.get()], 'Click on a distinct iris feature to track')

    def record_feature_coordinates(self):
        '''
        Stores feature coordinates clicked on to be used in the torsion quantification method.
        '''
        feature = clk.get_click_coordinates()
        self.feature_coordinates.append(feature)
        self.num_features.set(str(len(self.feature_coordinates)))

    def clear_coordinates(self):
        '''
        Clear the list of features if multiple have been clicked on and recorded.
        '''
        self.feature_coordinates = []
        self.num_features.set(str(len(self.feature_coordinates)))

        self.upper_iris_occ = None
        self.lower_iris_occ = None

        self.lower_set_check.set('Not Set')
        self.upper_set_check.set('Not Set')

    def update(self):
        '''
        Update the GUI page to enable/disable buttons and entry fields depending on the state.
        '''

        # If correlation is to be performed on the full iris, do not allow the user to enter parameters for the subset method.
        if self.Fulliris.get():
            self.segment_theta_entry.config(state='disabled')
            self.window_theta_entry.config(state='disabled')
            self.feature_loc_button.config(state='disabled')
            self.feature_rec_button.config(state='disabled')

        # If correlation is to be performed on a subset of the iris, allow the user to enter parameters for the subset method. Also do Not
        # allow the user to replace portions of the iris with noise.
        if self.Subset.get():
            self.segment_theta_entry.config(state='normal')
            self.window_theta_entry.config(state='normal')
            self.feature_loc_button.config(state='normal')
            self.feature_rec_button.config(state='normal')
            self.NoiseReplacement.set(0)


        # If the user wants to replace portions of the iris with noise, allow the user to use the occlusion setting function.
        if self.NoiseReplacement.get():
            self.upper_occ_get_button.config(state='normal')
            self.upper_occ_rec_button.config(state='normal')

            self.lower_occ_get_button.config(state='normal')
            self.lower_occ_rec_button.config(state='normal')
        # If the user does not want to replace portions of the iris with noise disable the occlusion setting buttons.
        else:
            self.upper_occ_get_button.config(state='disabled')
            self.upper_occ_rec_button.config(state='disabled')

            self.lower_occ_get_button.config(state='disabled')
            self.lower_occ_rec_button.config(state='disabled')

def run():

    app = OcularTorsionApplication()
    app.title('Ocular Torsion Measurement')
    app.mainloop()
