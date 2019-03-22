import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('.'), os.path.pardir)))
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from matplotlib.patches import Arrow
from matplotlib.patches import Wedge

class FrameTracker(object):
    """
    Object that displays a video frame. Class used by frame_scroll method.

    Parameters
    ------------------------
    ax : object containing elements of a figure
        Used to set window title and axis labels

    video : array_like
        series of video frames
    """
    def __init__(self, ax, video):
        self.ax = ax
        self.ax.set_title('Use keyboard to navigate images')
        self.video = video
        self.slices = len(self.video)
        self.ind = 0
        self.im = ax.imshow(self.video[self.ind])
        self.ax.set_xlabel('Frame %s' % self.ind)

    def on_key(self, event):
        if event.key == 'up':
            self.ind = (self.ind + 1) % self.slices
        elif event.key =='down':
            self.ind = (self.ind - 1) % self.slices
        elif event.key =='right':
            self.ind = (self.ind + 50) % self.slices
        elif event.key =='left':
            self.ind = (self.ind - 50) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.video[self.ind])
        self.ax.set_xlabel('Frame %s' % self.ind)
        self.im.axes.figure.canvas.draw()

class EyelidTracker(FrameTracker):
    """
    Object that displays a video frame with the located eyelid overlayed. Class used by eyelid_scroll method.

    Parameters
    ------------------------
    ax : object containing elements of a figure
        Used to set window title and axis labels

    video : array_like
        series of video frames

    eyelid : dictionary
        dictionary of pupils where the key is the frame index and the value is the frame with the eyelid removed.
        Does not need to include all video frames.
    """
    def __init__(self, ax, video, eyelid_list):
        FrameTracker.__init__(self, ax, video)
        self.eyelid_list = eyelid_list

    def update(self):
        display_img = self.video[self.ind]
        if self.ind in self.eyelid_list:
            self.eyelid_at_ind = self.eyelid_list[self.ind]
            if self.eyelid_at_ind is not None:
                display_img = self.eyelid_at_ind
        self.im.set_data(display_img)
        self.ax.set_xlabel('Frame %s' % self.ind)
        self.im.axes.figure.canvas.draw()

class PolarTransformTracker(FrameTracker):
    """
    Object that displays a video frame with the located eyelid overlayed. Class used by polar_transform_scroll method.

    Parameters
    ------------------------
    ax : object containing elements of a figure
        Used to set window title and axis labels

    video : array_like
        series of video frames

    eyelid : dictionary
        dictionary of pupils where the key is the frame index and the value is the frame with the eyelid removed.
        Does not need to include all video frames.
    """
    def __init__(self, ax, video, polar_transform_list):
        FrameTracker.__init__(self, ax, video)
        self.polar_transform_list = polar_transform_list

    def update(self):
        display_img = self.video[self.ind]
        if self.ind in self.polar_transform_list:
            self.polar_transform_at_ind = self.polar_transform_list[self.ind]
            if self.polar_transform_at_ind is not None:
                display_img = self.polar_transform_at_ind
        self.im.set_data(display_img)
        self.ax.set_xlabel('Frame %s' % self.ind)
        self.im.axes.figure.canvas.draw()

class PupilTracker(FrameTracker):
    """
    Object that displays a video frame with the located pupil overlayed. Class used by pupil_scroll method.

    Parameters
    ------------------------
    ax : object containing elements of a figure
        Used to set window title and axis labels

    video : array_like
        series of video frames

    pupil_list : dictionary
        dictionary of pupils where the key is the frame index and the value is the pupil.
        Does not need to include all video frames.
    """
    def __init__(self, ax, video, pupil_list):
        FrameTracker.__init__(self, ax, video)
        self.pupil_list = pupil_list

    def update(self):
        try:
            self.pupil_patch.remove()
            self.center_patch.remove()
        except AttributeError:
            pass
        except ValueError:
            pass

        if self.ind in self.pupil_list:
            self.pupil_at_ind = self.pupil_list[self.ind]
            if self.pupil_at_ind:
                self.pupil_circle = Ellipse((self.pupil_at_ind.center_col,self.pupil_at_ind.center_row), self.pupil_at_ind.major, self.pupil_at_ind.minor, angle=(self.pupil_at_ind.angle-90),fill=False,ec=[1,0,0])
                self.pupil_center = Circle((self.pupil_at_ind.center_col,self.pupil_at_ind.center_row),int(0.1*self.pupil_at_ind.radius),fill=True,ec=[1,0,0], fc=[1,0,0])
                self.pupil_patch = self.ax.add_patch(self.pupil_circle)
                self.center_patch = self.ax.add_patch(self.pupil_center)
            else:
                print('ERROR: No pupil at frame index %d' % (self.ind))

        self.im.set_data(self.video[self.ind])
        self.ax.set_xlabel('Frame %s' % self.ind)
        self.im.axes.figure.canvas.draw()


class TorsionTracker(FrameTracker):
    '''
    Torsion tracking object. Window updates x y axis to visualize torsion. Class used by torsion_scroll method.

    Parameters:
    ------------------------
                ax : object containing elements of a figure
                        Used to set window title and axis labels

                video : array_like
                        video to scroll through

                pupil_list : dictionary
                        dictionary of pupils where the key is the frame index and the value is the pupil.
                        Does not need to include all video frames.

                offset_first_frame : dictionary
                        dictionary of rotation angles. key is the frame index and the value is the rotation.
                        Does not need to include all video frames
    '''

    def __init__(self, ax, video, pupil_list, offset_first_frame):
        FrameTracker.__init__(self, ax, video)
        self.pupil_list = pupil_list
        self.torsion_list = offset_first_frame


    def update(self):
        try:
            self.pupil_patch.remove()
            self.center_patch.remove()
            self.x_patch.remove()
            self.y_patch.remove()
        except AttributeError:
            pass
        except ValueError:
            pass

        if self.ind in self.pupil_list:
            self.pupil_at_ind = self.pupil_list[self.ind]
            if self.pupil_at_ind:
                self.pupil_circle = Circle((self.pupil_at_ind.center_col,self.pupil_at_ind.center_row),self.pupil_at_ind.radius,fill=False,ec=[1,0,0])
                self.pupil_center = Circle((self.pupil_at_ind.center_col,self.pupil_at_ind.center_row),int(0.1*self.pupil_at_ind.radius),fill=True,ec=[1,0,0],fc=[1,0,0])
                self.pupil_patch = self.ax.add_patch(self.pupil_circle)
                self.center_patch = self.ax.add_patch(self.pupil_center)
            else:
                print('ERROR: No pupil at frame index %d' % (self.ind))

        if self.ind in self.torsion_list:
            self.angle = self.torsion_list[self.ind]
            radius = self.video.height/2
            if self.pupil_at_ind:
                self.x_axis = Arrow(self.pupil_at_ind.center_col, self.pupil_at_ind.center_row,radius*np.cos(np.pi*(self.angle)/180),-radius*np.sin(np.pi*(self.angle)/180), width = 5, ec=[1,0,0], fc=[1,0,0], fill=True)
                self.y_axis = Arrow(self.pupil_at_ind.center_col, self.pupil_at_ind.center_row,radius*np.cos(np.pi*(self.angle+90)/180),-radius*np.sin(np.pi*(self.angle+90)/180), width = 5, ec=[1,0,0], fc=[1,0,0], fill=True)
                self.x_patch = self.ax.add_patch(self.x_axis)
                self.y_patch = self.ax.add_patch(self.y_axis)
            else:
                print('ERROR: No pupil at frame index %d' % (self.ind))

        self.im.set_data(self.video[self.ind])
        self.ax.set_xlabel('Frame %s' % self.ind)
        self.im.axes.figure.canvas.draw()

class WindowTracker(FrameTracker):
    '''
    Window tracking object. Window updates window location while frames are scrolling. Class used by window_scroll method.

    Parameters:
    ------------------------
                ax : object containing elements of a figure
                        Used to set window title and axis labels

                video : array_like
                        video to scroll through

                pupil_list : dictionary
                        dictionary of pupils where the key is the frame index and the value is the pupil.
                        Does not need to include all video frames.

                offset_first_frame : dictionary
                        dictionary of rotation angles. key is the frame index and the value is the rotation.
                        Does not need to include all video frames

                theta_window : tuple of angles
                        theta[0] is the lower bound of the window
                        theta[1] is the upper bound of the window

                WINDOW_RADIUS: integer
                        Pixel width of the window radius
    '''
    def __init__(self, ax, video, pupil_list, offset_first_frame,theta_window,WINDOW_RADIUS):
        FrameTracker.__init__(self, ax, video)
        self.pupil_list = pupil_list
        self.offset_first_frame = offset_first_frame
        self.theta_window = theta_window
        self.WINDOW_RADIUS = WINDOW_RADIUS

    def update(self):
        try:
            self.pupil_patch.remove()
            self.center_patch.remove()
            self.window_patch.remove()
        except AttributeError:
            pass
        except ValueError:
            pass

        if self.ind in self.pupil_list:
            self.pupil_at_ind = self.pupil_list[self.ind]
            if self.pupil_at_ind:
                self.pupil_circle = Circle((self.pupil_at_ind.center_col,self.pupil_at_ind.center_row),self.pupil_at_ind.radius,fill=False,ec=[1,0,0])
                self.pupil_center = Circle((self.pupil_at_ind.center_col,self.pupil_at_ind.center_row),int(0.1*self.pupil_at_ind.radius),fill=True,ec=[1,0,0],fc=[1,0,0])
                self.pupil_patch = self.ax.add_patch(self.pupil_circle)
                self.center_patch = self.ax.add_patch(self.pupil_center)
            else:
                print('ERROR: No pupil at frame index %d' % (self.ind))

        if self.ind in self.offset_first_frame:
            self.angle = self.offset_first_frame[self.ind]
            radius = self.video.height/2
            self.window = Wedge((self.pupil_at_ind.center_col,self.pupil_at_ind.center_row),self.pupil_at_ind.radius+self.WINDOW_RADIUS,-(self.theta_window[1]+self.angle),-(self.theta_window[0]+self.angle),self.WINDOW_RADIUS,fill=False,ec=[1,0,0])
            self.window_patch = self.ax.add_patch(self.window)

        self.im.set_data(self.video[self.ind])
        self.ax.set_xlabel('Frame %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def frame_scroll(video):
    '''
    Allows user to scroll through video frames using the keyboard.

    Parameters:
    ------------------------
                video : array_like
                        video to scroll through
    '''
    fig, ax = plt.subplots(1, 1)
    tracker = FrameTracker(ax, video)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key)
    plt.show()

def eyelid_scroll(video, eyelid_list):
    '''
    Overlays eyelid during frame scroll

    Parameters:
    ------------------------
                video : array_like
                        video to scroll through

                eyelid_list : dictionary
                        dictionary of pupils where the key is the frame index and the value is the pupil.
                        Does not need to include all video frames.
    '''
    fig, ax = plt.subplots(1, 1)
    tracker = EyelidTracker(ax, video, eyelid_list)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key)
    plt.show()


def pupil_scroll(video,pupil_list):
    '''
    Overlays pupil during frame scroll

    Parameters:
    ------------------------
                video : array_like
                        video to scroll through

                pupil_list : dictionary
                        dictionary of pupils where the key is the frame index and the value is the pupil.
                        Does not need to include all video frames.
    '''
    fig, ax = plt.subplots(1, 1)
    tracker = PupilTracker(ax, video, pupil_list)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key)
    plt.show()

def torsion_scroll(video, pupil_list, offset_first_frame):
    '''
    Tracks torsion using rotating 2D axis during frame scroll.

    Parameters:
    ------------------------
                video : array_like
                        video to scroll through

                pupil_list : dictionary
                        dictionary of pupils where the key is the frame index and the value is the pupil.
                        Does not need to include all video frames.

                offset_first_frame : dictionary
                        dictionary of rotation angles. key is the frame index and the value is the rotation.
                        Does not need to include all video frames
    '''
    fig, ax = plt.subplots(1,1)
    tracker = TorsionTracker(ax, video, pupil_list, offset_first_frame)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key)
    plt.show()

def polar_transform_scroll(video, polar_transform_list):
    fig, ax = plt.subplots(1,1)
    tracker = PolarTransformTracker(ax, video, polar_transform_list)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key)
    plt.show()

def window_scroll(video,pupil_list,offset_first_frame,theta_window,WINDOW_RADIUS):
    '''
    Tracks window location during frame scroll.

    Parameters:
    ------------------------
                video : array_like
                        video to scroll through

                pupil_list : dictionary
                        dictionary of pupils where the key is the frame index and the value is the pupil.
                        Does not need to include all video frames.

                offset_first_frame : dictionary
                        dictionary of rotation angles. key is the frame index and the value is the rotation.
                        Does not need to include all video frames

                theta_window : tuple of angles
                        theta[0] is the lower bound of the window
                        theta[1] is the upper bound of the window

                WINDOW_RADIUS: integer
                        Pixel width of the window radius
    '''
    fig, ax = plt.subplots(1,1)
    tracker = WindowTracker(ax, video, pupil_list, offset_first_frame, theta_window,WINDOW_RADIUS)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key)
    plt.show()
