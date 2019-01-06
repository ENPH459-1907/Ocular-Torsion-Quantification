import numpy as np
import os
import cv2
import itertools

class OutOfIndexError(Exception):
    '''
    Frame being accessed was outside of the possible index values.
    '''
    def __init__(self, message):
        self.message = message

class ReadingImageError(Exception):
    '''
    Error reading image frame from video.
    '''
    def __init__(self, message):
        self.message = message

class VideoDoesNotExistError(Exception):
    '''
    Video file was not found.
    '''
    def __init__(self, message):
        self.message = message

class Video:
    '''
    Object to represent a video and its associated metadata.
    '''

    def __init__(self, path, grayscale=1):
        '''
        Create Video object from video file at specified path. By default images
        are grayscale.

        Inputs:
            path - relative or absolute location of video file
            grayscale - returns images as grayscale
        '''

        # open video file
        if not os.path.isfile(path):
            raise VideoDoesNotExistError('The video file {} does not exist'.format(path))

        self.path = os.path.abspath(path)
        self.capture = cv2.VideoCapture(self.path, 0)

        # properties of video
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.__length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # optional parameters
        self.grayscale = grayscale

    def __len__(self):
        '''
        Return the number of frames.
        '''
        return self.__length

    def __iter__(self):
        '''
        Object iteration.

        ex.
            v = Video(...)
            frames = [frame for frame in v]
        '''
        for i in range(len(self)):
            yield self[i]

    def __read_next(self, index):
        '''
        Returns the next frame at the specified index.
        '''

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)

        if len(self) < index or index < 0:
            raise OutOfIndexError('Please specify an index within 0 and {}'.format(len(self)))

        retval, image = self.capture.read()

        if not retval:
            raise ReadingImageError('Could not read current frame.')

        if self.grayscale == 1:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image

    def __getitem__(self, index):
        '''
        Array-like access for Video object.

        ex. v[0] or v[1:10]
        '''

        # temp generator so that we can return both an image or a generator
        def gen(start, stop, step):
            for ii in range(start, stop, step):
                yield self.__read_next(ii)

        # if users wants a slice return an iterable generator
        if isinstance( index, slice ):
            return gen(*index.indices(len(self)))
        # otherwise, return the image
        else:
            return self.__read_next(index)

    def elapsed_time(self):
        '''
        Returns current position of video file in milliseconds.
        '''
        return self.capture.get(cv2.CAP_PROP_POS_MSEC)
