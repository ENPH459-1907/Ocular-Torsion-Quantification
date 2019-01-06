'''
Generate manually rotated frames for testing torsion quantification methods.
'''
import cv2
import numpy as np


# TODO add noise, randomness and cyclic rotations
def make_rotations(image, max_angle, num_frames=None, resolution=1, transform=None):
    '''
    Create a list of manually equally rotated images up to a specified max angle.

    If transform function is provided, apply transform to rotated images. The only
    input to the transform function should be an image, for extra paramters use a
    lambda function.

    ex. transform = lambda x: some_transform(x, other_param=1)

    INPUT
        image - NxM image as numpy array
        max_angle - Maximam rotation of first image
        num_frames - Specify the number of frames to make, by default the number of
            frames is max_angle / resolution
        resolution - By default the rotated degree is a whole number, by decreasing
            the resolution you decrease the degree per change in frame
        transform -

    OUTPUT
        frames - a list of rotated frames ranging from 0 degrees of rotation to
            max_angle degrees of rotation

    '''

    frames = []

    # number of frames to make
    if num_frames is None:
        num_frames = int(max_angle / resolution)

    # get the rows and columns of the original image
    rows, cols = image.shape

    # TODO non-equispaced
    # create an "equispaced" list of angles
    angles = [i / num_frames * max_angle for i in range(0,num_frames+1)]

    for angle in angles:

        # create the rotation matrix
        M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)

        # rotate the original image by the specified angle
        rotated = cv2.warpAffine(image, M, (cols, rows))

        # if transform function is provided, apply transform to rotated images
        # the only input to the transform function should be an image
        # For extra paramters use a lambda function
        # ex. transform = lambda x: some_transform(x, other_param=1)
        if transform:
            frames.append(transform(rotated))
        else:
            frames.append(rotated)

    return frames
