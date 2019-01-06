'''
Calculate the amount of torsion via phase correlation.
'''
import numpy as np
from skimage.feature import register_translation

def phase_correlation(fixed, moved, polar=True):
    '''
    Find the translation of moved frame relative to the fixed frame using
    phase correlation.

    INPUTS
        fixed - Prior frame as NxM image
        moved - Moved frame relative to the fixed frame, also NxM image

    OUTPUTS
        shifts - Coordinates of relative shift
    '''

    shape = fixed.shape

    source = np.fft.fft2(fixed)
    target = np.fft.fft2(moved)

    product = source * target.conj()

    cross_correlation = np.fft.ifft2(product)

    product /= np.abs(product)

    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)

    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    # if polar:
    return shifts

def scipy_phase_correlation(fixed, moved, upsample_factor=1):
    '''
    Find the translation of moved frame relative to the fixed frame
    (using scipy package).

    Reference:
    # http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html
    # http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.register_translation

    INPUTS
        fixed - Prior frame as NxM image
        moved - Moved frame relative to the fixed frame, also NxM image

    OUTPUTS
        shifts - Coordinates of relative shift

    '''

    # From the following location:


    # pixel precision first
    shifts, error, diffphase = register_translation(fixed, moved, upsample_factor)

    return shifts
