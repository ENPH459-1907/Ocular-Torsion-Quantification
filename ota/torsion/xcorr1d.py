from scipy import signal
import numpy as np

class DifferentSignalShapeError(Exception):
    def __init__(self, message):
        self.message = message

def signal_offset(sig1, sig2):
    if len(sig1) != len(sig2):
        raise DifferentSignalShapeError('Sig1 and Sig2 must be of the same length.')

    l = len(sig1)
    c = signal.correlate(sig1, sig2)
    offset = l - np.argmax(c)

    return offset