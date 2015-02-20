"""Class representing z-stack."""

import numpy as np
import scipy
from skimage.exposure import equalize_adapthist

from imgutil import read_stack_array, stack_as_uint8, normalise2D

def normalise_stack(stack):
    """Normalise and return stack."""
    
    stack_array = np.dstack([normalise2D(stack.plane(n)) 
                               for n in range(stack.zdim)])

    s = Stack(stack_array)

    s.history = stack.history + ['normalised_stack']

    return s

def equalize_stack(stack):

    stack_array = np.dstack([equalize_adapthist(stack.plane(n)) 
                               for n in range(stack.zdim)])

    s = Stack(stack_array)

    s.history = stack.history + ['equalized_stack']

    return s

class Stack(object):
    """Class to represent a stack of 2D images comprising a 3D image. By default
    store the stack as an 3D array of uint8 values."""

    def __init__(self, stack_array):
        self.stack_array = stack_as_uint8(stack_array)

    @classmethod
    def from_path(cls, path, channel=0):
        stack_array = read_stack_array(path, channel)

        stack = cls(stack_array)

        stack.history = ['loaded from path: {}'.format(path)]

        return stack

    def save_plane(self, filename, z_index):

        scipy.misc.imsave(filename, self.stack_array[:,:,z_index])

    def plane(self, z_index):

        return self.stack_array[:,:,z_index]

    @property
    def zdim(self):
        xd, yd, zd = self.stack_array.shape

        return zd
