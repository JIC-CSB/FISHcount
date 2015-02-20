import skimage.filter
import skimage.measure
from skimage.morphology import remove_small_objects as _remove_small_objects
from skimage.morphology import disk, binary_closing, watershed, dilation
from skimage.exposure import equalize_adapthist
import scipy.misc
import numpy as np

autosave=True

def make_named_transform(name):
    """Decorator function. Takes a function operating on a ndarray and returning
    a ndarray and turns it into a function operating on an ImageArray and 
    returning an ImageArray, with the given name parameter."""

    def make_transform(func):

        def func_as_transform(image, name=name, **kwargs):

            image_array = image.image_array

            ret_array = func(image_array, **kwargs)

            ia = ImageArray(ret_array, name)
            ia.history += [image.history]

            return ia

        return func_as_transform

    return make_transform

class ImageArray(object):

    def __init__(self, image_array, name=None):
        self.image_array = image_array
        self.name = name

        if autosave:
            self.save()

        self.history = []

    def save(self, filename=None):
        if filename is None:
            if self.name is None:
                raise Exception('Cannot save with no name!')
            else:
                filename = self.name + '.png'

        scipy.misc.imsave(filename, self.image_array)

def projection_by_function(sa, z_function):
    """Generate a projection by applying the given function to each line of
    constant x, y in the image."""

    xmax, ymax, _ = sa.shape

    projection = np.zeros((xmax, ymax), dtype=sa.dtype)

    for x in range(xmax):
        for y in range(ymax):
            projection[x, y] = z_function(sa[x, y, :])

    return projection

def min_intensity_projection(stack, name='min_projection'):
    """Return minimum intensity projection for stack."""

    sa = stack.stack_array

    min_proj = projection_by_function(sa, min)

    ia = ImageArray(min_proj, name)
    ia.history = stack.history + ['min intensity projection from stack']

    return ia

def max_intensity_projection(stack, name='projection'):
    """Return max intensity projection for stack."""

    sa = stack.stack_array

    iz_max = np.argmax(sa, 2)

    xmax, ymax, _ = sa.shape

    projection = np.zeros((xmax, ymax), dtype=sa.dtype)

    for x in range(xmax):
        for y in range(ymax):
            projection[x, y] = sa[x, y, iz_max[x, y]]

    ia = ImageArray(projection, name)
    ia.history = stack.history + ['max intensity projection from stack']

    return ia

def equalize_adaptive(image, n_tiles=8, clip_limit=0.01, name='equalize_adaptive'):
    
    eqproj = equalize_adapthist(image.image_array, 
                                ntiles_x=n_tiles, 
                                ntiles_y=n_tiles,
                                clip_limit=clip_limit)

    ia = ImageArray(eqproj, name)
    ia.history = image.history + [name]

    return ia

def gaussian_filter(image, sigma=0.4, name='gaussian_filter'):

    gauss = skimage.filter.gaussian_filter(image.image_array, sigma=sigma)

    ia = ImageArray(gauss, name)
    ia.history = image.history + [name]

    return ia


@make_named_transform('find_edges')
def find_edges(ndarray):
    return skimage.filter.sobel(ndarray)

@make_named_transform('thresh_otsu')
def threshold_otsu(ndarray, mult=1):

    otsu_value = skimage.filter.threshold_otsu(ndarray)

    return ndarray > mult * otsu_value

def find_connected_components(image, neighbors=8, background=None, 
                              name='connected_components'):
    """Find connected components in the given image, returning an image labelled
    with the component ids. Because background components end up labelled -1, we
    add 1 to all return values."""

    connected_components, n_cc = skimage.measure.label(image.image_array, 
                                                       neighbors=8, 
                                                       background=background,
                                                       return_num=True)

    ia = ImageArray(1 + connected_components, name)
    ia.history = image.history + [name]

    return ia

def remove_small_objects(image, min_size=50, name='remove_small_objects'):

    nosmall = _remove_small_objects(image.image_array, min_size=min_size)

    ia = ImageArray(nosmall, name)
    ia.history = image.history + [name]

    return ia

def close_holes(image, salem=None, name='close_holes'):

    if salem is None:
        salem = disk(3)

    closed = binary_closing(image.image_array, salem)

    ia = ImageArray(closed, name)
    ia.history = image.history + [name]

    return ia

def watershed_with_seeds(image, seed_image, mask_image=None, name='watershed'):
    """Perform watershed segmentation from given seeds. Inputs should be of the
    form:

    image : grayscale image, with higher values representing more signal

    seed_image : grayscale image where each pixel value represents a unique
    region"""

    if mask_image is None:
        mask = None
    else:
        mask = mask_image.image_array

    # We multiply the image by -1 because the algorithm implementation expects
    # higher values to be easier for the 'water' to pass
    segmented = watershed(-image.image_array, 
                          seed_image.image_array, 
                          mask=mask)


    ia = ImageArray(segmented, name)
    ia.history = image.history + [name]

    return ia

def component_find_centroid(connected_components, index):
    loc = np.mean(np.where(connected_components.image_array == index), axis=1)

    x, y = map(int, loc)

    return x, y

def component_centroids(connected_components, name='component_centroids'):
    """Given a set of connected components as an image where the pixel value
    representst the component ID, reduce each component to its centroid."""

    component_ids = set(np.unique(connected_components.image_array)) - set([0])
    component_centroids = np.zeros(connected_components.image_array.shape,
                                   dtype=connected_components.image_array.dtype)
    for index in component_ids:
        x, y = component_find_centroid(connected_components, index)
        component_centroids[x, y] = index

    ia = ImageArray(component_centroids, name)
    ia.history = connected_components.history + [name]

    return ia

def dilate_simple(image, name='dilate_simple'):
    
    dilated = dilation(image.image_array)

    ia = ImageArray(dilated, name)
    ia.history = connected_components.history + [name]

    return ia
