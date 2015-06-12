from functools import wraps

import numpy as np
import scipy.misc

from jicimagelib.transform import transformation
from jicimagelib.util.array import normalise

from skimage.exposure import equalize_adapthist
from skimage.morphology import watershed
#from skimage.morphology import remove_small_objects as _remove_small_objects
import skimage.measure
import skimage.morphology
import skimage.filters

def normalise_array(array):
    """Return array normalised such that all values are between 0 and 1."""

    min_val = array.min()
    max_val = array.max()

    array_range = max_val - min_val

    if array_range == 0:
        return array

    return (array.astype(np.float) - min_val) / array_range

def convert_to_uint8(array):
    """Convert the input array to uint8, by firstly normalising it and then
    multiplying by 255 and converting to np.uint8."""

    return (255 * normalise_array(array)).astype(np.uint8)

def uint8ify(transform_func):
    """Decorator to take a transformation function and ensure that it returns
    uint8."""

    @wraps(transform_func)
    def converted_transform(*args, **kwargs):
        transform_result = transform_func(*args, **kwargs)

        array_as_uint8 = 255 * normalise_array(transform_result)

        return array_as_uint8.astype(np.uint8)

    return converted_transform

@transformation
def max_intensity_projection(stack, name='projection'):
    """Return max intensity projection for stack."""

    iz_max = np.argmax(stack, 2)

    xmax, ymax, _ = stack.shape

    projection = np.zeros((xmax, ymax), dtype=stack.dtype)

    for x in range(xmax):
        for y in range(ymax):
            projection[x, y] = stack[x, y, iz_max[x, y]]

    return projection

def projection_by_function(sa, z_function):
    """Generate a projection by applying the given function to each line of
    constant x, y in the image."""

    xmax, ymax, _ = sa.shape

    projection = np.zeros((xmax, ymax), dtype=sa.dtype)

    for x in range(xmax):
        for y in range(ymax):
            projection[x, y] = z_function(sa[x, y, :])

    return projection

@transformation
def min_intensity_projection(stack, name='min_projection'):
    """Return minimum intensity projection for stack."""

    min_proj = projection_by_function(stack, min)

    return min_proj

def scale_median(input_image):
    """Normalise a single channel 8 bit image (as numpy array)."""

    norm_factor = 10. / np.median(input_image)

    return input_image * norm_factor

def scale_median_stack(stack):
    """Normalise and return stack."""
    
    _, _, zdim = stack.shape

    stack_array = np.dstack([scale_median(stack[:,:,n])
                               for n in range(zdim)])

    return stack_array

@transformation
def equalize_adaptive(image, n_tiles=8, clip_limit=0.01):
    
    eqproj = equalize_adapthist(image,
                                ntiles_x=n_tiles, 
                                ntiles_y=n_tiles,
                                clip_limit=clip_limit)

    return eqproj

@transformation
def gaussian_filter(image, sigma=0.4):

    gauss = skimage.filters.gaussian_filter(image, sigma=sigma)

    return gauss

@transformation
def find_edges(image):
    return skimage.filters.sobel(image)

@transformation
def threshold_otsu(ndarray, mult=1):

    otsu_value = skimage.filters.threshold_otsu(ndarray)

    return ndarray > mult * otsu_value

@transformation
def remove_small_objects(image, min_size=50):

    nosmall = skimage.morphology.remove_small_objects(image.astype(bool), 
                                                      min_size=min_size)

    return nosmall

def component_find_centroid(connected_components, index):
    loc = np.mean(np.where(connected_components == index), axis=1)

    x, y = map(int, loc)

    return x, y

#@transformation
# FIXME - Make into a transformation
def component_centroids(connected_components):
    """Given a set of connected components as an image where the pixel value
    representst the component ID, reduce each component to its centroid."""

    component_ids = set(np.unique(connected_components)) - set([0])
    component_centroids = np.zeros(connected_components.shape,
                                   dtype=connected_components.dtype)
    for index in component_ids:
        x, y = component_find_centroid(connected_components, index)
        component_centroids[x, y] = index

    return component_centroids

# FIXME - this would be nice as a transformation, but we have to handle saving
#@transformation
def find_connected_components(image, neighbors=8, background=None):
    """Find connected components in the given image, returning an image labelled
    with the component ids. Because background components end up labelled -1, we
    add 1 to all return values."""

    connected_components, n_cc = skimage.measure.label(image,
                                                       neighbors=8, 
                                                       background=background,
                                                       return_num=True)

    return connected_components

@transformation
def watershed_with_seeds(image, seed_image, mask_image=None):
    """Perform watershed segmentation from given seeds. Inputs should be of the
    form:

    image : grayscale image, with higher values representing more signal

    seed_image : grayscale image where each pixel value represents a unique
    region"""

    # We multiply the image by -1 because the algorithm implementation expects
    # higher values to be easier for the 'water' to pass
    segmented = watershed(-image,
                          seed_image,
                          mask=mask_image)


    return segmented

@transformation
def filter_segmentation(image_array, min_size=None):
    """Filter the given segmentation, removing objects smaller than the minimum
    size. If min_size is none, remove everything smaller than 10% of the mean
    size."""

    filtered_ia = np.copy(image_array)

    background = 0

    # Produce a dictionary of id : list of coordinates for each id representing
    # a region in the image
    by_coords = {index : zip(*np.where(image_array==index))
                 for index in np.unique(image_array)}

    if min_size is None:
        mean_size = np.mean(map(len, by_coords.values()))
        min_size = int(0.1 * mean_size)

    for index, coords in by_coords.items():
        if len(coords) < min_size:
            filtered_ia[zip(*coords)] = background

    return filtered_ia

#def component_find_centroid(connected_components, index):
#    loc = np.mean(np.where(connected_components.image_array == index), axis=1)
#
#    x, y = map(int, loc)
#
#    return x, y
