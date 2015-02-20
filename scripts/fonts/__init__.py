import freetype

__all__ = ['load',
            'FreeMono']

def load(filename):
    """Load a font and return it"""

    face = freetype.Face(filename)

    return face
