import sys
import freetype

class Font(object):
    """See:
    http://dbader.org/blog/monochrome-font-rendering-with-freetype-and-python"""

    def __init__(self, filename, size):
        try:
            self.face = freetype.Face(filename)
        except freetype.ft_errors.FT_Exception:
            print "Failed to load font: {}".format(filename)
            sys.exit(2)

        self.face.set_pixel_sizes(0, size)

    def glyph_for_character(self, char):
        self.face.load_char(char, freetype.FT_LOAD_RENDER |
                            freetype.FT_LOAD_TARGET_MONO)
        return Glyph.from_glyphslot(self.face.glyph)
        
    def render_character(self, char):
        glyph = self.glyph_for_character(char)
        return glyph.bitmap

    def text_dimensions(self, text):
        """Return (width, height, baseline) of text rendered in current font"""

        width = 0
        max_ascent = 0
        max_descent = 0
        previous_char = None

        for char in text:
            glyph = self.glyph_for_character(char)
            max_ascent = max(max_ascent, glyph.ascent)
            max_descent = max(max_descent, glyph.descent)
            width += glyph.advance_width
            previous_char = char

        height = max_ascent + max_descent

        return width, height, max_descent

    def render_text(self, text, width=None, height=None, baseline=None):
        """Render text into a Bitmap and return it"""

        if None in (width, height, baseline):
            width, height, baseline = self.text_dimensions(text)

        x = 0
        previous_char = None
        outbuffer = Bitmap(width, height)

        for char in text:
            glyph = self.glyph_for_character(char)
            y = height - glyph.ascent - baseline
            outbuffer.bitblt(glyph.bitmap, x, y)
            x += glyph.advance_width
            previous_char = char

        return outbuffer

class Glyph(object):

    def __init__(self, pixels, width, height, top, advance_width):
        self.bitmap = Bitmap(width, height, pixels)

        self.top = top

        self.descent = max(0, self.height - self.top)
        self.ascent = max(0, max(self.top, self.height) - self.descent)

        # Horizontal distance to place next character
        self.advance_width = advance_width

    @property
    def width(self):
        return self.bitmap.width

    @property
    def height(self):
        return self.bitmap.height

    @staticmethod
    def from_glyphslot(slot):        
        pixels = Glyph.unpack_mono_bitmap(slot.bitmap)
        width, height = slot.bitmap.width, slot.bitmap.rows
        top = slot.bitmap_top

        advance_width = slot.advance.x / 64
        return Glyph(pixels, width, height, top, advance_width)

    @staticmethod
    def unpack_mono_bitmap(bitmap):
        """Unpack a freetype FT_LOAD_TARGET_MONO glyph bitmap into a bytearray 
        where each pixel is represented as a single byte."""

        data = bytearray(bitmap.rows * bitmap.width)

        # Iterate over packed byted in the input bitmap
        for y in range(bitmap.rows):
            for byte_index in range(bitmap.pitch):
                
                # Read byte with packed pixel data
                byte_value = bitmap.buffer[y * bitmap.pitch + byte_index]

                # Update how many bits we've processed
                num_bits_done = byte_index * 8

                # Work out where to write pixels we're going to unpack
                rowstart = y * bitmap.width + byte_index * 8

                # Iterate over each bit that's part of the output bitmap
                for bit_index in range(min(8, bitmap.width - num_bits_done)):
                    
                    # Unpack next pixel
                    bit = byte_value & (1 << (7 - bit_index))

                    # Write pixel to output bytearray
                    data[rowstart + bit_index] = 1 if bit else 0

        return data
    
class Bitmap(object):
    """2D bitmap image as list of byte values."""

    def __init__(self, width, height, pixels=None):
        self.width = width
        self.height = height
        self.pixels = pixels or bytearray(width * height)

    def __repr__(self):
        """String representation of bitmap's pixels."""
        rows = ''
        for y in range(self.height):
            for x in range(self.width):
                rows += '*' if self.pixels[y * self.width + x] else ' '
            rows += '\n'

        return rows

    def bitblt(self, src, x, y):
        srcpixel = 0
        dstpixel = y * self.width+ x
        row_offset = self.width - src.width

        for sy in range(src.height):
            for sx in range(src.width):
                self.pixels[dstpixel] = src.pixels[srcpixel]
                srcpixel += 1
                dstpixel += 1
            dstpixel += row_offset

