#pylint: disable=invalid-name

class Rect(object):
    def __init__(self, *args):
        super(Rect, self).__init__()
        nargs = len(args)
        if nargs == 4:
            self._rect = list(args)
        elif nargs == 1:
            self._rect = list(args[0])
        elif nargs == 0:
            self._rect = [0, 0, 0, 0]
        else: 
            raise ValueError(f"Invalid arguments to Rect: {args}")

    def __iter__(self):
        for val in self._rect:
            yield val
    
    def __getitem__(self, key):
        return self._rect[key]

    def __setitem__(self, key, val):
        self._rect[key] = val
    
    def __str__(self):
        return f"{{x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}}}"

    def __repr__(self):
        return f"<Rect {self}>"

    def __eq__(self, other):
        return self._rect == list(other)

    def __ne__(self, other):
        return self._rect != list(other)

    def copy(self):
        return Rect(self)

    @property
    def x(self):
        return self._rect[0]

    @x.setter
    def x(self, val):
        self._rect[0] = val

    @property
    def y(self):
        return self._rect[1]
    
    @y.setter
    def y(self, val):
        self._rect[1] = val

    @property
    def w(self):
        return self._rect[2]

    @w.setter
    def w(self, val):
        self._rect[2] = val

    @property
    def h(self):
        return self._rect[3]

    @h.setter
    def h(self, val):
        self._rect[3] = val

    @property
    def left(self):
        return self.x

    @left.setter
    def left(self, val):
        self.w = self.right - val + 1
        self.x = val

    @property
    def right(self):
        return self.x + self.w -1

    @right.setter
    def right(self, val):
        self.w = val - self.x + 1

    @property
    def top(self):
        return self.y

    @top.setter
    def top(self, val):
        self.h = self.bottom - val + 1
        self.y = val

    @property
    def bottom(self):
        return self.y + self.h - 1

    @bottom.setter
    def bottom(self, val):
        self.h = val - self.y + 1

    @property
    def top_left(self):
        return (self.left, self.top)

    @property
    def top_right(self):
        return (self.right, self.top)

    @property
    def bottom_right(self):
        return (self.right, self.bottom)

    @property
    def bottom_left(self):
        return (self.left, self.bottom)

    @property
    def corners(self):
        return (self.top_left, self.bottom_left, self.bottom_right, self.top_right)

    @property
    def center(self):
        return (self.left, + (self.w - 1) * 0.5, self.top + (self.h - 1) * 0.5)
        
    def map_to_pixels(self):
        self.left = int(round(self.left))
        self.right = int(round(self.right))
        self.top = int(round(self.top))
        self.bottom = int(round(self.bottom))

    def scale(self, scale):
        if isinstance(scale, (int, float)):
            scale = [scale, scale]

        self.x = self.x * scale[0]
        self.y = self.y * scale[1]
        self.w = self.w * scale[0]
        self.h = self.h * scale[1]

    def pad(self, padding_x, padding_y=None):
        padding_y = padding_y if padding_y is not None else padding_x
        self.x = self.x - padding_x
        self.w = self.w + padding_x * 2
        self.y = self.y - padding_y
        self.h = self.h + padding_y * 2

    def crop(self, rect):
        self.left = max(self.left, rect.left)
        self.right = min(self.right, rect.right)
        self.top = max(self.top, rect.top)
        self.bottom = min(self.bottom, rect.bottom)

    def crop_image(self, img):
        return img[int(self.top):int(self.bottom) + 1, int(self.left):int(self.right) + 1]

    def contains_point(self, point):
        return point[0] >= self.left and point[0] <= self.right and point[1] >= self.top and point[1] <= self.bottom

    def contains_rect(self, rect):
        return rect.left >= self.left and rect.right <= self.right and rect.top >= self.top and rect.bottom <= self.bottom

    def intersects_rect(self, rect):
        return rect.left <= self.right and rect.right >= self.left and rect.top <= self.bottom and rect.bottom >= self.top

    def intersects_x(self, x, x_end=None):
        x_start = x
        x_end = x_end if x_end is not None else x
        return x_start <= self.right and x_end >= self.left

    def intersects_y(self, y, y_end=None):
        y_start = y
        y_end = y_end if y_end is not None else y
        return y_start <= self.bottom and y_end >= self.top

    def get_absloute_pos(self, relative_pos):
        return (relative_pos[0] + self.left, relative_pos[1] + self.top)

    def get_relative_pos(self, absolute_pos):
        return (absolute_pos[0] - self.left, absolute_pos[1] - self.top)

    @staticmethod
    def union(rect1, rect2):
        left = min(rect1.left, rect2.left)
        right = max(rect1.right, rect2.right)
        top = min(rect1.top, rect2.top)
        bottom = max(rect1.bottom, rect2.bottom)
        return Rect(left, top, right - left + 1, bottom - top + 1)