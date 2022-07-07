from viewport import Viewport
from math import log
from PIL import Image
from PIL.ImageColor import getrgb
from dataclasses import dataclass
import numpy as np
np.warnings.filterwarnings("ignore")
from PIL import ImageEnhance
import matplotlib.cm
from scipy.interpolate import interp1d


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def stability(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.escape_count(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value


    def escape_count(self, c: complex, smooth=False) -> int | float:
        z = 0
        for iteration in range(self.max_iterations):
            z = z ** 2 + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iteration + 1 - log(log(abs(z))) / log(2)
        return self.max_iterations

    
    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1


def z_recursive(n, c):
    if n == 0:
        return 0
    else:
        return z_recursive(n-1, c) ** 2 + c


def z_iterative(c, z=0):
    while True:
        yield z
        z = z ** 2 + c


'''def is_stable(c, iterations):
    z = 0
    for _ in range(iterations):
        z = z ** 2 + c
    
    return abs(z) <= 2'''


def is_stable(c, max_iterations):
    z = 0
    for _ in range(max_iterations):
        z = z ** 2 + c
        if abs(z) > 2:
            return False
    return True


def get_members(c, iterations):
    mask = is_stable(c, iterations)
    return c[mask]


def mandelbrot(candidate):
    return z_iterative(c= candidate, z=0)


def julia(candidate, parameter):
    return z_iterative(c=parameter, z=candidate)


def complex_matrix(xmin, xmax, ymin, ymax, pixel_density):
    re = np.linspace(xmin, xmax, int((xmax-xmin) * pixel_density))
    im = np.linspace(ymin, ymax, int((ymax-ymin) * pixel_density))
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j


def paint(mandelbrot_set, viewport, palette, smooth):
    for pixel in viewport:
        stability = mandelbrot_set.stability(complex(pixel), smooth)
        index = int(min(stability * len(palette), len(palette) - 1))
        pixel.color = palette[index % len(palette)]


def denormalize(palette):
    return [
        tuple(int(channel * 255) for channel in color)
        for color in palette
        ]


def make_gradient(colors, interpolation="linear"):
    X = [i / (len(colors) - 1) for i in range(len(colors))]
    Y = [[color[i] for color in colors] for i in range(3)]
    channels = [interp1d(X, y, kind=interpolation) for y in Y]
    return lambda x: [np.clip(channel(x), 0, 1) for channel in channels]


def hsb(hue_degrees: int, saturation: float, brightness: float):
    return getrgb(
        f"hsv({hue_degrees % 360},"
        f"{saturation * 100}%,"
        f"{brightness * 100}%)"
        )


'''black = (0, 0, 0)
blue = (0, 0, 1)
maroon = (0.5, 0, 0)
navy = (0, 0, 0.5)
red = (1, 0, 0)

colors = [black, navy, blue, maroon, red, black]
gradient = make_gradient(colors, interpolation="cubic")
num_colors = 256

#colormap = matplotlib.cm.get_cmap("twilight").colors
#palette = denormalize(colormap)
palette = denormalize([gradient(i / num_colors) for i in range(num_colors)])

mandelbrot_set = MandelbrotSet(max_iterations=30, escape_radius=1000)
width, height = 3440, 1440
scale = 0.002
image = Image.new(mode='RGB', size=(width, height))
viewport = Viewport(image, center=-0.75, width=5)
paint(mandelbrot_set, viewport, palette, smooth=True)
image.show()'''

mandelbrot_set = MandelbrotSet(max_iterations=20, escape_radius=1000)
width, height = 3440, 1440
image = Image.new(mode='RGB', size=(width, height))
viewport = Viewport(image, center=-0.75, width=5)
for pixel in Viewport(image, center=-0.75, width=5):
    stability = mandelbrot_set.stability(complex(pixel), smooth=True)
    pixel.color = (0, 0, 0) if stability == 1 else hsb(
        hue_degrees=int(stability * 360),
        saturation=stability,
        brightness=1,
    )

image.show()

'''mandelbrot_set = MandelbrotSet(max_iterations=256, escape_radius=1000)
image = Image.new(mode="L", size=(3440, 1440))
for pixel in Viewport(image, center=-0.7435 + 0.1314j, width=0.002):
    c = complex(pixel)
    instability = 1 - mandelbrot_set.stability(c, smooth=True)
    pixel.color = int(instability * 255)

enhancer = ImageEnhance.Brightness(image)
enhancer.enhance(1.25).show()
image.show()'''