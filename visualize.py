import numpy as np
from scipy.interpolate import RegularGridInterpolator
from utils import unit_vector

# epsilon for checking if a directional vector is zero
EPS = 1e-10
# unit vector to generate a directional vector of the cross-section plane
U = [1, 0, 0]


def render_cross_section(data, center, normal, image_range=(128., 128.), image_resolution=(512, 512)):
    """Render a cross section of a 3D volume."""
    data_x, data_y, data_z = data.shape
    image_range_x, image_range_y = image_range
    image_resolution_x, image_resolution_y = image_resolution

    xxx, yyy, zzz = np.linspace(0, data_x, data_x, endpoint=False), np.linspace(
        0, data_y, data_y, endpoint=False), np.linspace(0, data_z, data_z, endpoint=False)

    # normal, dx, dy are mutually perpendicular
    u = np.array(U)
    dx = u - (np.dot(u, normal) / np.dot(normal, normal)) * normal
    if np.linalg.norm(dx) < EPS:
        raise ValueError('dx is zero')
    dx = unit_vector(dx)
    dy = np.cross(normal, dx)
    dy = unit_vector(dy)
    origin = center - (image_range_x/2) * dx - (image_range_y/2) * dy

    ys, xs = np.ogrid[:image_range_x:image_range_x/image_resolution_x,
                      :image_range_y:image_range_y/image_resolution_y]
    grid = origin + ys[..., np.newaxis] * dy + xs[..., np.newaxis] * dx

    image = RegularGridInterpolator(
        (xxx, yyy, zzz), data, bounds_error=False, fill_value=0.0)(grid)

    return image
