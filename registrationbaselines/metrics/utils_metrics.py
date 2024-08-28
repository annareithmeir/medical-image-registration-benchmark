from typing import Tuple

import numpy as np
import scipy.spatial

from registrationbaselines.core.types import intArray3D, intArray2D, intArray1D


def find_points_inside_convex_hull(points: intArray2D,
                                   hull: scipy.spatial.ConvexHull,
                                   image_shape: Tuple[int, ...]
                                   ) -> Tuple[intArray1D, intArray1D, intArray1D]:
    """
    deln = scipy.spatial.Delaunay(points[hull.vertices]):   This is where Delaunay triangulation comes in.
            It's applied to the vertices of the convex hull. Delaunay triangulation creates a triangulation
            of points such that no point is inside the circumcircle of any triangle.

    idx = np.stack(np.indices(image.shape), axis=-1): This creates an array of all possible coordinates in the
            image shape.

    out_idx = np.nonzero(deln.find_simplex(idx) + 1): deln.find_simplex(idx) checks which simplex (triangle in 2D,
            tetrahedron in 3D) each point in idx belongs to.
            Adding 1 and using np.nonzero() effectively finds all points that are inside the convex hull.

    @param points: The points inside the convex hull.
    @param hull: The convex hull.
    @param image_shape: The shape of the image.
    @return: The indices of the points inside the convex hull.
    """

    deln = scipy.spatial.Delaunay(points[hull.vertices])

    idx = np.stack(np.indices(image_shape), axis=-1)

    out_idx = np.nonzero(deln.find_simplex(idx) + 1)

    return out_idx


def get_convex_hull_mask(image: intArray3D) -> intArray3D:
    """
    Creates a mask from the convex hull. All values outside of the hull are set to 0.

    Usefull for when e.g. segmentations in the deformed image are outside of the FOV of the fixed image.

    ToDo - what does Delaunay do?

    @param image: The image.
    @return: The mask.
    """
    points = np.transpose(np.where(image))

    hull = scipy.spatial.ConvexHull(points)

    out_idx = find_points_inside_convex_hull(points,
                                             hull,
                                             image.shape)

    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1

    out_img = out_img.astype(np.uint8)

    return out_img
