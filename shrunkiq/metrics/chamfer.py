import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist


def get_edge_points(image: np.ndarray | Image.Image) -> np.ndarray:
    """Convert image to edge points.

    Args:
        image: Input image (PIL Image or numpy array)

    Returns:
        np.ndarray: Nx2 array of edge point coordinates
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image.convert('L'))
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect edges using Canny
    edges = cv2.Canny(image, 100, 200)

    # Get coordinates of edge points
    points = np.column_stack(np.where(edges > 0))

    return points

def chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> tuple[float, float, float]:
    """Compute bidirectional Chamfer distance between two point sets.

    Args:
        points1: First set of points (Nx2 array)
        points2: Second set of points (Mx2 array)

    Returns:
        Tuple containing:
        - Forward distance (mean distance from points1 to nearest points2)
        - Backward distance (mean distance from points2 to nearest points1)
        - Symmetric Chamfer distance (mean of forward and backward)
    """
    # Compute pairwise distances between all points
    dists = cdist(points1, points2)

    # Forward distance: for each point in points1, find distance to nearest point in points2
    forward_distance = np.mean(np.min(dists, axis=1))

    # Backward distance: for each point in points2, find distance to nearest point in points1
    backward_distance = np.mean(np.min(dists, axis=0))

    # Symmetric Chamfer distance
    chamfer_dist = (forward_distance + backward_distance) / 2.0

    return chamfer_dist

def compute_image_chamfer_distance(
    image1: np.ndarray | Image.Image,
    image2: np.ndarray | Image.Image,
) -> float:
    """Compute Chamfer distance between two images.

    Args:
        image1: First image
        image2: Second image
        normalize: Whether to normalize distances by image diagonal

    Returns:
        Symmetric Chamfer distance
    """
    # Get edge points from both images
    points1 = get_edge_points(image1)
    points2 = get_edge_points(image2)

    # Handle empty edge cases
    if len(points1) == 0 or len(points2) == 0:
        return float('nan')

    # Compute Chamfer distance
    chamfer = chamfer_distance(points1, points2)
    return chamfer
