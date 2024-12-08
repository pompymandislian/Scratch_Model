import numpy as np

def _calculate_distance(point_1, point_2, metric="euclidean"):
    """
    Calculate the distance between two points in n-dimensional space using NumPy.
    """

    # Convert the points to numpy arrays
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)

    # Check same dimensionality
    if point_1.shape != point_2.shape:
        raise ValueError("Points must have the same number of dimensions")

    # Calculate distance based on the metric
    if metric == "euclidean":
        # Euclidean distance: sqrt(sum((x2 - x1)^2 for all dimensions))
        return np.linalg.norm(point_2 - point_1)

    elif metric == "manhattan":
        # Manhattan distance: sum(abs(x2 - x1) for all dimensions)
        return np.sum(np.abs(point_2 - point_1))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")

