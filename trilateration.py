import numpy as np
from scipy.optimize import least_squares

def trilateration(anchors, distances):
    """
    Estimate position using trilateration with least squares optimization.
    
    Parameters:
    -----------
    anchors : list of tuples or numpy array
        Known positions of anchor points [(x1, y1), (x2, y2), (x3, y3), ...]
        Can be 2D or 3D coordinates
    distances : list or numpy array
        Measured distances from target to each anchor [d1, d2, d3, ...]
    
    Returns:
    --------
    numpy array
        Estimated position [x, y] or [x, y, z]
    dict
        Additional information including success status and residual error
    """
    anchors = np.array(anchors)
    distances = np.array(distances)
    
    if len(anchors) != len(distances):
        raise ValueError("Number of anchors must match number of distances")
    
    if len(anchors) < 3:
        raise ValueError("At least 3 anchors required for trilateration")
    
    # Determine dimensionality
    n_dim = anchors.shape[1]
    
    # Initial guess: centroid of anchors
    x0 = np.mean(anchors, axis=0)
    
    # Residual function
    def residuals(pos):
        return np.sqrt(np.sum((anchors - pos)**2, axis=1)) - distances
    
    # Solve using least squares
    result = least_squares(residuals, x0, method='lm')
    
    return result.x, {
        'success': result.success,
        'residual': np.linalg.norm(result.fun),
        'message': result.message,
        'n_iterations': result.nfev
    }


def trilateration_2d_analytical(anchors, distances):
    """
    Analytical trilateration for exactly 3 anchors in 2D.
    Uses geometric solution without optimization.
    
    Parameters:
    -----------
    anchors : list of tuples or numpy array
        Exactly 3 anchor positions [(x1, y1), (x2, y2), (x3, y3)]
    distances : list or numpy array
        Distances from target to each anchor [d1, d2, d3]
    
    Returns:
    --------
    numpy array
        Estimated position [x, y]
    """
    if len(anchors) != 3 or len(distances) != 3:
        raise ValueError("Analytical 2D trilateration requires exactly 3 anchors")
    
    (x1, y1), (x2, y2), (x3, y3) = anchors
    r1, r2, r3 = distances
    
    # Transform to coordinate system where anchor1 is at origin
    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2
    
    D = 2 * (x3 - x1)
    E = 2 * (y3 - y1)
    F = r1**2 - r3**2 - x1**2 + x3**2 - y1**2 + y3**2
    
    # Solve system of equations
    denom = A * E - B * D
    
    if abs(denom) < 1e-10:
        raise ValueError("Anchors are collinear, cannot solve")
    
    x = (C * E - F * B) / denom
    y = (A * F - D * C) / denom
    
    return np.array([x, y])


# Example usage
if __name__ == "__main__":
    print("=== Trilateration Example ===\n")
    
    # 2D Example with 4 anchors
    anchors_2d = [
        (0, 0),
        (10, 0),
        (10, 10),
        (0, 10)
    ]
    
    # True position
    true_pos = np.array([3, 4])
    
    # Calculate distances with some noise
    true_distances = [np.linalg.norm(true_pos - np.array(a)) for a in anchors_2d]
    noise = np.random.normal(0, 0.1, len(true_distances))
    measured_distances = true_distances + noise
    
    # Estimate position
    estimated_pos, info = trilateration(anchors_2d, measured_distances)
    
    print(f"True position: {true_pos}")
    print(f"Estimated position: {estimated_pos}")
    print(f"Error: {np.linalg.norm(estimated_pos - true_pos):.4f}")
    print(f"Optimization successful: {info['success']}")
    print(f"Residual: {info['residual']:.4f}\n")
    
    # 3D Example
    print("=== 3D Trilateration ===\n")
    anchors_3d = [
        (0, 0, 0),
        (10, 0, 0),
        (0, 10, 0),
        (0, 0, 10)
    ]
    
    true_pos_3d = np.array([3, 4, 5])
    true_distances_3d = [np.linalg.norm(true_pos_3d - np.array(a)) for a in anchors_3d]
    
    estimated_pos_3d, info_3d = trilateration(anchors_3d, true_distances_3d)
    
    print(f"True position: {true_pos_3d}")
    print(f"Estimated position: {estimated_pos_3d}")
    print(f"Error: {np.linalg.norm(estimated_pos_3d - true_pos_3d):.6f}")