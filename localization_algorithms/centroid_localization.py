import numpy as np

def centroid_localization(anchors, weights=None):
    """
    Estimate position using simple centroid (center of mass) calculation.
    
    Parameters:
    -----------
    anchors : list of tuples or numpy array
        Known positions of anchor points [(x1, y1), (x2, y2), ...]
        Can be 2D or 3D coordinates
    weights : list or numpy array, optional
        Weights for each anchor (e.g., signal strength, inverse distance)
        If None, all anchors are weighted equally
    
    Returns:
    --------
    numpy array
        Estimated position as weighted centroid
    """
    anchors = np.array(anchors)
    
    if weights is None:
        # Simple average
        return np.mean(anchors, axis=0)
    
    weights = np.array(weights)
    
    if len(anchors) != len(weights):
        raise ValueError("Number of anchors must match number of weights")
    
    if np.sum(weights) == 0:
        raise ValueError("Sum of weights cannot be zero")
    
    # Weighted average
    return np.average(anchors, axis=0, weights=weights)


def weighted_centroid_rssi(anchors, rssi_values, path_loss_exponent=2.0):
    """
    Centroid localization using RSSI (Received Signal Strength Indicator) as weights.
    Stronger signals contribute more to the position estimate.
    
    Parameters:
    -----------
    anchors : list of tuples or numpy array
        Known positions of anchor points
    rssi_values : list or numpy array
        RSSI values (in dBm) from each anchor
    path_loss_exponent : float
        Path loss exponent for signal propagation model
    
    Returns:
    --------
    numpy array
        Estimated position
    """
    rssi_values = np.array(rssi_values)
    
    # Convert RSSI to linear scale and use as weights
    # Higher RSSI = closer anchor = higher weight
    weights = 10 ** (rssi_values / 10)
    
    return centroid_localization(anchors, weights)


def weighted_centroid_distance(anchors, distances):
    """
    Centroid localization using inverse distance as weights.
    Closer anchors contribute more to the position estimate.
    
    Parameters:
    -----------
    anchors : list of tuples or numpy array
        Known positions of anchor points
    distances : list or numpy array
        Estimated distances from target to each anchor
    
    Returns:
    --------
    numpy array
        Estimated position
    """
    distances = np.array(distances)
    
    # Use inverse distance as weight (avoid division by zero)
    epsilon = 1e-10
    weights = 1.0 / (distances + epsilon)
    
    return centroid_localization(anchors, weights)


def proximity_based_centroid(anchors, in_range_flags):
    """
    Simple proximity-based centroid using only anchors within range.
    Binary decision: either an anchor is in range or not.
    
    Parameters:
    -----------
    anchors : list of tuples or numpy array
        Known positions of anchor points
    in_range_flags : list or numpy array
        Boolean flags indicating if each anchor is in communication range
    
    Returns:
    --------
    numpy array
        Estimated position as average of in-range anchors
    """
    anchors = np.array(anchors)
    in_range_flags = np.array(in_range_flags, dtype=bool)
    
    in_range_anchors = anchors[in_range_flags]
    
    if len(in_range_anchors) == 0:
        raise ValueError("No anchors in range")
    
    return np.mean(in_range_anchors, axis=0)


# Example usage
if __name__ == "__main__":
    print("=== Centroid Localization Examples ===\n")
    
    # Define anchor positions
    anchors = [
        (0, 0),
        (10, 0),
        (10, 10),
        (0, 10)
    ]
    
    # True position (for comparison)
    true_pos = np.array([3, 4])
    
    # 1. Simple unweighted centroid
    print("1. Simple Centroid (unweighted):")
    simple_centroid = centroid_localization(anchors)
    print(f"   Estimated position: {simple_centroid}")
    print(f"   Error: {np.linalg.norm(simple_centroid - true_pos):.4f}\n")
    
    # 2. Distance-weighted centroid
    print("2. Distance-Weighted Centroid:")
    true_distances = [np.linalg.norm(true_pos - np.array(a)) for a in anchors]
    noise = np.random.normal(0, 0.2, len(true_distances))
    measured_distances = np.array(true_distances) + noise
    
    dist_weighted = weighted_centroid_distance(anchors, measured_distances)
    print(f"   Measured distances: {measured_distances}")
    print(f"   Estimated position: {dist_weighted}")
    print(f"   Error: {np.linalg.norm(dist_weighted - true_pos):.4f}\n")
    
    # 3. RSSI-weighted centroid
    print("3. RSSI-Weighted Centroid:")
    # Simulate RSSI values (stronger signal = higher RSSI = closer)
    # RSSI typically ranges from -30 (close) to -90 (far) dBm
    rssi_values = [-40, -65, -70, -55]  # Simulated values
    
    rssi_weighted = weighted_centroid_rssi(anchors, rssi_values)
    print(f"   RSSI values: {rssi_values} dBm")
    print(f"   Estimated position: {rssi_weighted}")
    print(f"   Error: {np.linalg.norm(rssi_weighted - true_pos):.4f}\n")
    
    # 4. Proximity-based centroid
    print("4. Proximity-Based Centroid:")
    # Only first 3 anchors in range
    in_range = [True, True, True, False]
    
    proximity_centroid = proximity_based_centroid(anchors, in_range)
    print(f"   In-range anchors: {[anchors[i] for i, flag in enumerate(in_range) if flag]}")
    print(f"   Estimated position: {proximity_centroid}")
    print(f"   Error: {np.linalg.norm(proximity_centroid - true_pos):.4f}\n")
    
    # 5. Custom weights
    print("5. Custom Weighted Centroid:")
    custom_weights = [0.5, 1.0, 0.8, 0.3]  # Arbitrary weights
    
    custom_weighted = centroid_localization(anchors, custom_weights)
    print(f"   Weights: {custom_weights}")
    print(f"   Estimated position: {custom_weighted}")
    print(f"   Error: {np.linalg.norm(custom_weighted - true_pos):.4f}\n")
    
    # 3D Example
    print("=== 3D Centroid Localization ===\n")
    anchors_3d = [
        (0, 0, 0),
        (10, 0, 0),
        (0, 10, 0),
        (0, 0, 10)
    ]
    
    true_pos_3d = np.array([3, 4, 5])
    distances_3d = [np.linalg.norm(true_pos_3d - np.array(a)) for a in anchors_3d]
    
    centroid_3d = weighted_centroid_distance(anchors_3d, distances_3d)
    print(f"True position: {true_pos_3d}")
    print(f"Estimated position: {centroid_3d}")
    print(f"Error: {np.linalg.norm(centroid_3d - true_pos_3d):.4f}")