import time
import statistics
from typing import List, Optional, Tuple

class RTTDistanceCalculator:
    """
    Calculate distance between WiFi nodes based on Round Trip Time (RTT).
    
    Uses the speed of light in air to convert time measurements to distance.
    Includes calibration and filtering options for improved accuracy.
    """
    
    # Speed of light in meters per second (in air)
    SPEED_OF_LIGHT = 299702547  # m/s
    
    def __init__(self, 
                 processing_delay: float = 0.0,
                 calibration_offset: float = 0.0,
                 medium_factor: float = 1.0):
        """
        Initialize the distance calculator.
        
        Args:
            processing_delay: Known processing delay in seconds (both nodes combined)
            calibration_offset: Distance offset in meters from calibration measurements
            medium_factor: Speed reduction factor for propagation medium (1.0 for air)
        """
        self.processing_delay = processing_delay
        self.calibration_offset = calibration_offset
        self.medium_factor = medium_factor
        self.speed = self.SPEED_OF_LIGHT * medium_factor
    
    def calculate_distance(self, rtt: float) -> float:
        """
        Calculate distance from a single RTT measurement.
        
        Args:
            rtt: Round trip time in seconds
            
        Returns:
            Distance in meters
        """
        # Subtract known processing delays
        actual_rtt = rtt - self.processing_delay
        
        # Distance = (RTT * speed) / 2 (because it's a round trip)
        distance = (actual_rtt * self.speed) / 2.0
        
        # Apply calibration offset
        distance += self.calibration_offset
        
        return max(0.0, distance)  # Distance can't be negative
    
    def calculate_distance_multiple(self, 
                                    rtt_measurements: List[float],
                                    method: str = 'median',
                                    outlier_threshold: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate distance from multiple RTT measurements with filtering.
        
        Args:
            rtt_measurements: List of RTT measurements in seconds
            method: Statistical method - 'mean', 'median', or 'trimmed_mean'
            outlier_threshold: If set, remove measurements beyond this many std devs
            
        Returns:
            Tuple of (distance in meters, standard deviation in meters)
        """
        if not rtt_measurements:
            raise ValueError("No RTT measurements provided")
        
        measurements = rtt_measurements.copy() # Because Python memory management is weird and we don't want to modify the original list
        
        # Remove outliers if threshold is set
        if outlier_threshold and len(measurements) > 3:
            mean_rtt = statistics.mean(measurements)
            std_rtt = statistics.stdev(measurements)
            measurements = [m for m in measurements 
                          if abs(m - mean_rtt) <= outlier_threshold * std_rtt]
        
        if not measurements:
            raise ValueError("All measurements were filtered as outliers")
        
        # Calculate distances for all measurements
        distances = [self.calculate_distance(rtt) for rtt in measurements]
        
        # Apply statistical method
        if method == 'mean':
            distance = statistics.mean(distances)
        elif method == 'median':
            distance = statistics.median(distances)
        elif method == 'trimmed_mean':
            # Remove top and bottom 10%
            sorted_distances = sorted(distances)
            trim_count = len(sorted_distances) // 10
            if trim_count > 0:
                trimmed = sorted_distances[trim_count:-trim_count]
                distance = statistics.mean(trimmed)
            else:
                distance = statistics.mean(distances)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate standard deviation
        std_dev = statistics.stdev(distances) if len(distances) > 1 else 0.0
        
        return distance, std_dev
    
    def measure_rtt(self, send_func, receive_func, payload: bytes = b'PING') -> float:
        """
        Measure RTT by sending a packet and waiting for response.
        
        Args:
            send_func: Function to send packet, signature: send_func(payload)
            receive_func: Function to receive response, signature: receive_func() -> bytes
            payload: Data to send
            
        Returns:
            RTT in seconds
        """
        start_time = time.perf_counter()
        send_func(payload)
        receive_func()
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def calibrate(self, 
                  known_distance: float,
                  rtt_measurements: List[float]) -> None:
        """
        Calibrate the calculator using measurements at a known distance.
        
        This determines the calibration_offset needed to match reality.
        
        Args:
            known_distance: Actual distance between nodes in meters
            rtt_measurements: List of RTT measurements at this distance
        """
        # Calculate what distance we would measure without calibration
        # temp_offset = self.calibration_offset
        self.calibration_offset = 0.0
        
        measured_distance, _ = self.calculate_distance_multiple(rtt_measurements)
        
        # Set offset to correct the error
        self.calibration_offset = known_distance - measured_distance
        
        print(f"Calibration complete:")
        print(f"  Known distance: {known_distance:.2f} m")
        print(f"  Measured distance (uncalibrated): {measured_distance:.2f} m")
        print(f"  Calibration offset: {self.calibration_offset:.2f} m")


# Example use
if __name__ == "__main__":
    # Initialize calculator
    calc = RTTDistanceCalculator()
    
    # Example 1: Single RTT measurement
    rtt = 0.000001  # 1 microsecond RTT
    distance = calc.calculate_distance(rtt)
    print(f"Single measurement: RTT={rtt*1e6:.2f} µs, Distance={distance:.2f} m")
    
    # Example 2: Multiple measurements with filtering
    rtt_samples = [
        0.0000033, 0.0000035, 0.0000034, 0.0000036, 0.0000033,
        0.0000034, 0.0000055, 0.0000033, 0.0000035, 0.0000034
    ]
    
    distance, std_dev = calc.calculate_distance_multiple(
        rtt_samples, 
        method='median',
        outlier_threshold=2.0  # Remove measurements > 2 std devs from mean
    )
    print(f"\nMultiple measurements:")
    print(f"  Distance: {distance:.2f} ± {std_dev:.2f} m")
    
    # Example 3: With calibration
    print("\n--- Calibration Example ---")
    calc_calibrated = RTTDistanceCalculator()
    
    # Simulate measurements at 5m distance in reality, but with some noise
    known_dist = 5.0
    calibration_rtts = [0.0000033 + i*0.0000001 for i in range(10)]
    
    calc_calibrated.calibrate(known_dist, calibration_rtts)
    
    # Now measure unknown distance
    new_measurements = [0.0000067 + i*0.0000001 for i in range(10)] # More noisy measurements
    distance, std_dev = calc_calibrated.calculate_distance_multiple(new_measurements)
    print(f"\nNew measurement with calibration:")
    print(f"  Distance: {distance:.2f} ± {std_dev:.2f} m")