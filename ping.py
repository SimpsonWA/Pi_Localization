import os
import time
import math
import subprocess

# Function to ping a given IP and return the round-trip time (RTT)
def ping_ip(ip, count=4):
    try:
        # Ping the IP address and get the result
        response = subprocess.run(['ping', '-c', str(count), ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if response.returncode == 0:
            # Extract the RTT (Round Trip Time) from the ping output
            output = response.stdout.decode()
            rtt_line = [line for line in output.splitlines() if "avg" in line]
            if rtt_line:
                rtt = rtt_line[0].split('/')[-3]  # Get the average RTT in ms
                return float(rtt)  # Return the RTT in milliseconds
            else:
                return None
        else:
            print(f"Ping to {ip} failed.")
            return None
    except Exception as e:
        print(f"Error pinging {ip}: {e}")
        return None

# Function to estimate distance based on RTT
def estimate_distance_from_rtt(rtt, rtt_at_1m=1.0, n=2.0):
    """
    :param rtt: Round Trip Time (RTT) in ms
    :param rtt_at_1m: Expected RTT at 1 meter (default is 1 ms for ideal conditions)
    :param n: Path loss exponent (default is 2.0 for free space)
    :return: Estimated distance in meters
    """
    try:
        if rtt is None:
            return None
        # Use RTT and modify the log-distance model to estimate distance
        distance = (rtt / rtt_at_1m) ** (1 / n)
        return distance
    except Exception as e:
        print(f"Error estimating distance: {e}")
        return None

# Function to estimate distance based on Batman TQ
def estimate_distance_from_tq(tq, tq_at_1m=255.0, n=2.0):
    """
    :param tq: Transmission Quality (TQ) value from Batman-adv (typically 0–255)
    :param tq_at_1m: Expected TQ at 1 meter distance (default is 255 for ideal link)
    :param n: Path loss exponent controlling how quickly signal quality drops (default is 2.0)
    :return: Estimated distance in meters (float)
    """
    try:
        if tq is None or tq <= 0:
            return None
        # Use inverse power model — as TQ decreases, distance increases
        distance = (tq_at_1m / tq) ** n
        return distance
    except Exception as e:
        print(f"Error estimating distance: {e}")
        return None

# Main code
def main():
    ips = {
        'Pi2': '192.168.200.2',  # IP address of Pi 2
        'Pi3': '192.168.200.3',  # IP address of Pi 3
        'Pi4': '192.168.200.4',  # IP address of Pi 4
    }

    # Expected RTT for 1 meter
    rtt_at_1m = 20.0  # Expected RTT at 1 meter (in ms)

    print("Starting RTT-based distance estimation...")

    for pi, ip in ips.items():
        print(f"\nPinging {pi} at IP {ip}...")
        rtt = ping_ip(ip)

        if rtt is not None:
            print(f"Round Trip Time (RTT) to {pi}: {rtt:.2f} ms")
            distance = estimate_distance_from_rtt(rtt, rtt_at_1m)
            if distance is not None:
                print(f"Estimated distance from Pi 1 to {pi}: {distance:.2f} meters")
            else:
                print(f"Unable to estimate distance for {pi}.")
        else:
            print(f"Could not get RTT for {pi}.")

if __name__ == '__main__':
    main()


