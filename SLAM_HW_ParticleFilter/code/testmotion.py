import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

# Load the occupancy map
map_obj = MapReader('../data/map/wean.dat')
occupancy_map = map_obj.get_map()

# Load the odometry readings from the log file
logfile = open('../data/log/robotdata3.log', 'r')

# Initialize the particle's initial position and trajectory
initial_pose = [40, 40, 100]  # Initial pose of the particle [x, y, theta]
trajectory = [initial_pose]  # List to store the trajectory of the particle

# Create a MotionModel object
motion_model = MotionModel()

# Parse the odometry readings and simulate the particle's movement
for line in logfile:
    meas_type = line[0]
    meas_vals = np.fromstring(line[1:], dtype=np.float64, sep=' ')

    if meas_type == "O":
        odometry_robot = meas_vals[0:3]
        print(odometry_robot[0])
        print(odometry_robot[1])
        print(odometry_robot[2])

        # Simulate particle movement using the motion model
        new_pose = motion_model.update(odometry_robot[0],odometry_robot[1],odometry_robot[2])
        trajectory.append(new_pose)  # Store the new pose
        initial_pose = new_pose  # Update the initial pose for the next iteration

# Convert trajectory to centimeters for plotting
trajectory_cm = np.array(trajectory) /10  # Map resolution is 10 cm

# Plot the trajectory on the occupancy map
plt.imshow(occupancy_map, cmap='gray')
plt.plot(trajectory_cm[:, 0], trajectory_cm[:, 1], 'r-')
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Trajectory of the Particle')
plt.show()