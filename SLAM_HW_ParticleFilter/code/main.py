'''
    when time index < 200, use low variance filter to make all particles into a high possibility area
    then reduce particles to 30, which is a reasonable number for both calculation speed and robustness
    
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021

    Finished by Tianqi Yu (tianqiyu@andrew.cmu.edu), 2024
'''

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


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])
'''
def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o',s= 10)
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()
'''
def visualize_timestep(X_bar, tstep, output_path, running_time, time_stamp):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o', s=10)
    #plt.text(0, 1, f'Running Time: {running_time:.2f}s\nTime Stamp: {time_stamp:.2f}s', transform=scat.axes.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    if hasattr(visualize_timestep, 'text'):
        visualize_timestep.text.set_text(f'Running Time: {running_time:.2f}s\nTime Stamp: {time_stamp:.2f}s\nParticles: {len(X_bar)}')
    else:
        visualize_timestep.text = plt.text(0, 1, f'Running Time: {running_time:.2f}s\nTime Stamp: {time_stamp:.2f}s\nParticles: {len(X_bar)}', transform=scat.axes.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()




def visualize_raycast(occupancy_map, raycast_endpoints):
    plt.figure()
    plt.imshow(occupancy_map, cmap='Greys')

    # Plot raycast endpoints
    for x_end, y_end in raycast_endpoints:
        plt.plot(x_end / 10, y_end / 10, 'bo',markersize=1)  # Convert to map coordinates and plot

    plt.show()


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    y_free, x_free = np.where(occupancy_map == 0)

    sizex = len(x_free)
    index = np.random.choice(sizex-1, num_particles)

    x0 = x_free[index] * 10
    y0 = y_free[index] * 10
    x0 = x0.reshape(num_particles, 1)
    y0 = y0.reshape(num_particles, 1)
    theta0 = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    w0 = np.ones((num_particles, 1), dtype=np.float64)
    w0 = w0 / num_particles
    X_bar_init = np.hstack((x0, y0, theta0, w0))
    return X_bar_init


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    #parser.add_argument('--path_to_map', default='C:\\Users\\ROG\\Desktop\\SLAM_HW_ParticleFilter\\data\\map\\wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata5.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    #X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    start_time = time.time()
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]
        if meas_type == "O":
            continue
        if meas_type == "L":
            # convert measurement values from string to double
            meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

            # odometry reading [x, y, theta] in odometry frame
            odometry_robot = meas_vals[0:3]
            time_stamp = meas_vals[-1]

            # ignore pure odometry measurements for (faster debugging)
            # if ((time_stamp <= 0.0) | (meas_type == "O")):
            #     continue

            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

            print("Processing time step {} at time {}s".format(
                time_idx, time_stamp))

            if first_time_idx:
                u_t0 = odometry_robot
                first_time_idx = False
                continue

            X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
            u_t1 = odometry_robot

            # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
            # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
            for m in range(0, num_particles):
                """
                MOTION MODEL
                """
                x_t0 = X_bar[m, 0:3]
                x_t1 = motion_model.update(u_t0, u_t1, x_t0)

                """
                SENSOR MODEL
                """
                
                if (meas_type == "L"):
                    z_t = ranges
                    w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                    #print(w_t)
                    X_bar_new[m, :] = np.hstack((x_t1, w_t))
                else:
                    X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))
                
            X_bar = X_bar_new
            u_t0 = u_t1
            """
            RESAMPLING
            """
            #X_bar = resampler.low_variance_sampler(X_bar)
            
            if 200 < time_idx and len(X_bar)>30:
                X_bar = resampler.reduce_filter(X_bar)
                num_particles = len(X_bar)
                print(len(X_bar))
            else:
                X_bar = resampler.low_variance_sampler(X_bar)

            if args.visualize:
                #visualize_timestep(X_bar, time_idx, args.output)
                visualize_timestep(X_bar, time_idx, args.output, time.time() - start_time, time_stamp)
