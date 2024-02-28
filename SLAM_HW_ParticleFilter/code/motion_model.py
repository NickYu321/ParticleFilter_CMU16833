''' 
    Using odometry model to update the location (x, y) and orientation (theta)
    Four parameters to tune, bascially larger represent larger randomness of particle moving

    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
    
    finished by Tianqi Yu (tianqiyu@andrew.cmu.edu), 2024
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.000012
        self._alpha2 = 0.000012
        self._alpha3 = 0.000012
        self._alpha4 = 0.000012


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        Based on Table 5.6
        """
        x_bar,y_bar,theta_bar = u_t0
        x_bar1,y_bar1,theta_bar1 = u_t1
        x,y,theta = x_t0

        delta_rot1 = math.atan2(y_bar1-y_bar,x_bar1-x_bar)-theta_bar
        delta_trans = math.sqrt((y_bar1-y_bar)**2+(x_bar1-x_bar)**2)
        delta_rot2 = theta_bar1-theta_bar-delta_rot1
        
        sample_rot1 = self._alpha1*(delta_rot1**2) + self._alpha2*(delta_trans**2)
        sample_trans = self._alpha3*(delta_trans**2) + self._alpha4*(delta_rot1**2) + self._alpha4*(delta_rot2**2)
        sample_rot2 = self._alpha1*(delta_rot2**2) + self._alpha2*(delta_trans**2)

        delta_rot1_hat = delta_rot1- np.random.normal(0,math.sqrt(sample_rot1))
        delta_trans_hat = delta_trans- np.random.normal(0,math.sqrt(sample_trans))
        delta_rot2_hat = delta_rot2 - np.random.normal(0,math.sqrt(sample_rot2))
        
        xp = x + delta_trans_hat*math.cos(theta+delta_rot1_hat)
        yp = y + delta_trans_hat*math.sin(theta+delta_rot1_hat)
        thetap = theta + delta_rot1_hat + delta_rot2_hat

        return [xp, yp, thetap]
