'''
    Using laser range finder sensor and raycast to get distance from sensor to objects
    use get_probability calculate possibility p for each raycast beam
    multiply all beams p for a particle's weight

    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021

    finished by Tianqi Yu (tianqiyu@andrew.cmu.edu), 2024
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        #self._z_hit = 1
        #self._z_short = 0.1
        #self._z_max = 0.1
        #self._z_rand = 100
        self._z_hit = 5   #too large will be into the wall, allow too much error
        self._z_short = 0.12
        self._z_max = 1
        self._z_rand = 900 #too small can't converge
        self._z_sum = self._z_hit+self._z_short+self._z_max+self._z_rand
        self._z_hit = self._z_hit/self._z_sum
        self._z_short = self._z_short/self._z_sum
        self._z_max = self._z_max/self._z_sum
        self._z_rand = self._z_rand/self._z_sum
        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 900

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 15
        #raycast step length
        self._step = 10
        self.occupancy_map = occupancy_map
        
        #test raycast
        #self.raycast_endpoints = []

    def get_probability(self, distance_l, distance_r):
        # P_hit
        if 0 <= distance_l <= self._max_range:
            pHit = np.exp(-1 / 2 * (distance_l - distance_r) ** 2 / (self._sigma_hit ** 2))
            pHit = pHit / (np.sqrt(2 * np.pi))/self._sigma_hit

        else:
            pHit = 0

        # P_short
        if 0 <= distance_l <= distance_r:
            eta = 1
            pShort = eta * self._lambda_short * np.exp(-self._lambda_short * distance_l)

        else:
            pShort = 0

        # P_max
        if distance_l >= self._max_range:
            pMax = self._max_range
        else:
            pMax = 0

        # P_rand
        if 0 <= distance_l < self._max_range:
            pRand = 1 / self._max_range
        else:
            pRand = 0

        p = self._z_hit * pHit + self._z_short * pShort + self._z_max * pMax + self._z_rand * pRand
        return p
    
    def raycast_distance(self,x_l,y_l,theta_l):
        x_end = x_l
        y_end = y_l
        #tune step here
        delta_x = self._step*np.cos(theta_l)
        delta_y = self._step*np.sin(theta_l)

        while 0 <= x_end < 8000 and 0 <= y_end <8000:
            x_idx = int(np.floor(x_end/10))
            y_idx = int(np.floor(y_end/10))
            #self.occupancy_map[y_idx, x_idx] < self._min_probability) & (self.occupancy_map[y_idx, x_idx] >= 0)
            if (self.occupancy_map[y_idx, x_idx] > self._min_probability):
                break  # Exit the loop if occupied
            x_end += delta_x
            y_end += delta_y

            #test raycast
            #self.raycast_endpoints.append((x_end,y_end))
        #distance traveled by the laser beam
        distance = math.sqrt((x_l - x_end) ** 2 + (x_l - y_end) ** 2)
        return distance


    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        #test raycast, clean at every iteration
        #self.raycast_endpoints = []
        x,y,theta = x_t1
        #print(theta)
        # length all in cm, but the map have 10cm grid resolution
        x_l = x + 25 * math.cos(theta)
        y_l = y + 25 * math.sin(theta)
        prob_zt1 = 1.0

        for angle in range (-90,90,self._subsampling): #every n degrees
            # angle and distance from lader
            theta_l = (angle + theta * 180 / math.pi ) * (math.pi/180)
            distance_l = z_t1_arr[angle+90]
            # calculate raycast distance
            distance_r = self.raycast_distance(x_l,y_l,theta_l)
            p = self.get_probability(distance_l, distance_r)
            prob_zt1 *= p
        tem_idx_y = int(np.floor(y/10))
        tem_idx_x = int(np.floor(x/10))
        #print(tem_idx_y)
        #print(tem_idx_x)
        #print(self.occupancy_map[tem_idx_y,tem_idx_x])
        if (self.occupancy_map[tem_idx_y,tem_idx_x] > self._min_probability):
            prob_zt1 = 0
        return prob_zt1
    
    

