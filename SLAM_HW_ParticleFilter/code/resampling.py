'''
    resample particles
    low variance sampler move particles into a more possiable area
    reduce_filter reduce less weighted particles to speed up 
    
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021

    finished by Tianqi Yu (tianqiyu@andrew.cmu.edu), 2024
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled


    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        
        X_bar_new =  list()
        M = X_bar.shape[0]
        X_bar[:,3] = X_bar[:,3] / np.sum(X_bar[:,3])
        r = np.random.uniform(0,1.0/M)
        c = X_bar[0,3]
        i = 0
        for m in range(1,M+1):
            u = r + (m-1)*1/M
            while u > c:
                i+=1
                c = c + X_bar[i,3]
            X_bar_new.append(X_bar[i])
        X_bar_new = np.array(X_bar_new)
        return X_bar_new
    
    def reduce_filter(self, X_bar, min_weight_threshold=1e-6):

        X_bar[:, 3] = X_bar[:, 3] / np.sum(X_bar[:, 3])

        # Sort on weights
        sorted_indices = np.argsort(X_bar[:, 3])
        sorted_particles = X_bar[sorted_indices]

        cumulative_weights = np.cumsum(sorted_particles[:, 3])

        # threshold index to discarded
        threshold_index = np.searchsorted(cumulative_weights, min_weight_threshold)

        # Resample particles with weights above the threshold
        #selected_indices = sorted_indices[threshold_index:]
        selected_particles = sorted_particles[threshold_index:]

        #new weights
        num_selected_particles = len(selected_particles)
        selected_particles[:, 3] = 1.0 / num_selected_particles

        # Resample with replacement
        resampled_indices = np.random.choice(num_selected_particles, size=num_selected_particles, replace=True)
        X_bar_resampled = selected_particles[resampled_indices]

        return X_bar_resampled