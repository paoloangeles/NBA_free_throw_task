# -*- coding: utf-8 -*-
"""

@author: Paolo
"""

## IMPORT APPROPRIATE MODULES
import numpy as np
from scipy.stats import gamma
import time


## 2) CODING AND APPLIED STATISTICS ##
### Drawn random samples from a given distribution

## Uniform random variables
def Draw_samples_from_Uni(lower_limit, upper_limit, size):
    ## constants based on ANSI C implementation (Saucier, 2000)
    mod = 2**32
    multiplier = 1103515245
    increment = 12345
    U = np.zeros(size) ## setup initial numpy vector to reserve memory space
    seed = round(time.time()) ## set seed according to current time
    for i in range(size):
        seed = (multiplier*seed + increment)%mod
        U[i] = lower_limit + (upper_limit - lower_limit) * seed/mod
    return U

## Draw samples from given distribution 
def Draw_samples_from_dist(num_samples, dist, **params):
    uniform_samps = Draw_samples_from_Uni(0, 1, num_samples)
    
    ## Using inverse transform sampling and the Box-Muller transform
    if dist == "Normal":
        
        if "mean" in params:
            mean = params["mean"]
        else:
            print("Please specify 'mean' parameter.")
            return
        if "std" in params:
            std = params["std"]
        else:
            print("Please specify 'std' parameter.")
            return
        
        uniform_samps_pairs = Draw_samples_from_Uni(0, 1, num_samples)
        z = np.zeros(num_samples)
        for i in range(num_samples):
            z0 = np.sqrt(-2*np.log(uniform_samps[i]))*np.cos(2*np.pi*uniform_samps_pairs[i])
            # z1 = np.sqrt(-2*np.log(uniform_samps[i]))*np.sin(2*np.pi*uniform_samps_pairs[i]) ## however we can just use z0, we don't need pairs
            z[i] = z0*std + mean
        return z
    
    elif dist == "Exponential":
        
        if "lamb" in params:
            lamb = params["lamb"]
        else:
            print("Please specify 'lamb' parameter.")
            return
        
        x = np.zeros(num_samples)
        for i in range(num_samples):
            x[i] = -(1/lamb) * (np.log(1 - uniform_samps[i])) ## take the inverse Exponential CDF and apply inverse transform sampling
        return x
    
    elif dist == "Gamma":
        
        if "shape" in params:
            shape = params["shape"]
        else:
            print("Please specify 'shape' parameter.")
            return
        if "loc" in params:
            loc = params["loc"]
        else:
            print("Please specify 'loc' parameter.")
            return
        if "scale" in params:
            scale = params["scale"]
        else:
            print("Please specify 'scale' parameter.")
            return
        
        g = np.zeros(num_samples)
        for i in range(num_samples):
            g[i] = gamma.ppf(uniform_samps[i], shape, loc, scale)
        return g            
    else:
        print("Please input a correct distribution. Type either 'Normal', 'Exponential' or 'Gamma'")
    
    
norm_samps = Draw_samples_from_dist(100, "Normal", mean = 0, std = 1)
expon_samps = Draw_samples_from_dist(100, "Exponential", lamb = 1)
gamma_samps = Draw_samples_from_dist(100, "Gamma", shape = 1, loc = 1, scale = 1)