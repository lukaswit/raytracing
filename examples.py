# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:32:27 2021

@author: Lukas
"""

import raytracing as rt
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Examples
# =============================================================================

def example_01():
    """Reflection of rays at flat surfaces"""
    
    # create three flat mirrors
    m1 = rt.FlatSurface(np.array([50, 0]), np.array([1, 1]), (-5, 5))
    m2 = rt.FlatSurface(np.array([50, 50]), np.array([-1, 1]), (-5, 5))
    m3 = rt.FlatSurface(np.array([0, 50]), np.array([-1, 1]), (-5, 5))
    
    x1, y1 = m1.get_contour()
    x2, y2 = m2.get_contour()
    x3, y3 = m3.get_contour()

    # create rays
    rays = []
    nrays = 3
    width = 4.0
    
    for ii in range(nrays):
        y0 = -0.5 * width  + ii * width / (nrays - 1)
        ray = rt.Ray(np.array([0.0, y0]), np.array([1.0, 0.0]))
        rays.append(ray)
        
    # propagate rays
    for ray in rays:
        ray.propagate(surface=m1)
        ray.reflect(m1)
        ray.propagate(surface=m2)
        ray.reflect(m2)
        ray.propagate(surface=m3)
        ray.reflect(m3)
        ray.propagate(distance=20.0)

    # plot
    fig, ax = plt.subplots()
    
    ax.plot(x1, y1, 'k')
    ax.plot(x2, y2, 'k')
    ax.plot(x3, y3, 'k')
    
    for ray in rays:
        x, y = ray.get_positions()
        ax.plot(x, y, 'r')
    
    ax.set_xlabel('Width x (mm)')
    ax.set_ylabel('Height y (mm)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
# =============================================================================
# Run example 
# =============================================================================

example_01()