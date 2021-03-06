# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:32:27 2021

@author: Lukas
"""

import raytracing as rt
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light

# =============================================================================
# Examples
# =============================================================================

def example_01():
    """Reflection of rays at flat surfaces"""
    
    # create three flat mirrors
    m1 = rt.FlatSurface(np.array([50, 0]), np.array([1, 1]), (-5, 5))
    m2 = rt.FlatSurface(np.array([50, 50]), np.array([-1, 1]), (-5, 5))
    m3 = rt.FlatSurface(np.array([0, 50]), np.array([-1, 1]), (-5, 5))
    
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
    fig.canvas.set_window_title('Example 01')
    
    for x in [m1, m2, m3]:
        x.plot(ax, style='k')
    
    for ray in rays:
        ray.plot(ax, style='r-', linewidth=1)
    
    ax.set_xlabel('Width x (mm)')
    ax.set_ylabel('Height y (mm)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    

def example_02():
    """Spherical mirror telescope"""
    
    # tilt angle in rad
    angle = 0.1
    
    # focal length of both mirrors
    f1 = 75.0
    f2 = 25.0
    
    # create spherical mirror 1
    m1 = rt.SphericalSurface(np.array([100., 0.]),
                             np.array([-np.cos(angle), np.sin(angle)]),
                             2*f1,
                             (-0.07, 0.07))
    
    # second spherical mirror, m2
    m2 = rt.SphericalSurface(np.array([100 - (f1+f2)*np.cos(2*angle), (f1+f2)*np.sin(2*angle)]),
                             np.array([np.cos(angle), -np.sin(angle)]),
                             2*f2,
                             (-0.1, 0.1))
    
    # create rays
    rays = []
    nrays = 5
    width = 5.0
    
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
        ray.propagate(distance=100.0)
      
    # plot mirrors and rays
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Example 02')
    
    m1.plot(ax, 'k')
    m2.plot(ax, 'k')
    
    for ray in rays:
        ray.plot(ax, 'r-', linewidth=1)
    
    ax.set_xlabel('Width x (mm)')
    ax.set_ylabel('Height y (mm)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    

def example_03():
    """Focusing of a ray bundle by a parabolic mirror"""
    
    # create parabolic mirror
    f = 50.0            # focal length in mm
    fp = [100, 0.0]     # coordinates of focal point
    n = [-1., 0.0]      # vector pointing from vertex to focal point
    pr = [-1.2, 1.2]    # parameter range (angle range) in rad
    pm = rt.ParabolicSurface(f, fp, n, pr)
    
    # create the ray bundle
    rays = []
    nrays = 10
    width = 80.0

    for ii in range(nrays):
        y0 = -0.5 * width  + ii * width / (nrays - 1)
        ray = rt.Ray(np.array([0.0, y0]), np.array([1.0, 0.0]))
        rays.append(ray)
    
    # propagate rays
    for ray in rays:
        ray.propagate(surface=pm)
        ray.reflect(pm)
        ray.propagate(distance=100.0)
    
    # plot
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Example 03')
    
    pm.plot(ax)
        
    for ray in rays:
        ray.plot(ax, 'r', linewidth=1)
        
    ax.set_xlabel('Width x (mm)')
    ax.set_ylabel('Height y (mm)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


def example_04():
    """Refraction at a air-glass interfaces"""
    
    # create the interface
    surface1 = rt.FlatSurface([50.0, 0.0], [0.0, 1.0], [-100, 200])
    surface2 = rt.FlatSurface([400.0, 0.0], [0.0, 1.0], [-100, 200])
    
    # create materials
    air = rt.Material.fromName('Air')
    bk7 = rt.Material.fromName('BK7')
    
    # create rays
    wvl_min = 500
    wvl_max = 700
    n_rays = 21
    wvl = np.linspace(wvl_min, wvl_max, n_rays)
    
    rays = []
    for ii in range(n_rays):
        freq = speed_of_light / wvl[ii] * 1e-3
        rays.append(rt.Ray([0.0,-50.0], [1.0, 0.8], freq=freq))
    
    # propagate rays
    for ray in rays:
        ray.propagate(surface=surface1)
        ray.refract(surface1, air, bk7)
        ray.propagate(surface=surface2)
        ray.refract(surface2, bk7, air)
        ray.propagate(distance=50)
    
    # plot
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Example 04')
    
    surface1.plot(ax, 'k')
    surface2.plot(ax, 'k')
    
    for ray in rays:
        cc = rt.wavelength_to_color(ray.wvl)
        ray.plot(ax, style='-', color=cc)
    
    ax.set_xlabel('Width x (mm)')
    ax.set_ylabel('Height y (mm)')
    ax.set_aspect('equal')
    
    ax.set_ylim(144, 152)
    ax.set_xlim(396, 404)
    
    ax.text(400-.1, 146, 'Glass', ha='right')
    ax.text(400+.1, 146, 'Air', ha='left')
    
    plt.tight_layout()
    plt.show()

    
    
# =============================================================================
# Run example 
# =============================================================================

# example_01()  # Flat mirrors
# example_02()  # Spherical mirror
# example_03()  # Parabolic mirror
example_04()  # Refraction at air-glass interface


