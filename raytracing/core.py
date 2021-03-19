# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 20:46:27 2021

@author: Lukas
"""

# imports
import numpy as np
import scipy.optimize as opt
import numpy.linalg as lina
from scipy.constants import speed_of_light
from .materials import Material
import matplotlib.pyplot as plt
import sys


# =============================================================================
# Some helper functions (could be migrated to additional file later)
# =============================================================================

def rotation_matrix_2d(angle):
    """Transformation matrix (in two dimensions) for rotation by a
    given angle.
    
    Parameters
    ----------
    angle : float
        Rotation angle

    Returns
    -------
    m : array (2, 2)
        Transformation matrix
    """
    
    m = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return m


def wavelength_to_color(wavelength, gamma=0.8):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    
    Modified to return a six digit color code instead of an RGB tuple.
    
    Example:
    
    import matplotlib.pyplot as plt
    
    x = ...
    y = ...
    
    cc = wavelength_to_color(633.0)
    ax.plot(x, y, '-', color=cc)
    
    
    Parameters
    ----------
    wavelength : float
        Wavelength in nm

    Returns
    -------
    color : str
        six digit color code
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R = int(255 * R)
    G = int(255 * G)
    B = int(255 * B)
    
    def clamp(x): 
        return max(0, min(x, 255))

    color = "#{0:02x}{1:02x}{2:02x}".format(clamp(R), clamp(G), clamp(B))
    return color


def fix_edges(y, n):
    """
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    y : TYPE
        DESCRIPTION.

    """
    
    for ii in range(n):
        y[n-ii-1] = 2*y[n-ii] - y[n-ii+1]
        y[-n+ii] = 2*y[-n+ii-1] - y[-n+ii-2]
    return y


def minimum_deviation(n_p, n_e, apex_angle):
    """Calculates the incidence angle resulting in minimum deviation
    by a prism with given refractive index and apex angle.    

    Parameters
    ----------
    n_p : float
        Refractive index of prism
    n_e : float
        Refractive index of environment
    apex_angle : float
        Apex angle in rad

    Returns
    -------
    angle_in : float
        incidence angle in rad

    """
    
    angle_in = np.arcsin(n_p / n_e * np.sin(0.5 * apex_angle))
    return angle_in


# =============================================================================
# Ray tracing classes
# =============================================================================

class Ray(object):
    """The ray class representing a single ray of light. The current state
    of the ray is represented by its position and direction in the 
    two-dimensional simulation space.
    
    Parameters
    ----------
    position : array of float
        vector specifying the initial ray position, with position[0] being 
        the x- and position[1] the y-component in units of mm
    direction : array of float 
        vector specifying the initial direction of the ray, with direction[0] 
        being the x- and direction[1] being the y-component
    freq : float
        optical frequency in THz
    
    Attributes
    ----------
    wvl : float
        ray wavelength in nm
    positions : list 
        list of ray positions
    optpath : list 
        list containing the optical path between consecutive ray
        positions
    phase : list 
        list containing the optical phase accumulated between two 
        consecutive ray positions
    terminated : bool 
        True if propagation has failed (intersection with a surface could 
        not be found)
    phase_acc : float
        accumulated optical phase
    optpath_acc : float
        accumulated optical path
    """
    
    
    def __init__(self, position, direction, freq=500.0):
        
        self.position = np.array(position)
        self.direction = np.array(direction) / lina.norm(np.array(direction))
        self.freq = freq
        self.wvl = speed_of_light / freq * 1e-3  # in nm
        self.positions = [position]
        self.optpath = []
        self.phase = []
        self.phase_acc = 0.0
        self.optpath_acc = 0.0
        self.terminated = False
        self.p = 0.0
        

    def normalize_direction(self):
        """Normalizes the current ray direction (to length 1)"""
        
        self.direction = self.direction / np.linalg.norm(self.direction)


    def propagate(self, surface=None, 
                  distance=0.0, 
                  material=Material.fromName('Vacuum')):
        """Propagates the ray to a specified surface, or for a given distance,
        with a given material determining the refractive index.
        
        Parameters
        ----------
        surface : Surface instance, optional
            Surface to propagate to. The default is None, in which case the
            ray will simply be propagated for a specified distance
        distance : float, optional
            Propagation distance in mm. The default is 0.0. Only relevant if
            surface is None
        material : Material instance, optional
            The material the ray is propagating in. 
            The default is Material.fromName('Vacuum').
        """
        
        self.normalize_direction()
        
        if self.terminated:
            # nothing happens because this ray has been terminated
            print('Ray will not be propagated since it has been terminated.')
        
        elif surface is None:
            # free space propagation
            
            # update ray position
            self.position = self.position + distance * self.direction
            self.positions.append(self.position)

        else:
            # propagation to surface
            
            # ------------------------------------------
            # pre-conditioning before final optimization
            # ------------------------------------------
            
            # define error function for optimizing the propagation factor only
            def costfunc_propagation_factor(f, k, r_surf, x0):
                y = np.sum((x0 + f*k - r_surf)**2)
                return y

            # next, optimize the surface parameter
            def costfunc_surface_param(p, f, k, surface, x0):
                y = np.sum((x0 + f * k - surface.surface(p)) ** 2)
                return y
            
            r_surface = surface.surface(np.mean(surface.param_range))
            
            for ii in range(0, 2):

                # optimize only propation factor
                f_guess = self.direction @ (r_surface - self.position)
    
                # optimize surface paramter
                sol = opt.minimize_scalar(costfunc_surface_param,
                                          args=(f_guess, self.direction, surface, self.position))
                p_guess = sol.x
                r_surface = surface.surface(p_guess)

            # ------------------------------------------
            # final optimization
            # ------------------------------------------
    
            # construct initial guess for final optimization    
            x0 = np.array([f_guess, p_guess])
            
            def target_function(p, raypos, raydir, surface):
                x = raypos + np.abs(p[0]) * raydir - surface.surface(p[1])
                return x
            
            sol = opt.root(target_function, x0, args=(self.position, self.direction, surface), tol=1e-10,
                           method='lm')
    
            if not sol.success:
                raise UserWarning('Could not find intersection of ray and surface.')
                self.terminated = True
    
            elif sol.x[0] < 0.0:
               raise UserWarning('Non-physical result obtained when trying to propagate ray to surface.')
               self.terminated = True
            else:
                # check if solution is within range for surface parameter
                if not np.min(surface.param_range) < sol.x[1] < np.max(surface.param_range):
                    self.terminated = True
                    
            # save parameter value
            self.p = sol.x[1]
            
            # add new position
            self.position = self.position + sol.x[0] * self.direction
            self.positions.append(self.position)
               
        if not self.terminated:
                        
            # add spectral phase
            dl = np.linalg.norm(self.positions[-1] - self.positions[-2]) * material.n(self.wvl)
            self.optpath.append(dl)
            self.phase.append(2 * np.pi * self.freq * dl / (speed_of_light * 1e-9))
            
            # update accumulated optical path length and phase
            self.optpath_acc = np.sum(self.optpath)
            self.phase_acc = np.sum(self.phase)
            
            # if the surface is a monitor, register the ray.
            # TODO
    
    
    def reflect(self, surface):
        """Reflects the ray at a given surface. This function assumes that the ray
        was last propagated to this surface.
        

        Parameters
        ----------
        surface : Surface instance
            The surface the ray is reflected by
        """        

        # Get unit vectors normal and parallel to the surface
        en = surface.normal(self.p)
        ep = np.array([-en[1], en[0]])

        # Construct the matrix that transforms the coordinates of the direction vector in the
        # x-y coordinate system into the coordinate system defined by en and ep, that is, the
        # coordinate system defined by the unit vectors normal and parallel to the surface
        m = np.array([en, ep])

        # Get the 'transformed' direction vector
        k_in = np.dot(m, self.direction)

        # 'Perform' the reflection, that is, invert the component of k_trans that is normal to
        # the surface
        k_out = np.array([-k_in[0], k_in[1]])

        # Apply the inverse transformation to get the new direction vector in the original (xy) coordinate system
        self.direction = np.dot(m.T, k_out)
        self.normalize_direction()

    
    
    def refract(self, surface, material_in, material_out, verbose=False):
        """Refraction of the ray at a material interface. Assumes that the
        ray was last propagated to the surface where refraction occurs.

        Parameters
        ----------
        surface : Surface instance
            The surface where refraction of the beam occurs.
        material_in : raytracing.Material
            Material the incident beam is travelling in
        material_out : raytracing.Material
            Material the refracted beam is travelling in
        """
        
        self.normalize_direction()

        # Get unit vectors normal and parallel to the surface
        en = surface.normal(self.p)
        ep = np.array([-en[1], en[0]])

        # Determine the incidence angle
        x = np.abs(np.dot(self.direction, en)) / lina.norm(en)
        if x > 1.0:
            x = np.around(x, 12)  # to prevent numerical errors at normal incidence
        angle_in = np.arccos(x)
        # Determine outgoing angle
        n_in = material_in.n(self.wvl)
        n_out = material_out.n(self.wvl)
        
        y = np.sin(angle_in) * n_in / n_out

        if np.abs(y) <= 1.0:
            angle_out = np.arcsin(y)
        else:
            angle_out = angle_in
            print("Warning: refracted ray angle could not be determined.")

        if verbose:
            print("Angle in = %.2f deg" % (angle_in / np.pi * 180))
            print("Angle out = %.2f deg" % (angle_out / np.pi * 180))

        # Now determine the new propagation direction.
        k_out_trial = np.dot(rotation_matrix_2d(angle_out), en)
        cn = np.dot(k_out_trial, en)
        cp = np.dot(k_out_trial, ep)

        if cn * np.dot(self.direction, en) < 0.0:
            cn = -cn
        if cp * np.dot(self.direction, ep) < 0.0:
            cp = -cp

        k_out = cn * en + cp * ep
        if verbose:
            print('New direction:')
            print(k_out)
            
        # Set new direction
        self.direction = k_out / np.linalg.norm(k_out)
       
    
    def get_positions(self):
        """Returns the x and y coordinates of the ray positions as separate
        arrays. Useful for plotting.
        
        Returns
        -------
        x : ndarray, shape (n,)
            x-coordinates of ray positions
        y : ndarray, shape (n,)
            y-coordinates of ray positions
        """

        x = np.zeros(len(self.positions))
        y = np.zeros(len(self.positions))

        for ii in range(0, len(self.positions)):
            x[ii] = self.positions[ii][0]
            y[ii] = self.positions[ii][1]

        return x, y
    
   
    def plot(self, ax, style='r', **kwargs):
        """Plots the ray
        
        Parameters
        ----------
        ax : Axes 
            matplotlib axes to plot the contour on
        style : str
            Plot style
        """
        
        x, y = self.get_positions()
        ax.plot(x, y, style, **kwargs)
    

class Surface(object):
    """Class for describing 'surfaces' in two-dimensions. Is based on
    a parametrization of the surface which uses a single parameter 'p', that
    is, the surface is described by a two functions x(p) and y(p).

    Parameters
    ----------
    surface : callable
        Function describing the position on the surface as a function of 
        a single parameter. Must return a 2d vector [x(p), y(p)].
    normal : callable
        Function that returns the surface normal for a given value of the
        surface parameter p
    param_range : tuple
        Tuple of two values specifying the range of valid values for the surface
        parameter p
    """

    def __init__(self, surface, normal, param_range):

        self.surface = surface
        self.normal = normal
        self.param_range = param_range
        

    def get_contour(self, npoints=51):
        
        # get surface contour
        p = np.linspace(np.min(self.param_range), np.max(self.param_range), npoints)
        
        x = np.zeros(npoints)
        y = np.zeros(npoints)

        for ii in range(0, npoints):
            r = self.surface(p[ii])
            x[ii] = r[0]
            y[ii] = r[1]

        return x, y
    
    
    def plot(self, ax, style='k', npoints=51, **kwargs):
        """Function for plotting the surface contour.
        
        Parameters
        ----------
        ax : Axes 
            matplotlib axes to plot the contour on
        style : str
            Plot style
        """
        
        x, y = self.get_contour()
        ax.plot(x, y, style, **kwargs)
        

class SphericalSurface(Surface):
    """A spherical surface. (= segment of a circle)
    
    Parameters
    ----------
    p0 : ndarray, shape (2,)
        Vector pointing to the central point on the surface
    n0 : ndarray, shape (2,) 
        vector pointing from point p0 towards the center of the sphere/circle
    radius: float
        radius of curvature
    angle_range : tuple
        defines the size of the circular segment in radians. (-np.pi, np.pi) 
        corresponds to a full circle
    """

    def __init__(self, p0, n0, radius, angle_range):

        self.p0 = np.array(p0)
        self.n0 = np.array(n0) / lina.norm(n0)
        self.radius = radius
        
        angle_offset = np.arctan2(-n0[1], -n0[0])
        
        self.angle_range = (angle_range[0] + angle_offset,
                            angle_range[1] + angle_offset)
        
        origin = self.p0 + self.n0 * self.radius
        
        def circle_surface(p):
            return origin + radius*np.array([np.cos(p), np.sin(p)])

        def circle_normal(p):
            return -np.array([np.cos(p), np.sin(p)])

        Surface.__init__(self, circle_surface, circle_normal, self.angle_range)
        

class FlatSurface(Surface):
    """Flat surface (a straight line in 2 dimensions) which is defined by a 
    'support vector' p0 pointing to a point on the surface and the direction
    vector dd which defines the direction of the straight line
    
    Parameters
    ----------
    p0 : ndarray, shape (2,)
        2d vector pointing to a point on the surface/line
    d : ndarray, shape (2,)
        2d vector specifying the direction of the surface/line
    param_range : tuple
        Tuple of values specifying the surface size, in mm. For example
        (-3, 4) corresponds to a straight line that extents 3 mm in the direction
        -dd and 4 mm in the direction dd from the point p0
    """
    
    def __init__(self, p0, d, param_range):

        self.p0 = np.array(p0)
        self.d = np.array(d) / lina.norm(np.array(d))

        p0 = self.p0
        d = self.d

        def surface(p):
            return p0 + p * d

        def normal(p):
            n = np.array([-d[1], d[0]])
            return n / lina.norm(n)

        Surface.__init__(self, surface, normal, param_range)
        
        
class ParabolicSurface(Surface):

    def __init__(self, focal_length, focal_point, n, param_range):
        """Parabolic surface, using paramtrization in polar coordinates.
        
        Parameters
        ----------
        focal_length : float
            Distance from vertex to focal point
        focal_point : ndarray, shape (2,) 
            Coordinates of focal point
        n : ndarray, shape(2,) 
            Vector pointing from vertex to focal point
        param_range : tuple
            Parameter range (the parabola is parametrized in polar coordinates
            so that the surface parameter p is an angle)
        """

        # Normalize vector n
        n = -np.array(n) / lina.norm(n)

        def surface(p):
            q = np.cos(p) + 1.0
            x = focal_point[0] + 2.0*focal_length*(n[0]*np.cos(p)/q - n[1]*np.sin(p)/q)
            y = focal_point[1] + 2.0*focal_length*(n[0]*np.sin(p)/q + n[1]*np.cos(p)/q)
            return np.array([x, y])

        def normal(p):
            # First, construct the tangent t which is the derivative of the vector r (pointing to a point of the curve)
            # with respect to the parameter p
            q = np.cos(p) + 1.0
            tx = -n[0]*np.sin(p) / q**2 - n[1] / q
            ty = n[0] / q - n[1]*np.sin(p) / q**2
            # Construct the normal 'y' by rotating the tangent by 90 degrees
            y = np.array([-ty, tx])
            return y / lina.norm(y)

        Surface.__init__(self, surface, normal, param_range=param_range)
        

class Prism(object):
    """Prism class for the description of triangular prisms consisting of two
    sides (referred to as side1 and side2) and a base, each an instance of
    the FlatSurface class. The 'n' vector
    points from the prism apex to the center of the prism base and thus
    controls the orietation of the prism. 'side1' corresponds
    to this 'n' vector rotated clockwise by half the apex angle.
    'side2' is given by counter-clockwise rotation rotation of 'n' by half
    the apex angle.
    
    Parameters
    ----------
    angle : float
        Apex angle of the prism in degrees
    apex_position : ndarray, shape (2,)
        Position of the prism apex in units of mm
    height : float
        Prism height in mm
    n : ndarray, shape (2,)
        Vector pointing from the prism apex to the center of the prism base
    material : raytracing.Material
        The prism material
    """

    def __init__(self, angle, apex_position, height, n, material=Material.fromName('FusedSilica')):
        
        self.apex_angle = angle / 180. * np.pi  # apex angle in rad
        self.apex_position = np.array(apex_position)
        self.h = height
        self.n = np.array(n) / np.linalg.norm(n)
        self.material = material
        
        # =====================================================================
        # Create prism sides and base
        # =====================================================================
        
        # get direction of prism base
        n_base = np.array([self.n[-1], -self.n[0]])
        
        # get direction of sides
        dir1 = rotation_matrix_2d(-0.5 * self.apex_angle)@self.n
        dir2 = rotation_matrix_2d(0.5 * self.apex_angle)@self.n

        # get length of sides and base
        side_length = self.h / np.cos(0.5 * self.apex_angle)
        base_length = 2 * self.h * np.tan(0.5 * self.apex_angle)

        # create sides of the prism
        self.side1 = FlatSurface(self.apex_position, dir1, param_range=[0, side_length])
        self.side2 = FlatSurface(self.apex_position, dir2, param_range=[0, side_length])
        self.base = FlatSurface(self.apex_position + self.h*self.n, n_base,
                                param_range=[-0.5*base_length, 0.5*base_length])

    def plot(self, ax, style='k', **kwargs):
        """Plots the prism contour
        
        Parameters
        ----------
        ax : Axes 
            matplotlib axes to plot the contour on
        style : str
            Plot style
        """
        
        self.side1.plot(ax, style=style, **kwargs)
        self.side2.plot(ax, style=style, **kwargs)
        self.base.plot(ax, style=style, **kwargs)
        

class Monitor(FlatSurface):
    
    def __init__(self, p0, d, param_range, label='Monitor', wvl0=None):
        
        # initialize surface
        super().__init__(p0, d, param_range)
        
        # initialize arrays for storing monitor data
        self.freq = np.array([])
        self.wvl = np.array([])
        
        # center wavelength
        self.wvl0 = wvl0
        
        # spatial information
        self.pos = []
        self.dir = []
        self.spatial_chirp = np.array([])
        self.angle = np.array([])
        
        # dispersion information
        self.phase = np.array([])
        self.opt_path = np.array([])
        self.gd = np.array([])
        self.gdd = np.array([])
    
    def register_ray(self, ray):
        
        # add frequency and wavelength 
        self.freq = np.append(self.freq, ray.freq)
        self.wvl = np.append(self.wvl, speed_of_light / ray.freq * 1e-3)
          
        # position and direction
        self.pos.append(ray.position)
        self.dir.append(ray.direction)
        
        # optical path (which can be directly converted to phase later on)
        self.opt_path = np.append(self.opt_path, ray.optpath_acc)
                
    def evaluate(self):
        
        self.pos = np.array(self.pos)
        self.dir = np.array(self.dir)
               
                
        # sort according to frequency
        idx = np.argsort(self.freq)
        
        # sort arrays with recorded data
        self.freq = self.freq[idx]
        self.wvl = self.wvl[idx]
        self.pos = self.pos[idx]
        self.dir = self.dir[idx]
        self.opt_path = self.opt_path[idx]
                
        # center wavelength
        if self.wvl0 is None:
            self.wvl0 = np.median(self.wvl)
            
        self.freq0 = speed_of_light / self.wvl0 * 1e-3
                
        # evaluate optical phase
        self.phase = 2 * np.pi * self.freq / speed_of_light * self.opt_path * 1e9
        
        # evaluate ray angle (in the global cartsian coordinate system)
        self.angle = np.arctan2(self.dir[:, 1], self.dir[:, 0])
        
        # now evaluat group delay and group delay dispersion by
        # differentiation of the optical phase
        dw = 2 * np.pi * (self.freq[-1] - self.freq[0]) / (len(self.freq) - 1) * 1e-3

        self.gd = np.gradient(self.phase, dw)  # group delay in fs
        # Fix edges
        self.gd = fix_edges(self.gd, 1)

        self.gdd = np.gradient(self.gd, dw)    # group delay dispersion in fs^2
        self.gdd = fix_edges(self.gdd, 4)
        
        # shift group delay
        i0 = np.searchsorted(self.freq, self.freq0)
        self.gd = self.gd - self.gd[i0]
        
        # evaluate spatial chirp at the detector/monitor
        # the chirp is evaluated along the axis 'd' which defines the
        # orientation of the monitor surface
        self.spatial_chirp = np.zeros(len(self.wvl))
        for ii in range(0, len(self.freq)):
            dr = self.pos[ii] - self.p0
            self.spatial_chirp[ii] = np.dot(dr, self.d)
        
    
    def plot_report(self):
        # Create plot showing the monitor data
        fig = plt.figure(figsize=(6, 5))
        
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        # Group delay and group delay dispersion
        ax1.plot(self.wvl, self.gd, '-' ,color='k')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Group delay (fs)')
        
        ax2.plot(self.wvl, self.gdd, '-', color='k')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel(r'Group delay dispersion (fs$^2$)')
        
        ax3.plot(self.wvl, self.spatial_chirp, '-', color='k')
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Position (mm)')
        
        ax4.plot(self.wvl, self.angle * 180 / np.pi, 'k-')
        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Angle (deg)')
        
        plt.tight_layout()
        plt.show()
        
    def report(self):
        pass
        
        
        