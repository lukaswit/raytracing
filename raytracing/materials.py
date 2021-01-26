# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:52:52 2021

@author: Lukas
"""

import numpy as np
from scipy.constants import speed_of_light

class Material(object):

    def __init__(self, name, n):
        self.n = n
        self.name = name
        
    @classmethod
    def fromName(cls, name):
        available_materials = ['Vacuum',
                           'FusedSilica',
                           'Air',
                           'BK7',
                           'CaF',
                           'Sapphire',
                           'BBO_no',
                           'BBO_ne']
        if name not in available_materials:
            err_msg = 'Invalid material name: {:s}. Available materials:'.format(name)
            for x in available_materials:
                err_msg += ' ' + x + ','
            raise ValueError(err_msg)
        
        # Collection of expression for refractive index (various materials)
        def n_air(x):
            x = x * 1e-3  # convert from nm to um
            n = 1 + 0.05792105 / (238.0185 - x ** -2) + 0.00167917 / (57.362 - x ** -2)
            return n
        
        
        def n_bk7(x):
            x = x * 1e-3  # convert from nm to um
            n = (1 + 1.03961212 / (1 - 0.00600069867 / x ** 2) + 0.231792344 / (1 - 0.0200179144 / x ** 2) + 1.01046945 / (
                        1 - 103.560653 / x ** 2)) ** .5
            return n
        
        
        def n_fused_silica(x):
            x = x * 1e-3  # convert from nm to um
            n = (1 + 0.6961663 / (1 - (0.0684043 / x) ** 2) + 0.4079426 / (1 - (0.1162414 / x) ** 2) + 0.8974794 / (
                        1 - (9.896161 / x) ** 2)) ** .5
            return n
        
        
        def n_calcium_fluoride(x):
            x = x * 1e-3  # convert from nm to um
            n = (1 + 0.443749998 / (1 - 0.00178027854 / x ** 2) + 0.444930066 / (1 - 0.00788536061 / x ** 2) + 0.150133991 / (
                        1 - 0.0124119491 / x ** 2) + 8.85319946 / (1 - 2752.28175 / x ** 2)) ** .5
            return n
        
        
        def n_sapphire(x):
            x = x * 1e-3  # convert from nm to um
            n = (1 + 1.023798 / (1 - (0.06144821 / x) ** 2) + 1.058264 / (1 - (0.1106997 / x) ** 2) + 5.280792 / (
                        1 - (17.92656 / x) ** 2)) ** .5
            return n
        
        
        def no_bbo(x):
            x = x * 1e-3  # convert from nm to um
            n = (2.7405 + 0.0184 / (x ** 2 - 0.0179) - 0.0155 * x ** 2) ** .5
            return n
        
        
        def ne_bbo(x):
            x = x * 1e-3  # convert from nm to um
            n = (2.3730 + 0.0128 / (x ** 2 - 0.0156) - 0.0044 * x ** 2) ** .5
            return n
        
        # Create the material an return class instance with proper
        # refractive index function
        if name == 'Vacuum':
            return cls('Vacuum', lambda x: 1.0)
        elif name == 'FusedSilica':
            return cls('FusedSilica', n_fused_silica)
        elif name == 'Air':
            return cls('Air', n_air)
        elif name == 'BK7':
            return cls('BK7', n_bk7)
        elif name == 'CaF':
            return cls('CaF', n_calcium_fluoride)
        elif name == 'Sapphire':
            return cls('Sapphire', n_sapphire)
        elif name == 'BBO_no':
            return cls('BBO_no', no_bbo)
        elif name == 'BBO_ne':
            return cls('BBO_ne', ne_bbo)
        

    def n_first_derivative(self, x):
        dx = 0.0005
        n1 = self.n(x + dx)
        n2 = self.n(x - dx)
        dndx = (n1 - n2) / (2. * dx)
        return dndx

    def n_second_derivative(self, x):
        dx = 0.0005
        f1 = self.n_first_derivative(x + dx)
        f2 = self.n_first_derivative(x - dx)
        deriv_2nd = (f1 - f2) / (2. * dx)
        return deriv_2nd

    def n_third_derivative(self, x):
        dx = 0.0005
        f1 = self.n_second_derivative(x + dx)
        f2 = self.n_second_derivative(x - dx)
        deriv_3rd = (f1 - f2) / (2. * dx)
        return deriv_3rd

    def gvd(self, x):
        """Calculates the group velocity dispersion at a given wavelength        

        Parameters
        ----------
        x : float
            wavelength in nm

        Returns
        -------
        gvd : float
            group velocity dispersion in fs^2 / mm
        """
        
        gvd = x**3 / (2. * np.pi * (speed_of_light * 1e-9)**2) * self.n_second_derivative(x)
        return gvd


    def tod(self, x):
        # x: wavelength in nm
        n2 = self.n_second_derivative(x)
        n3 = self.n_third_derivative(x)
        tod = -x**4 / (4. * np.pi**2 * (speed_of_light * 1e-9)**3) * (3*n2 + x * n3) * 1e3
        return tod

    def group_velocity(self, x):
        # x: wavelength in nm
        # Returns the group velocity in mm/fs
        vg = speed_of_light * 1e-12 / (self.n(x) - x * self.n_first_derivative(x))
        return vg



