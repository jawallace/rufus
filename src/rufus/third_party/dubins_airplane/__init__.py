'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

This module contains the third-party, open-source implementation of Dubins
Airplane.

Source: https://github.com/unr-arl/DubinsAirplane

The source is unmodified with the following exceptions:
    (1) The custom implementation of max and min were removed (simply removed
        the `from ElementaryFunctions import max, min` line

    (2) `print` statements were made python 3 compliant

    (3) ExtractDubinsAirplanePath was modified to accept the time step
        parameter as an input rather than a constant
'''

# hoist all symbols into this namespace
from rufus.third_party.dubins_airplane.DubinsAirplaneFunctions import *
