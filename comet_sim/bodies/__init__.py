"""
Bodies package initialization.
"""

from .comet import CometBody, CometComposition
from .celestial import CelestialBody, create_sun, create_jupiter, create_earth, create_saturn, create_solar_system

__all__ = [
    'CometBody',
    'CometComposition', 
    'CelestialBody',
    'create_sun',
    'create_jupiter',
    'create_earth', 
    'create_saturn',
    'create_solar_system'
]
