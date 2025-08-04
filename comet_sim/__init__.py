"""
Comet Fragmentation Simulator

A comprehensive physics-based simulation system for modeling comet fragmentation
under gravitational and tidal forces.
"""

from .core.simulator import CometSimulator
from .bodies.comet import CometBody
from .bodies.celestial import CelestialBody
from .physics.forces import GravitationalForces, TidalForces, FragmentationModel
from .physics.fragmentation import CometFragmentation, FragmentationEvent, Fragment

__version__ = "1.0.0"
__author__ = "Comet Fragmentation Research Team"

__all__ = [
    "CometSimulator",
    "CometBody", 
    "CelestialBody",
    "GravitationalForces",
    "TidalForces",
    "FragmentationModel"
]
