"""
Comet body representation and dynamics.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class CometComposition:
    """Represents the composition of a comet."""
    ice_fraction: float = 0.6
    rock_fraction: float = 0.3
    dust_fraction: float = 0.1
    
    def __post_init__(self):
        total = self.ice_fraction + self.rock_fraction + self.dust_fraction
        if not np.isclose(total, 1.0):
            raise ValueError(f"Composition fractions must sum to 1.0, got {total}")


class CometBody:
    """
    Represents a comet with physical and dynamic properties.
    
    This class models a comet as a celestial body with:
    - Physical properties (mass, size, density, composition)
    - Dynamic state (position, velocity)
    - Structural properties (integrity, fragmentation threshold)
    """
    
    def __init__(self,
                 name: str,
                 mass: float,
                 radius: float,
                 density: float,
                 composition: Dict[str, float],
                 position: np.ndarray,
                 velocity: np.ndarray,
                 structural_integrity: float):
        """
        Initialize a comet body.
        
        Args:
            name: Comet identifier
            mass: Mass in kg
            radius: Mean radius in meters
            density: Bulk density in kg/m³
            composition: Dict with 'ice', 'rock', 'dust' fractions
            position: Position vector in meters [x, y, z]
            velocity: Velocity vector in m/s [vx, vy, vz]
            structural_integrity: Breaking stress in Pa
        """
        self.name = name
        self.mass = float(mass)
        self.radius = float(radius)
        self.density = float(density)
        
        # Validate composition
        self.composition = CometComposition(
            ice_fraction=composition.get('ice', 0.6),
            rock_fraction=composition.get('rock', 0.3),
            dust_fraction=composition.get('dust', 0.1)
        )
        
        # Dynamic state
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        
        # Structural properties
        self.structural_integrity = float(structural_integrity)
        self.original_integrity = self.structural_integrity
        
        # Derived properties
        self.volume = (4/3) * np.pi * radius**3
        self.surface_area = 4 * np.pi * radius**2
        
        # State tracking
        self.age = 0.0  # Time since creation
        self.is_fragmented = False
        self.stress_history = []
        
        # Thermal properties (simplified)
        self.temperature = 2.7  # Kelvin (cosmic background)
        self.albedo = 0.04  # Typical comet albedo
        
    def update_state(self, dt: float):
        """
        Update the comet's internal state over time step dt.
        
        Args:
            dt: Time step in seconds
        """
        self.age += dt
        
        # Simple thermal evolution (heating from sun)
        distance_from_sun = np.linalg.norm(self.position)
        au = 1.496e11  # Astronomical unit in meters
        
        # Simple solar heating model
        solar_flux = 1361 * (au / distance_from_sun)**2  # W/m²
        absorbed_power = solar_flux * self.surface_area * (1 - self.albedo)
        
        # Simplified temperature calculation
        stefan_boltzmann = 5.67e-8  # W/m²/K⁴
        equilibrium_temp = (absorbed_power / (self.surface_area * stefan_boltzmann))**(1/4)
        
        # Gradual temperature adjustment
        temp_change_rate = 0.1  # Simplified thermal inertia
        self.temperature += (equilibrium_temp - self.temperature) * temp_change_rate * dt / 3600
        
        # Sublimation effects (very simplified)
        if self.temperature > 150:  # K, approximate sublimation threshold
            sublimation_rate = (self.temperature - 150) * 1e-12  # kg/s/K (very rough)
            mass_loss = sublimation_rate * self.surface_area * dt
            
            # Reduce mass and radius
            if mass_loss < self.mass * 0.001:  # Limit to 0.1% per step
                self.mass -= mass_loss
                # Update radius assuming constant density
                self.radius = (3 * self.mass / (4 * np.pi * self.density))**(1/3)
                self.volume = (4/3) * np.pi * self.radius**3
                self.surface_area = 4 * np.pi * self.radius**2
    
    def apply_stress(self, stress: float):
        """
        Apply stress to the comet and track damage.
        
        Args:
            stress: Applied stress in Pa
        """
        self.stress_history.append(stress)
        
        # Cumulative damage model (simplified)
        if stress > self.structural_integrity * 0.5:
            damage_factor = stress / self.structural_integrity
            self.structural_integrity *= (1 - 0.01 * damage_factor)
            
        # Check for critical failure
        if stress > self.structural_integrity:
            self.is_fragmented = True
    
    def get_orbital_elements(self, primary_mass: float) -> Dict[str, float]:
        """
        Calculate approximate orbital elements relative to a primary body.
        
        Args:
            primary_mass: Mass of primary body (e.g., Sun) in kg
            
        Returns:
            Dictionary of orbital elements
        """
        G = 6.67430e-11  # Gravitational constant
        
        r = np.linalg.norm(self.position)
        v = np.linalg.norm(self.velocity)
        
        # Semi-major axis from vis-viva equation
        mu = G * primary_mass
        specific_energy = 0.5 * v**2 - mu / r
        
        if specific_energy < 0:  # Bound orbit
            a = -mu / (2 * specific_energy)
            
            # Angular momentum
            h_vec = np.cross(self.position, self.velocity)
            h = np.linalg.norm(h_vec)
            
            # Eccentricity
            e_squared = 1 + (2 * specific_energy * h**2) / (mu**2)
            e = np.sqrt(max(0, e_squared))
            
            # Inclination
            if h > 0:
                i = np.arccos(h_vec[2] / h) * 180 / np.pi
            else:
                i = 0.0
                
            # Perihelion distance
            q = a * (1 - e)
            
            # Aphelion distance
            Q = a * (1 + e)
            
            return {
                'semi_major_axis': a,
                'eccentricity': e,
                'inclination': i,
                'perihelion_distance': q,
                'aphelion_distance': Q,
                'orbital_period': 2 * np.pi * np.sqrt(a**3 / mu)
            }
        else:
            return {
                'semi_major_axis': float('inf'),
                'eccentricity': float('inf'),
                'inclination': 0.0,
                'perihelion_distance': r,
                'aphelion_distance': float('inf'),
                'orbital_period': float('inf')
            }
    
    def get_physical_properties(self) -> Dict[str, float]:
        """Get summary of physical properties."""
        return {
            'mass': self.mass,
            'radius': self.radius,
            'density': self.density,
            'volume': self.volume,
            'surface_area': self.surface_area,
            'temperature': self.temperature,
            'structural_integrity': self.structural_integrity,
            'integrity_fraction': self.structural_integrity / self.original_integrity,
            'age': self.age,
            'ice_fraction': self.composition.ice_fraction,
            'rock_fraction': self.composition.rock_fraction,
            'dust_fraction': self.composition.dust_fraction
        }
    
    def __str__(self) -> str:
        """String representation of the comet."""
        r_au = np.linalg.norm(self.position) / 1.496e11
        v_kms = np.linalg.norm(self.velocity) / 1000
        
        return (f"Comet {self.name}: "
                f"m={self.mass:.2e} kg, "
                f"r={self.radius:.0f} m, "
                f"pos={r_au:.3f} AU, "
                f"vel={v_kms:.1f} km/s, "
                f"T={self.temperature:.1f} K")
    
    def copy(self) -> 'CometBody':
        """Create a copy of this comet."""
        composition_dict = {
            'ice': self.composition.ice_fraction,
            'rock': self.composition.rock_fraction,
            'dust': self.composition.dust_fraction
        }
        
        new_comet = CometBody(
            name=f"{self.name}_copy",
            mass=self.mass,
            radius=self.radius,
            density=self.density,
            composition=composition_dict,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            structural_integrity=self.structural_integrity
        )
        
        new_comet.age = self.age
        new_comet.temperature = self.temperature
        new_comet.is_fragmented = self.is_fragmented
        new_comet.stress_history = self.stress_history.copy()
        
        return new_comet


# Factory functions for common comet types
def create_small_comet(name: str = "Small Comet") -> CometBody:
    """Create a small comet (< 1 km diameter)."""
    return CometBody(
        name=name,
        mass=1e12,  # 1 billion kg
        radius=500,  # 500 m
        density=500,  # kg/m³
        composition={'ice': 0.7, 'rock': 0.2, 'dust': 0.1},
        position=np.array([1.5e11, 0, 0]),  # 1 AU
        velocity=np.array([0, 25000, 0]),  # 25 km/s
        structural_integrity=5e5  # 500 kPa
    )


def create_large_comet(name: str = "Large Comet") -> CometBody:
    """Create a large comet (> 10 km diameter)."""
    return CometBody(
        name=name,
        mass=1e15,  # 1 trillion kg
        radius=5000,  # 5 km
        density=600,  # kg/m³
        composition={'ice': 0.5, 'rock': 0.4, 'dust': 0.1},
        position=np.array([3e11, 0, 0]),  # 2 AU
        velocity=np.array([0, 20000, 0]),  # 20 km/s
        structural_integrity=2e6  # 2 MPa
    )


def create_shoemaker_levy_9() -> CometBody:
    """Create a comet similar to Shoemaker-Levy 9 before fragmentation."""
    return CometBody(
        name="SL9-type",
        mass=5e14,  # Estimated 500 billion kg
        radius=3000,  # ~3 km radius
        density=400,  # Low density
        composition={'ice': 0.8, 'rock': 0.15, 'dust': 0.05},
        position=np.array([7.8e11, 0, 0]),  # Near Jupiter
        velocity=np.array([0, 15000, 0]),  # Slow approach
        structural_integrity=1e5  # Very weak structure
    )
