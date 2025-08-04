"""
Celestial body representations for gravitational calculations.
"""

import numpy as np
from typing import Optional


class CelestialBody:
    """
    Represents a celestial body (planet, star, etc.) for gravitational calculations.
    
    This class models massive bodies that influence comet trajectories through
    gravitational and tidal forces.
    """
    
    def __init__(self,
                 name: str,
                 mass: float,
                 radius: float,
                 position: np.ndarray,
                 velocity: Optional[np.ndarray] = None):
        """
        Initialize a celestial body.
        
        Args:
            name: Body identifier (e.g., "Sun", "Jupiter")
            mass: Mass in kg
            radius: Radius in meters
            position: Position vector in meters [x, y, z]
            velocity: Velocity vector in m/s [vx, vy, vz] (optional)
        """
        self.name = name
        self.mass = float(mass)
        self.radius = float(radius)
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float) if velocity is not None else np.zeros(3)
        
        # Derived properties
        self.mu = 6.67430e-11 * self.mass  # Standard gravitational parameter
        self.surface_gravity = 6.67430e-11 * self.mass / (self.radius**2)
        
        # Hill sphere radius (approximate, assuming circular orbit around Sun)
        if name != "Sun":
            self.hill_radius = np.linalg.norm(self.position) * (self.mass / (3 * 1.989e30))**(1/3)
        else:
            self.hill_radius = float('inf')
    
    def distance_to(self, other_position: np.ndarray) -> float:
        """
        Calculate distance to another position.
        
        Args:
            other_position: Position vector to calculate distance to
            
        Returns:
            Distance in meters
        """
        return np.linalg.norm(self.position - other_position)
    
    def gravitational_acceleration(self, position: np.ndarray) -> np.ndarray:
        """
        Calculate gravitational acceleration at a given position.
        
        Args:
            position: Position vector in meters
            
        Returns:
            Acceleration vector in m/s²
        """
        displacement = self.position - position
        distance = np.linalg.norm(displacement)
        
        if distance == 0:
            return np.zeros(3)
            
        # F = GMm/r² in direction of displacement
        acceleration_magnitude = 6.67430e-11 * self.mass / (distance**2)
        acceleration_direction = displacement / distance
        
        return acceleration_magnitude * acceleration_direction
    
    def roche_limit(self, satellite_density: float) -> float:
        """
        Calculate the Roche limit for a satellite with given density.
        
        Args:
            satellite_density: Density of the satellite in kg/m³
            
        Returns:
            Roche limit distance in meters
        """
        # Roche limit for a fluid satellite
        primary_density = self.mass / ((4/3) * np.pi * self.radius**3)
        roche_distance = 2.44 * self.radius * (primary_density / satellite_density)**(1/3)
        
        return roche_distance
    
    def tidal_acceleration(self, position: np.ndarray, satellite_radius: float = 0) -> np.ndarray:
        """
        Calculate tidal acceleration at a given position.
        
        Args:
            position: Position vector in meters
            satellite_radius: Radius of satellite experiencing tidal force
            
        Returns:
            Tidal acceleration vector in m/s²
        """
        displacement = position - self.position
        distance = np.linalg.norm(displacement)
        
        if distance == 0:
            return np.zeros(3)
        
        # Tidal acceleration is the difference between gravitational acceleration
        # at the satellite's center and at its surface
        G = 6.67430e-11
        
        # Acceleration at center
        acc_center = G * self.mass / (distance**2)
        
        # Gradient of acceleration (tidal effect)
        tidal_gradient = 2 * G * self.mass / (distance**3)
        
        # Tidal acceleration magnitude
        tidal_magnitude = tidal_gradient * satellite_radius
        
        # Direction away from primary
        direction = displacement / distance
        
        return tidal_magnitude * direction
    
    def is_within_roche_limit(self, position: np.ndarray, satellite_density: float) -> bool:
        """
        Check if a position is within the Roche limit.
        
        Args:
            position: Position to check
            satellite_density: Density of satellite
            
        Returns:
            True if within Roche limit
        """
        distance = self.distance_to(position)
        roche_distance = self.roche_limit(satellite_density)
        
        return distance < roche_distance
    
    def update_position(self, dt: float):
        """
        Update position based on velocity (for moving bodies).
        
        Args:
            dt: Time step in seconds
        """
        self.position += self.velocity * dt
    
    def get_orbital_velocity(self, distance: float) -> float:
        """
        Calculate circular orbital velocity at given distance.
        
        Args:
            distance: Distance from center in meters
            
        Returns:
            Orbital velocity in m/s
        """
        return np.sqrt(6.67430e-11 * self.mass / distance)
    
    def escape_velocity(self, distance: float) -> float:
        """
        Calculate escape velocity at given distance.
        
        Args:
            distance: Distance from center in meters
            
        Returns:
            Escape velocity in m/s
        """
        return np.sqrt(2 * 6.67430e-11 * self.mass / distance)
    
    def __str__(self) -> str:
        """String representation of the celestial body."""
        r_au = np.linalg.norm(self.position) / 1.496e11 if self.name != "Sun" else 0
        
        return (f"{self.name}: "
                f"M={self.mass:.2e} kg, "
                f"R={self.radius:.0e} m, "
                f"pos={r_au:.2f} AU")


# Factory functions for common celestial bodies
def create_sun() -> CelestialBody:
    """Create the Sun."""
    return CelestialBody(
        name="Sun",
        mass=1.989e30,  # kg
        radius=6.96e8,  # meters
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0])
    )


def create_jupiter(orbital_distance: float = 5.2) -> CelestialBody:
    """
    Create Jupiter at specified orbital distance.
    
    Args:
        orbital_distance: Distance from Sun in AU
        
    Returns:
        Jupiter celestial body
    """
    au = 1.496e11  # meters per AU
    distance = orbital_distance * au
    
    # Circular orbital velocity around Sun
    orbital_velocity = np.sqrt(6.67430e-11 * 1.989e30 / distance)
    
    return CelestialBody(
        name="Jupiter",
        mass=1.898e27,  # kg
        radius=7.149e7,  # meters
        position=np.array([distance, 0.0, 0.0]),
        velocity=np.array([0.0, orbital_velocity, 0.0])
    )


def create_earth() -> CelestialBody:
    """Create Earth at 1 AU."""
    au = 1.496e11
    orbital_velocity = np.sqrt(6.67430e-11 * 1.989e30 / au)
    
    return CelestialBody(
        name="Earth",
        mass=5.972e24,  # kg
        radius=6.371e6,  # meters
        position=np.array([au, 0.0, 0.0]),
        velocity=np.array([0.0, orbital_velocity, 0.0])
    )


def create_saturn() -> CelestialBody:
    """Create Saturn at 9.5 AU."""
    au = 1.496e11
    distance = 9.5 * au
    orbital_velocity = np.sqrt(6.67430e-11 * 1.989e30 / distance)
    
    return CelestialBody(
        name="Saturn",
        mass=5.683e26,  # kg
        radius=5.823e7,  # meters
        position=np.array([distance, 0.0, 0.0]),
        velocity=np.array([0.0, orbital_velocity, 0.0])
    )


def create_solar_system(include_outer_planets: bool = True) -> list:
    """
    Create a list of solar system bodies.
    
    Args:
        include_outer_planets: Whether to include Saturn, Uranus, Neptune
        
    Returns:
        List of celestial bodies
    """
    bodies = [
        create_sun(),
        create_jupiter()
    ]
    
    if include_outer_planets:
        bodies.append(create_saturn())
        
        # Uranus
        au = 1.496e11
        distance = 19.2 * au
        orbital_velocity = np.sqrt(6.67430e-11 * 1.989e30 / distance)
        bodies.append(CelestialBody(
            name="Uranus",
            mass=8.681e25,  # kg
            radius=2.556e7,  # meters
            position=np.array([distance, 0.0, 0.0]),
            velocity=np.array([0.0, orbital_velocity, 0.0])
        ))
        
        # Neptune
        distance = 30.1 * au
        orbital_velocity = np.sqrt(6.67430e-11 * 1.989e30 / distance)
        bodies.append(CelestialBody(
            name="Neptune",
            mass=1.024e26,  # kg
            radius=2.476e7,  # meters
            position=np.array([distance, 0.0, 0.0]),
            velocity=np.array([0.0, orbital_velocity, 0.0])
        ))
    
    return bodies
