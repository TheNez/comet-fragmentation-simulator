"""
Physical force calculations for comet simulation.
"""

import numpy as np
from typing import List, Tuple


class GravitationalForces:
    """
    Calculates gravitational forces between bodies.
    
    Implements N-body gravitational dynamics using Newton's law of universal gravitation.
    """
    
    def __init__(self):
        self.G = 6.67430e-11  # Gravitational constant (m³/kg/s²)
    
    def force_between_bodies(self, 
                           pos1: np.ndarray, 
                           mass1: float,
                           pos2: np.ndarray,
                           mass2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate gravitational force between two bodies.
        
        Args:
            pos1: Position of first body [x, y, z] in meters
            mass1: Mass of first body in kg
            pos2: Position of second body [x, y, z] in meters
            mass2: Mass of second body in kg
            
        Returns:
            Tuple of (force_on_body1, force_on_body2) in Newtons
        """
        # Displacement vector from body1 to body2
        displacement = pos2 - pos1
        distance = np.linalg.norm(displacement)
        
        if distance == 0:
            return np.zeros(3), np.zeros(3)
        
        # Force magnitude
        force_magnitude = self.G * mass1 * mass2 / (distance**2)
        
        # Unit vector from body1 to body2
        direction = displacement / distance
        
        # Force on body1 (towards body2)
        force_on_1 = force_magnitude * direction
        
        # Force on body2 (towards body1, Newton's 3rd law)
        force_on_2 = -force_on_1
        
        return force_on_1, force_on_2
    
    def acceleration_from_body(self,
                             target_pos: np.ndarray,
                             source_pos: np.ndarray,
                             source_mass: float) -> np.ndarray:
        """
        Calculate gravitational acceleration on a target from a source body.
        
        Args:
            target_pos: Position of target body
            source_pos: Position of source body
            source_mass: Mass of source body
            
        Returns:
            Acceleration vector in m/s²
        """
        displacement = source_pos - target_pos
        distance = np.linalg.norm(displacement)
        
        if distance == 0:
            return np.zeros(3)
        
        # a = GM/r² in direction of source
        acceleration_magnitude = self.G * source_mass / (distance**2)
        direction = displacement / distance
        
        return acceleration_magnitude * direction
    
    def n_body_acceleration(self,
                          target_pos: np.ndarray,
                          other_bodies: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Calculate total gravitational acceleration from multiple bodies.
        
        Args:
            target_pos: Position of target body
            other_bodies: List of (position, mass) tuples for other bodies
            
        Returns:
            Total acceleration vector in m/s²
        """
        total_acceleration = np.zeros(3)
        
        for pos, mass in other_bodies:
            total_acceleration += self.acceleration_from_body(target_pos, pos, mass)
        
        return total_acceleration
    
    def potential_energy(self,
                        pos1: np.ndarray,
                        mass1: float,
                        pos2: np.ndarray,
                        mass2: float) -> float:
        """
        Calculate gravitational potential energy between two bodies.
        
        Args:
            pos1: Position of first body
            mass1: Mass of first body
            pos2: Position of second body
            mass2: Mass of second body
            
        Returns:
            Potential energy in Joules (negative value)
        """
        distance = np.linalg.norm(pos2 - pos1)
        
        if distance == 0:
            return float('-inf')
        
        return -self.G * mass1 * mass2 / distance


class TidalForces:
    """
    Calculates tidal forces that can fragment comets.
    
    Tidal forces arise from differential gravitational acceleration across
    the extent of the comet body.
    """
    
    def __init__(self):
        self.G = 6.67430e-11
    
    def tidal_acceleration_gradient(self,
                                  primary_pos: np.ndarray,
                                  primary_mass: float,
                                  satellite_pos: np.ndarray) -> np.ndarray:
        """
        Calculate tidal acceleration gradient at satellite position.
        
        Args:
            primary_pos: Position of primary body (e.g., Jupiter)
            primary_mass: Mass of primary body
            satellite_pos: Position of satellite (comet center)
            
        Returns:
            Tidal gradient tensor components [d²φ/dx², d²φ/dy², d²φ/dz²]
        """
        displacement = satellite_pos - primary_pos
        distance = np.linalg.norm(displacement)
        
        if distance == 0:
            return np.zeros(3)
        
        # Second derivative of gravitational potential
        # φ = -GM/r, so d²φ/dr² = 2GM/r³
        tidal_strength = 2 * self.G * primary_mass / (distance**3)
        
        # Direction vector
        r_unit = displacement / distance
        
        # Tidal gradient in radial direction (stretching)
        radial_gradient = tidal_strength
        
        # Transverse gradients (compression)
        transverse_gradient = -tidal_strength / 2
        
        # Return gradient components
        gradient = np.zeros(3)
        gradient[0] = radial_gradient if abs(r_unit[0]) > 0.5 else transverse_gradient
        gradient[1] = radial_gradient if abs(r_unit[1]) > 0.5 else transverse_gradient
        gradient[2] = radial_gradient if abs(r_unit[2]) > 0.5 else transverse_gradient
        
        return gradient
    
    def tidal_stress_at_position(self,
                               primary_pos: np.ndarray,
                               primary_mass: float,
                               satellite_pos: np.ndarray,
                               relative_pos: np.ndarray,
                               satellite_density: float) -> float:
        """
        Calculate tidal stress at a specific position within the satellite.
        
        Args:
            primary_pos: Position of primary body
            primary_mass: Mass of primary body
            satellite_pos: Center position of satellite
            relative_pos: Position relative to satellite center
            satellite_density: Density of satellite material
            
        Returns:
            Tidal stress in Pascals
        """
        # Distance from primary to satellite center
        displacement = satellite_pos - primary_pos
        distance = np.linalg.norm(displacement)
        
        if distance == 0:
            return 0.0
        
        # Direction from primary to satellite
        direction = displacement / distance
        
        # Distance from satellite center to the point of interest
        relative_distance = np.linalg.norm(relative_pos)
        
        if relative_distance == 0:
            return 0.0
        
        # Component of relative position along the primary-satellite line
        radial_component = np.dot(relative_pos, direction)
        
        # Tidal acceleration difference
        tidal_acceleration = self.G * primary_mass * radial_component / (distance**3)
        
        # Convert to stress (force per unit area)
        # Approximate as density times acceleration times characteristic length
        tidal_stress = satellite_density * abs(tidal_acceleration) * relative_distance
        
        return tidal_stress
    
    def roche_limit(self,
                   primary_mass: float,
                   primary_radius: float,
                   satellite_density: float) -> float:
        """
        Calculate the Roche limit for tidal disruption.
        
        Args:
            primary_mass: Mass of primary body
            primary_radius: Radius of primary body
            satellite_density: Density of satellite
            
        Returns:
            Roche limit distance in meters
        """
        # Density of primary
        primary_density = primary_mass / ((4/3) * np.pi * primary_radius**3)
        
        # Roche limit for a fluid satellite
        roche_distance = 2.44 * primary_radius * (primary_density / satellite_density)**(1/3)
        
        return roche_distance
    
    def is_within_roche_limit(self,
                            primary_pos: np.ndarray,
                            primary_mass: float,
                            primary_radius: float,
                            satellite_pos: np.ndarray,
                            satellite_density: float) -> bool:
        """
        Check if satellite is within the Roche limit.
        
        Args:
            primary_pos: Position of primary body
            primary_mass: Mass of primary body
            primary_radius: Radius of primary body
            satellite_pos: Position of satellite
            satellite_density: Density of satellite
            
        Returns:
            True if within Roche limit
        """
        distance = np.linalg.norm(satellite_pos - primary_pos)
        roche_distance = self.roche_limit(primary_mass, primary_radius, satellite_density)
        
        return distance < roche_distance


class GasPressureForces:
    """
    Calculates forces from outgassing and internal gas pressure.
    
    When ice sublimates, it creates gas pressure that can exceed
    the structural strength of the comet, leading to fragmentation.
    This is often the dominant fragmentation mechanism for comets.
    """
    
    def __init__(self):
        self.R_gas = 8.314  # Universal gas constant (J/mol/K)
        self.M_water = 0.018015  # Molar mass of water (kg/mol)
        self.sublimation_enthalpy = 2.83e6  # J/kg for water ice
    
    def sublimation_rate(self, 
                        temperature: float,
                        surface_area: float,
                        ice_fraction: float = 0.6) -> float:
        """
        Calculate mass loss rate due to sublimation.
        
        Uses Clausius-Clapeyron equation for vapor pressure.
        """
        if temperature < 100:  # Below significant sublimation
            return 0.0
        
        # Vapor pressure using simplified Clausius-Clapeyron
        # P = P₀ * exp(-L/(R*T)) where L is latent heat
        T_ref = 273.15  # Reference temperature (K)
        P_ref = 611.657  # Reference pressure (Pa) - triple point of water
        
        vapor_pressure = P_ref * np.exp(-self.sublimation_enthalpy / (self.R_gas/self.M_water) * (1/temperature - 1/T_ref))
        
        # Mass flux from kinetic theory
        # Φ = P / √(2πmkT) where m is molecular mass, k is Boltzmann constant
        k_B = 1.380649e-23  # Boltzmann constant
        molecular_mass = self.M_water / 6.02214076e23  # kg per molecule
        
        mass_flux = vapor_pressure / np.sqrt(2 * np.pi * molecular_mass * k_B * temperature)
        
        # Total sublimation rate
        sublimation_rate = mass_flux * surface_area * ice_fraction
        
        return sublimation_rate
    
    def internal_gas_pressure(self,
                            temperature: float,
                            ice_fraction: float,
                            porosity: float = 0.3,
                            dt: float = 1.0) -> float:
        """
        Calculate internal gas pressure from sublimation.
        
        Args:
            temperature: Internal temperature (K)
            ice_fraction: Fraction of mass that is ice
            porosity: Fraction of volume that is void space
            dt: Time step (s)
            
        Returns:
            Internal pressure (Pa)
        """
        if temperature < 120:  # Below significant internal sublimation
            return 0.0
        
        # Internal sublimation creates gas that must escape
        # P = nRT/V where n is moles of gas, V is available volume
        
        # Simplified: assume some fraction of ice sublimates internally
        internal_sublimation_rate = ice_fraction * np.exp((temperature - 120) / 50)  # Empirical
        
        # Gas pressure buildup (assuming limited escape)
        # For porous comets, gas can escape, but creates transient pressure
        gas_density = internal_sublimation_rate * dt / porosity
        
        # Ideal gas law: P = ρRT/M
        pressure = gas_density * self.R_gas * temperature / self.M_water
        
        # Limit maximum pressure (gas will find ways to escape)
        max_pressure = 1e6  # 1 MPa - very high for comet material
        
        return min(pressure, max_pressure)
    
    def outgassing_force(self,
                        comet_mass: float,
                        comet_radius: float,
                        temperature: float,
                        ice_fraction: float = 0.6) -> np.ndarray:
        """
        Calculate force from directed outgassing (rocket effect).
        
        Args:
            comet_mass: Mass of comet (kg)
            comet_radius: Radius of comet (m)  
            temperature: Surface temperature (K)
            ice_fraction: Fraction of mass that is ice
            
        Returns:
            Force vector (N) - can be random direction
        """
        if temperature < 150:  # Below significant outgassing
            return np.zeros(3)
        
        # Surface area
        surface_area = 4 * np.pi * comet_radius**2
        
        # Sublimation rate
        mass_loss_rate = self.sublimation_rate(temperature, surface_area, ice_fraction)
        
        # Outgassing velocity (escape velocity from comet surface)
        k_B = 1.380649e-23
        molecular_mass = self.M_water / 6.02214076e23
        outgas_velocity = np.sqrt(2 * k_B * temperature / molecular_mass)
        
        # Thrust force = mass_rate * velocity
        thrust_magnitude = mass_loss_rate * outgas_velocity
        
        # Random direction (asymmetric outgassing)
        direction = np.random.normal(0, 1, 3)
        direction = direction / np.linalg.norm(direction)
        
        return thrust_magnitude * direction
    
    def gas_pressure_stress(self,
                          temperature: float,
                          ice_fraction: float,
                          porosity: float = 0.3) -> float:
        """
        Calculate stress from internal gas pressure.
        
        This is often the dominant force for comet fragmentation,
        especially for "fluffy" comets like Shoemaker-Levy 9.
        """
        pressure = self.internal_gas_pressure(temperature, ice_fraction, porosity)
        
        # Convert pressure to stress
        # For thin-walled spherical pressure vessel: σ = PR/(2t)
        # For comets, we can approximate this as direct pressure stress
        
        # Gas pressure acts to expand the comet against structural integrity
        return pressure


class FragmentationModel:
    """
    Models the fragmentation process of comets under various stresses.
    
    Combines gravitational, thermal, and structural stresses to predict
    when and how a comet will break apart.
    """
    
    def __init__(self):
        self.critical_stress_ice = 1e6  # Pa, approximate tensile strength of ice
        self.critical_stress_rock = 1e8  # Pa, approximate tensile strength of rock
    
    def thermal_stress(self,
                      temperature: float,
                      reference_temp: float = 100.0,
                      thermal_expansion: float = 5e-5) -> float:
        """
        Calculate thermal stress from temperature changes.
        
        Args:
            temperature: Current temperature in Kelvin
            reference_temp: Reference temperature in Kelvin
            thermal_expansion: Thermal expansion coefficient (1/K)
            
        Returns:
            Thermal stress in Pascals
        """
        # Young's modulus for ice (approximate)
        youngs_modulus = 9e9  # Pa
        
        # Thermal strain
        thermal_strain = thermal_expansion * (temperature - reference_temp)
        
        # Stress = modulus × strain
        thermal_stress = youngs_modulus * abs(thermal_strain)
        
        return thermal_stress
    
    def rotational_stress(self,
                         angular_velocity: float,
                         radius: float,
                         density: float) -> float:
        """
        Calculate stress from rotation.
        
        Args:
            angular_velocity: Angular velocity in rad/s
            radius: Distance from rotation axis in meters
            density: Material density in kg/m³
            
        Returns:
            Centrifugal stress in Pascals
        """
        # Centrifugal force per unit volume
        centrifugal_acceleration = angular_velocity**2 * radius
        
        # Stress = density × acceleration × radius
        rotational_stress = density * centrifugal_acceleration * radius
        
        return rotational_stress
    
    def total_stress(self,
                    tidal_stress: float,
                    thermal_stress: float,
                    rotational_stress: float,
                    gas_pressure_stress: float = 0.0) -> float:
        """
        Combine different stress sources.
        
        Args:
            tidal_stress: Tidal stress in Pa
            thermal_stress: Thermal stress in Pa
            rotational_stress: Rotational stress in Pa
            gas_pressure_stress: Internal gas pressure stress in Pa
            
        Returns:
            Combined stress in Pa
        """
        # Simple addition for now (could be more sophisticated)
        # Gas pressure stress is often the dominant factor for comets
        return tidal_stress + thermal_stress + rotational_stress + gas_pressure_stress
    
    def fragmentation_probability(self,
                                total_stress: float,
                                material_strength: float,
                                time_step: float) -> float:
        """
        Calculate probability of fragmentation in a time step.
        
        Args:
            total_stress: Combined stress in Pa
            material_strength: Material strength in Pa
            time_step: Time step in seconds
            
        Returns:
            Probability of fragmentation (0-1)
        """
        if total_stress <= material_strength:
            return 0.0
        
        # Stress ratio
        stress_ratio = total_stress / material_strength
        
        # Probability increases with stress ratio and time
        # Using exponential model: P = 1 - exp(-λt) where λ ∝ stress_ratio
        lambda_rate = (stress_ratio - 1) * 1e-6  # Empirical rate constant
        probability = 1.0 - np.exp(-lambda_rate * time_step)
        
        return min(probability, 1.0)
    
    def predict_fragment_sizes(self,
                             parent_mass: float,
                             stress_level: float,
                             num_fragments: int = None) -> List[float]:
        """
        Predict the mass distribution of fragments.
        
        Args:
            parent_mass: Mass of parent body
            stress_level: Stress level causing fragmentation
            num_fragments: Number of fragments (if None, calculated)
            
        Returns:
            List of fragment masses
        """
        if num_fragments is None:
            # More stress → more fragments
            num_fragments = int(2 + stress_level / self.critical_stress_ice * 10)
            num_fragments = min(num_fragments, 50)  # Reasonable upper limit
        
        # Power law distribution (similar to asteroid families)
        # Larger fragments are less common
        fragment_masses = []
        remaining_mass = parent_mass
        
        for i in range(num_fragments - 1):
            # Largest fragment gets significant fraction
            if i == 0:
                fragment_fraction = 0.3 + 0.4 * np.random.random()
            else:
                # Smaller fragments follow power law
                fragment_fraction = np.random.random() * 0.1
            
            fragment_mass = remaining_mass * fragment_fraction
            fragment_masses.append(fragment_mass)
            remaining_mass -= fragment_mass
            
            if remaining_mass <= 0:
                break
        
        # Last fragment gets remaining mass
        if remaining_mass > 0:
            fragment_masses.append(remaining_mass)
        
        return sorted(fragment_masses, reverse=True)
    
    def fragment_velocities(self,
                           fragment_masses: List[float],
                           fragmentation_energy: float) -> List[np.ndarray]:
        """
        Calculate initial velocities of fragments.
        
        Args:
            fragment_masses: Masses of fragments
            fragmentation_energy: Energy available for fragmentation
            
        Returns:
            List of velocity vectors for each fragment
        """
        velocities = []
        
        # Distribute energy among fragments
        total_mass = sum(fragment_masses)
        
        for mass in fragment_masses:
            # Smaller fragments get higher velocities
            velocity_magnitude = np.sqrt(2 * fragmentation_energy * (total_mass / mass) / total_mass)
            
            # Random direction
            direction = np.random.normal(0, 1, 3)
            direction = direction / np.linalg.norm(direction)
            
            velocity = velocity_magnitude * direction
            velocities.append(velocity)
        
        return velocities
