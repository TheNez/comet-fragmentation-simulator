"""
Fragmentation modeling for comet breakup analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Fragment:
    """Represents a comet fragment after breakup."""
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    density: float
    composition: Dict[str, float]
    creation_time: float
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.position = np.array(self.position)
        self.velocity = np.array(self.velocity)


class FragmentationEvent:
    """
    Represents a single fragmentation event.
    
    Tracks the conditions and results of comet breakup.
    """
    
    def __init__(self,
                 parent_id: int,
                 time: float,
                 position: np.ndarray,
                 cause: str,
                 stress_level: float):
        """
        Initialize fragmentation event.
        
        Args:
            parent_id: ID of the parent body that fragmented
            time: Time of fragmentation
            position: Position where fragmentation occurred
            cause: Cause of fragmentation ('tidal', 'thermal', 'rotational')
            stress_level: Stress level that caused fragmentation
        """
        self.parent_id = parent_id
        self.time = time
        self.position = np.array(position)
        self.cause = cause
        self.stress_level = stress_level
        self.fragments: List[Fragment] = []
        self.total_energy_released = 0.0
    
    def add_fragment(self, fragment: Fragment):
        """Add a fragment to this event."""
        self.fragments.append(fragment)
    
    def get_fragment_count(self) -> int:
        """Get number of fragments created."""
        return len(self.fragments)
    
    def get_mass_distribution(self) -> List[float]:
        """Get sorted list of fragment masses."""
        return sorted([f.mass for f in self.fragments], reverse=True)


class CometFragmentation:
    """
    Advanced fragmentation model for comet breakup simulation.
    
    Implements physically-based fragmentation including:
    - Stress-based breakup criteria
    - Fragment size distributions
    - Velocity dispersions
    - Material property inheritance
    """
    
    def __init__(self):
        # Material properties
        self.ice_tensile_strength = 1e6      # Pa
        self.rock_tensile_strength = 1e8     # Pa
        self.ice_density = 920               # kg/m³
        self.rock_density = 2500             # kg/m³
        
        # Fragmentation parameters
        self.min_fragment_mass = 1e6         # kg (minimum trackable fragment)
        self.energy_partition_efficiency = 0.1  # Fraction of stress energy → kinetic energy
        
        # Tracking
        self.fragmentation_events: List[FragmentationEvent] = []
        self.next_fragment_id = 1000
    
    def calculate_material_strength(self, composition: Dict[str, float]) -> float:
        """
        Calculate effective material strength based on composition.
        
        Args:
            composition: Material composition fractions
            
        Returns:
            Effective tensile strength in Pa
        """
        ice_fraction = composition.get('ice', 0.0)
        rock_fraction = composition.get('rock', 0.0)
        
        # Weighted average (simple mixing rule)
        strength = (ice_fraction * self.ice_tensile_strength + 
                   rock_fraction * self.rock_tensile_strength)
        
        return strength
    
    def fragmentation_criterion(self,
                              total_stress: float,
                              material_strength: float,
                              time_step: float) -> bool:
        """
        Determine if fragmentation occurs.
        
        Args:
            total_stress: Combined stress in Pa
            material_strength: Material strength in Pa
            time_step: Time step duration
            
        Returns:
            True if fragmentation occurs
        """
        if total_stress <= material_strength:
            return False
        
        # Probability-based criterion for gradual failure
        stress_ratio = total_stress / material_strength
        failure_rate = (stress_ratio - 1.0) * 1e-5  # s⁻¹
        failure_probability = 1.0 - np.exp(-failure_rate * time_step)
        
        return np.random.random() < failure_probability
    
    def generate_fragment_masses(self,
                               parent_mass: float,
                               stress_ratio: float,
                               fragmentation_energy: float) -> List[float]:
        """
        Generate fragment mass distribution.
        
        Args:
            parent_mass: Mass of parent body
            stress_ratio: Ratio of stress to material strength
            fragmentation_energy: Energy available for fragmentation
            
        Returns:
            List of fragment masses
        """
        # Number of fragments increases with stress ratio
        base_fragments = 2
        stress_fragments = int(stress_ratio * 5)
        num_fragments = min(base_fragments + stress_fragments, 20)
        
        # Power law exponent (steeper = more small fragments)
        power_exponent = 1.5 + stress_ratio * 0.5
        
        # Generate power law distribution
        fragment_masses = []
        total_assigned = 0.0
        
        for i in range(num_fragments - 1):
            # Power law with cutoff
            random_fraction = np.random.power(power_exponent)
            
            # Largest fragment gets substantial mass
            if i == 0:
                mass_fraction = 0.2 + 0.4 * random_fraction
            else:
                mass_fraction = 0.1 * random_fraction
            
            fragment_mass = parent_mass * mass_fraction
            
            # Enforce minimum fragment size
            if fragment_mass < self.min_fragment_mass:
                continue
                
            fragment_masses.append(fragment_mass)
            total_assigned += fragment_mass
            
            # Don't exceed parent mass
            if total_assigned >= parent_mass * 0.95:
                break
        
        # Remaining mass in final fragment
        remaining_mass = parent_mass - total_assigned
        if remaining_mass >= self.min_fragment_mass:
            fragment_masses.append(remaining_mass)
        
        return fragment_masses
    
    def calculate_fragment_velocities(self,
                                    fragment_masses: List[float],
                                    fragmentation_energy: float,
                                    parent_velocity: np.ndarray) -> List[np.ndarray]:
        """
        Calculate initial velocities of fragments.
        
        Args:
            fragment_masses: List of fragment masses
            fragmentation_energy: Energy for velocity dispersion
            parent_velocity: Velocity of parent body
            
        Returns:
            List of velocity vectors
        """
        velocities = []
        
        # Total kinetic energy available
        kinetic_energy = fragmentation_energy * self.energy_partition_efficiency
        
        # Conservation of momentum
        total_mass = sum(fragment_masses)
        momentum_per_unit_mass = parent_velocity
        
        for mass in fragment_masses:
            # Base velocity from momentum conservation
            base_velocity = momentum_per_unit_mass.copy()
            
            # Additional velocity from fragmentation energy
            # Smaller fragments get higher velocities
            mass_factor = total_mass / mass
            velocity_magnitude = np.sqrt(2 * kinetic_energy * mass_factor / total_mass)
            
            # Random direction for fragmentation velocity
            random_direction = np.random.normal(0, 1, 3)
            random_direction = random_direction / np.linalg.norm(random_direction)
            
            fragmentation_velocity = velocity_magnitude * random_direction
            
            # Total velocity
            total_velocity = base_velocity + fragmentation_velocity
            velocities.append(total_velocity)
        
        return velocities
    
    def inherit_composition(self,
                          parent_composition: Dict[str, float],
                          fragment_mass: float,
                          total_parent_mass: float) -> Dict[str, float]:
        """
        Determine fragment composition based on parent.
        
        Args:
            parent_composition: Composition of parent body
            fragment_mass: Mass of this fragment
            total_parent_mass: Total mass of parent
            
        Returns:
            Composition dictionary for fragment
        """
        # For now, inherit parent composition with small variations
        composition = parent_composition.copy()
        
        # Add small random variations (±10%)
        for material in composition:
            variation = 0.1 * (2 * np.random.random() - 1)  # ±10%
            composition[material] *= (1 + variation)
            composition[material] = max(0.0, min(1.0, composition[material]))
        
        # Renormalize
        total = sum(composition.values())
        if total > 0:
            composition = {k: v/total for k, v in composition.items()}
        
        return composition
    
    def create_fragmentation_event(self,
                                 parent_body,
                                 time: float,
                                 stress_components: Dict[str, float],
                                 cause: str) -> FragmentationEvent:
        """
        Create a complete fragmentation event.
        
        Args:
            parent_body: The body that is fragmenting
            time: Current simulation time
            stress_components: Dictionary of stress values
            cause: Primary cause of fragmentation
            
        Returns:
            FragmentationEvent with all fragments
        """
        # Calculate total stress
        total_stress = sum(stress_components.values())
        
        # Material strength
        material_strength = self.calculate_material_strength(parent_body.composition)
        
        # Stress ratio
        stress_ratio = total_stress / material_strength
        
        # Fragmentation energy (empirical relationship)
        fragmentation_energy = total_stress * parent_body.volume * 0.01  # J
        
        # Create event
        event = FragmentationEvent(
            parent_id=getattr(parent_body, 'id', 0),
            time=time,
            position=parent_body.position.copy(),
            cause=cause,
            stress_level=total_stress
        )
        
        # Generate fragments
        fragment_masses = self.generate_fragment_masses(
            parent_body.mass, stress_ratio, fragmentation_energy
        )
        
        fragment_velocities = self.calculate_fragment_velocities(
            fragment_masses, fragmentation_energy, parent_body.velocity
        )
        
        # Create fragment objects
        for i, (mass, velocity) in enumerate(zip(fragment_masses, fragment_velocities)):
            # Calculate fragment radius (assuming spherical)
            density = parent_body.density
            radius = (3 * mass / (4 * np.pi * density))**(1/3)
            
            # Inherit composition
            composition = self.inherit_composition(
                parent_body.composition, mass, parent_body.mass
            )
            
            # Small random displacement from parent position
            position_offset = np.random.normal(0, parent_body.radius * 0.1, 3)
            fragment_position = parent_body.position + position_offset
            
            # Create fragment
            fragment = Fragment(
                mass=mass,
                position=fragment_position,
                velocity=velocity,
                radius=radius,
                density=density,
                composition=composition,
                creation_time=time
            )
            
            event.add_fragment(fragment)
        
        # Track energy
        event.total_energy_released = fragmentation_energy
        
        # Store event
        self.fragmentation_events.append(event)
        
        return event
    
    def get_fragmentation_statistics(self) -> Dict:
        """
        Get statistics about all fragmentation events.
        
        Returns:
            Dictionary with fragmentation statistics
        """
        if not self.fragmentation_events:
            return {"total_events": 0}
        
        total_events = len(self.fragmentation_events)
        total_fragments = sum(event.get_fragment_count() for event in self.fragmentation_events)
        
        # Cause breakdown
        causes = [event.cause for event in self.fragmentation_events]
        cause_counts = {cause: causes.count(cause) for cause in set(causes)}
        
        # Mass distribution
        all_masses = []
        for event in self.fragmentation_events:
            all_masses.extend(event.get_mass_distribution())
        
        stats = {
            "total_events": total_events,
            "total_fragments": total_fragments,
            "average_fragments_per_event": total_fragments / total_events,
            "fragmentation_causes": cause_counts,
            "mass_distribution": {
                "largest_fragment": max(all_masses) if all_masses else 0,
                "smallest_fragment": min(all_masses) if all_masses else 0,
                "median_fragment_mass": np.median(all_masses) if all_masses else 0
            }
        }
        
        return stats
