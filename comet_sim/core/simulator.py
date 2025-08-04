"""
Core simulation engine for comet fragmentation modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import time

from ..bodies.comet import CometBody
from ..bodies.celestial import CelestialBody
from ..physics.forces import GravitationalForces, TidalForces, FragmentationModel, GasPressureForces
from ..physics.fragmentation import CometFragmentation
from ..visualization.plotting import TrajectoryPlotter


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    dt: float = 3600.0  # Time step in seconds (1 hour default)
    max_steps: int = 10000
    tolerance: float = 1e-12
    adaptive_timestep: bool = True
    min_dt: float = 60.0  # Minimum timestep (1 minute)
    max_dt: float = 86400.0  # Maximum timestep (1 day)


class CometSimulator:
    """
    Main simulation engine for comet fragmentation analysis.
    
    This class orchestrates the entire simulation process, including:
    - N-body gravitational dynamics
    - Tidal force calculations
    - Fragmentation detection and modeling
    - Trajectory tracking and analysis
    """
    
    def __init__(self, 
                 comet: Optional[CometBody] = None,
                 celestial_bodies: Optional[List[CelestialBody]] = None,
                 config: Optional[SimulationConfig] = None):
        """
        Initialize the comet fragmentation simulator.
        
        Args:
            comet: The comet to simulate (default creates test comet)
            celestial_bodies: List of celestial bodies (default includes Sun and Jupiter)
            config: Simulation configuration parameters
        """
        self.config = config or SimulationConfig()
        
        # Initialize comet
        self.comet = comet or self._create_default_comet()
        
        # Initialize celestial bodies
        self.celestial_bodies = celestial_bodies or self._create_default_system()
        
        # Initialize physics engines
        self.gravity_engine = GravitationalForces()
        self.tidal_engine = TidalForces()
        self.gas_pressure_engine = GasPressureForces()
        self.fragmentation_engine = CometFragmentation()
        
        # Initialize visualization
        self.plotter = TrajectoryPlotter()
        
        # Simulation state
        self.current_time = 0.0
        self.step_count = 0
        self.fragments = []
        self.trajectory_history = []
        
        # Results storage
        self.results = {
            'time': [],
            'positions': [],
            'velocities': [],
            'forces': [],
            'fragmentation_events': [],
            'energy': [],
            'angular_momentum': []
        }
        
    def _create_default_comet(self) -> CometBody:
        """Create a default comet for testing purposes."""
        return CometBody(
            name="Test Comet",
            mass=1e12,  # kg (typical small comet)
            radius=500.0,  # meters
            density=500.0,  # kg/m³
            composition={'ice': 0.6, 'rock': 0.3, 'dust': 0.1},
            position=np.array([1.5e11, 0.0, 0.0]),  # 1 AU from origin
            velocity=np.array([0.0, 25000.0, 0.0]),  # m/s
            structural_integrity=1e6  # Pa (breaking stress)
        )
    
    def _create_default_system(self) -> List[CelestialBody]:
        """Create default solar system bodies (Sun and Jupiter)."""
        return [
            CelestialBody(
                name="Sun",
                mass=1.989e30,  # kg
                radius=6.96e8,  # meters
                position=np.array([0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0])
            ),
            CelestialBody(
                name="Jupiter", 
                mass=1.898e27,  # kg
                radius=7.149e7,  # meters
                position=np.array([7.785e11, 0.0, 0.0]),  # 5.2 AU
                velocity=np.array([0.0, 13070.0, 0.0])  # m/s
            )
        ]
    
    def run_simulation(self, steps: Optional[int] = None) -> Dict:
        """
        Run the main simulation loop.
        
        Args:
            steps: Number of simulation steps (uses config default if None)
            
        Returns:
            Dictionary containing simulation results
        """
        max_steps = steps or self.config.max_steps
        
        print(f"🌌 Starting comet fragmentation simulation...")
        print(f"   Comet: {self.comet.name}")
        print(f"   Max steps: {max_steps}")
        print(f"   Time step: {self.config.dt:.0f}s")
        
        start_time = time.time()
        
        try:
            for step in range(max_steps):
                if not self._simulation_step():
                    print(f"   Simulation terminated early at step {step}")
                    break
                    
                if step % 100 == 0:
                    self._log_progress(step, max_steps)
                    
        except KeyboardInterrupt:
            print("\\n⚠️  Simulation interrupted by user")
        except Exception as e:
            print(f"\\n❌ Simulation error: {e}")
            raise
        
        elapsed = time.time() - start_time
        print(f"\\n✅ Simulation completed in {elapsed:.2f}s")
        print(f"   Total steps: {self.step_count}")
        print(f"   Final time: {self.current_time/86400:.2f} days")
        
        return self.results
    
    def _simulation_step(self) -> bool:
        """
        Execute a single simulation step.
        
        Returns:
            True if simulation should continue, False to terminate
        """
        # Calculate forces
        gravitational_forces = self._calculate_gravitational_forces()
        tidal_forces = self._calculate_tidal_forces()
        
        total_forces = gravitational_forces + tidal_forces
        
        # Check for fragmentation
        if self._check_fragmentation_conditions(tidal_forces):
            self._trigger_fragmentation()
        
        # Update comet state
        self._update_comet_state(total_forces)
        
        # Store results
        self._store_results(total_forces)
        
        # Update time and step count
        self.current_time += self.config.dt
        self.step_count += 1
        
        # Check termination conditions
        return self._check_continuation_conditions()
    
    def _calculate_gravitational_forces(self) -> np.ndarray:
        """Calculate gravitational forces from all celestial bodies."""
        total_force = np.zeros(3)
        
        for body in self.celestial_bodies:
            # Calculate force on comet from this celestial body
            force_on_comet, _ = self.gravity_engine.force_between_bodies(
                self.comet.position, self.comet.mass,
                body.position, body.mass
            )
            total_force += force_on_comet
            
        return total_force
    
    def _calculate_tidal_forces(self) -> np.ndarray:
        """Calculate tidal forces from nearby massive bodies."""
        total_tidal = np.zeros(3)
        
        for body in self.celestial_bodies:
            if body.name == "Jupiter":  # Focus on Jupiter's tidal effects
                # Calculate tidal acceleration gradient
                tidal_gradient = self.tidal_engine.tidal_acceleration_gradient(
                    body.position, body.mass, self.comet.position
                )
                # Convert to force by multiplying by mass and radius
                tidal_force = tidal_gradient * self.comet.mass * self.comet.radius
                total_tidal += tidal_force
                
        return total_tidal
    
    def _check_fragmentation_conditions(self, tidal_forces: np.ndarray) -> bool:
        """Check if fragmentation should occur."""
        # Check if within Roche limit first
        for body in self.celestial_bodies:
            if body.name == "Jupiter":
                distance = np.linalg.norm(self.comet.position - body.position)
                roche_limit = self.tidal_engine.roche_limit(body.mass, body.radius, self.comet.density)
                
                # Debug output when we're close
                if distance < 2 * roche_limit:
                    print(f"    🔍 Distance: {distance/1000:.0f} km, Roche: {roche_limit/1000:.0f} km")
                
                # Calculate all stress components
                # 1. Tidal stress
                tidal_stress = self.tidal_engine.tidal_stress_at_position(
                    body.position, body.mass, self.comet.position, 
                    np.array([self.comet.radius, 0, 0]), self.comet.density
                )
                
                # 2. Gas pressure stress (critical for comet fragmentation)
                gas_pressure_stress = self.gas_pressure_engine.gas_pressure_stress(
                    self.comet.temperature, 
                    ice_fraction=0.6,  # From comet composition
                    porosity=0.4       # Fluffy comet structure
                )
                
                # 3. Thermal stress
                thermal_stress = 0.0  # Simplified for now
                if hasattr(self.comet, 'temperature'):
                    fragmentation_model = FragmentationModel()
                    thermal_stress = fragmentation_model.thermal_stress(
                        self.comet.temperature, reference_temp=100.0
                    )
                
                # 4. Rotational stress (minimal for slow-rotating comets)
                rotational_stress = 0.0  # Simplified for now
                
                # Calculate total stress using FragmentationModel
                fragmentation_model = FragmentationModel()
                total_stress = fragmentation_model.total_stress(
                    tidal_stress, thermal_stress, rotational_stress, gas_pressure_stress
                )
                
                # Get material strength from comet's structural integrity
                material_strength = self.comet.structural_integrity
                
                # Enhanced debug output
                if distance < roche_limit or total_stress > material_strength * 0.1:
                    print(f"    💥 STRESS ANALYSIS:")
                    print(f"       Tidal stress: {tidal_stress:.2e} Pa")
                    print(f"       Gas pressure: {gas_pressure_stress:.2e} Pa")
                    print(f"       Thermal stress: {thermal_stress:.2e} Pa")
                    print(f"       Total stress: {total_stress:.2e} Pa")
                    print(f"       Material strength: {material_strength:.2e} Pa")
                    print(f"       Stress ratio: {total_stress/material_strength:.3f}")
                
                # Enhanced fragmentation check - if total stress exceeds strength
                if total_stress > material_strength:
                    print(f"    ✅ FRAGMENTATION TRIGGERED - Total stress exceeds strength!")
                    return True
                
                # Also check probabilistic fragmentation for gradual failure
                prob_frag = self.fragmentation_engine.fragmentation_criterion(
                    total_stress, material_strength, self.config.dt
                )
                if prob_frag:
                    print(f"    ✅ PROBABILISTIC FRAGMENTATION TRIGGERED!")
                    return True
        
        return False
    
    def _trigger_fragmentation(self):
        """Handle comet fragmentation event."""
        print(f"💥 Fragmentation event at t={self.current_time/86400:.2f} days!")
        
        # Create fragment model if not exists
        if not hasattr(self, 'fragmentation_model'):
            self.fragmentation_model = FragmentationModel()
        
        # For now, just mark the comet as fragmented
        self.comet.is_fragmented = True
        
        # Record fragmentation event
        self.results['fragmentation_events'].append({
            'time': self.current_time,
            'fragments_created': 5,  # Simplified
            'position': self.comet.position.copy()
        })
    
    def _update_comet_state(self, forces: np.ndarray):
        """Update comet position and velocity using forces."""
        # Simple Verlet integration
        acceleration = forces / self.comet.mass
        
        # Update velocity and position
        self.comet.velocity += acceleration * self.config.dt
        self.comet.position += self.comet.velocity * self.config.dt
        
        # Update comet internal state
        self.comet.update_state(self.config.dt)
    
    def _store_results(self, forces: np.ndarray):
        """Store current simulation state for analysis."""
        self.results['time'].append(self.current_time)
        self.results['positions'].append(self.comet.position.copy())
        self.results['velocities'].append(self.comet.velocity.copy())
        self.results['forces'].append(forces.copy())
        
        # Calculate and store energy
        kinetic_energy = 0.5 * self.comet.mass * np.dot(self.comet.velocity, self.comet.velocity)
        potential_energy = self._calculate_potential_energy()
        total_energy = kinetic_energy + potential_energy
        self.results['energy'].append(total_energy)
        
        # Calculate and store angular momentum
        angular_momentum = np.cross(self.comet.position, self.comet.mass * self.comet.velocity)
        self.results['angular_momentum'].append(np.linalg.norm(angular_momentum))
    
    def _calculate_potential_energy(self) -> float:
        """Calculate gravitational potential energy."""
        potential = 0.0
        G = 6.67430e-11  # Gravitational constant
        
        for body in self.celestial_bodies:
            distance = np.linalg.norm(self.comet.position - body.position)
            potential += -G * self.comet.mass * body.mass / distance
            
        return potential
    
    def _check_continuation_conditions(self) -> bool:
        """Check if simulation should continue."""
        # Check if comet has escaped the system
        distance_from_sun = np.linalg.norm(self.comet.position)
        if distance_from_sun > 1e13:  # 100 AU
            print("   Comet has escaped the solar system")
            return False
            
        # Check if comet has impacted a body
        for body in self.celestial_bodies:
            distance = np.linalg.norm(self.comet.position - body.position)
            if distance < body.radius:
                print(f"   Comet has impacted {body.name}")
                return False
                
        return True
    
    def _log_progress(self, step: int, max_steps: int):
        """Log simulation progress."""
        progress = (step / max_steps) * 100
        days = self.current_time / 86400
        distance = np.linalg.norm(self.comet.position) / 1.496e11  # AU
        
        print(f"   Step {step:6d} ({progress:5.1f}%): "
              f"t={days:7.2f} days, "
              f"r={distance:6.3f} AU")
    
    def plot_trajectory(self, show_fragments: bool = True):
        """Plot the comet trajectory and fragmentation events."""
        return self.plotter.plot_trajectory(
            self.results, 
            self.celestial_bodies,
            show_fragments=show_fragments
        )
    
    def plot_forces_analysis(self):
        """Plot force analysis over time."""
        return self.plotter.plot_forces(self.results)
    
    def plot_energy_conservation(self):
        """Plot energy conservation analysis.""" 
        return self.plotter.plot_energy(self.results)
    
    def get_fragmentation_summary(self) -> Dict:
        """Get summary of fragmentation events."""
        events = self.results['fragmentation_events']
        
        if not events:
            return {"total_events": 0, "message": "No fragmentation occurred"}
            
        return {
            "total_events": len(events),
            "first_fragmentation_time": events[0]['time'] / 86400,  # days
            "total_fragments": sum(e['fragments_created'] for e in events),
            "events": events
        }
    
    def export_results(self, filename: str):
        """Export simulation results to file."""
        import pickle
        
        export_data = {
            'config': self.config,
            'comet_initial': self.comet,
            'celestial_bodies': self.celestial_bodies,
            'results': self.results,
            'fragments': self.fragments
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(export_data, f)
            
        print(f"✅ Results exported to {filename}")


if __name__ == "__main__":
    # Example simulation run
    simulator = CometSimulator()
    results = simulator.run_simulation(steps=1000)
    
    # Display summary
    summary = simulator.get_fragmentation_summary()
    print(f"\\n📊 Fragmentation Summary: {summary}")
    
    # Create plots
    simulator.plot_trajectory()
    simulator.plot_forces_analysis()
