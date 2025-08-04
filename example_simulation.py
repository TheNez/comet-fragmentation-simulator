"""
Example usage of the comet fragmentation simulation.

This script demonstrates how to set up and run a comet simulation,
including fragmentation analysis and visualization.
"""

import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from comet_sim.core import CometSimulator, SimulationConfig
from comet_sim.bodies import CometBody, CometComposition, create_jupiter, create_sun
from comet_sim.visualization import TrajectoryPlotter


def create_shoemaker_levy_9():
    """Create Comet Shoemaker-Levy 9 with realistic initial conditions."""
    
    # Comet composition (typical Jupiter-family comet)
    composition = CometComposition(
        ice_fraction=0.6,    # 60% water ice (Jupiter-family comets have less ice)
        rock_fraction=0.3,   # 30% rocky material
        dust_fraction=0.1    # 10% dust
    )
    
    # Shoemaker-Levy 9 realistic bound orbit around Jupiter
    # Set up a bound elliptical orbit that will make multiple passes through Roche limit
    jupiter_distance = 5.2 * 1.496e11  # Jupiter's orbital distance from Sun
    jupiter_radius = 7.149e7  # Jupiter's radius in meters
    jupiter_mass = 1.898e27  # Jupiter's mass
    
    # Orbital parameters for elliptical orbit around Jupiter
    # Apojove: 15 Jupiter radii, Perijove: 1.8 Jupiter radii (inside Roche limit)
    apojove_distance = 15.0 * jupiter_radius  # Far point
    perijove_distance = 1.8 * jupiter_radius  # Close point (inside Roche limit)
    semi_major_axis = (apojove_distance + perijove_distance) / 2
    
    # Start at apojove
    initial_position = np.array([
        jupiter_distance - apojove_distance,  # X: at apojove
        0.0,  # Y: in orbital plane
        apojove_distance * 0.02   # Z: slight inclination (1°)
    ])
    
    # Calculate orbital velocity at apojove for elliptical orbit
    # Using vis-viva equation: v² = GM(2/r - 1/a)
    G = 6.67430e-11
    velocity_at_apojove = np.sqrt(jupiter_mass * G * (2/apojove_distance - 1/semi_major_axis))
    
    initial_velocity = np.array([
        0.0,  # X component: 0 at apojove
        velocity_at_apojove * 0.98,  # Y: orbital velocity (slightly reduced for capture)
        velocity_at_apojove * 0.01   # Z: slight inclination
    ])
    
    print(f"    Orbital setup:")
    print(f"    - Apojove: {apojove_distance/jupiter_radius:.1f} Jupiter radii")
    print(f"    - Perijove: {perijove_distance/jupiter_radius:.1f} Jupiter radii") 
    print(f"    - Orbital velocity at apojove: {velocity_at_apojove/1000:.1f} km/s")
    
    # Physical properties of Shoemaker-Levy 9 (before fragmentation)
    # Estimated original size: 1-2 km diameter
    comet_radius = 750.0  # meters (1.5 km diameter)
    comet_density = 400   # kg/m³ (low density, fluffy comet)
    comet_mass = (4/3) * np.pi * comet_radius**3 * comet_density
    
    # Create comet with weaker structure (SL9 was fragile)
    comet = CometBody(
        name="Shoemaker-Levy 9",
        mass=comet_mass,
        radius=comet_radius,
        density=comet_density,
        composition={
            'ice': composition.ice_fraction,
            'rock': composition.rock_fraction,
            'dust': composition.dust_fraction
        },
        position=initial_position,
        velocity=initial_velocity,
        structural_integrity=2e5  # Pa, weaker than typical comet (SL9 was fragile)
    )
    
    return comet


def run_shoemaker_levy_simulation():
    """Run Shoemaker-Levy 9 simulation to observe fragmentation."""
    
    print("Creating Shoemaker-Levy 9 simulation...")
    
    # Create simulation configuration - shorter time for one orbital period
    config = SimulationConfig(
        dt=900.0,            # 15 minute time steps for high resolution fragmentation
        max_steps=5760,      # 60 days simulation (enough for one Jupiter orbit)
        adaptive_timestep=True,
        tolerance=1e-12
    )
    
    # Create celestial bodies
    sun = create_sun()
    jupiter = create_jupiter(orbital_distance=5.2)  # Jupiter at 5.2 AU
    celestial_bodies = [sun, jupiter]
    
    # Create Shoemaker-Levy 9
    comet = create_shoemaker_levy_9()
    
    # Initialize simulator
    simulator = CometSimulator(
        comet=comet,
        celestial_bodies=celestial_bodies,
        config=config
    )
    
    print(f"Shoemaker-Levy 9 simulation setup complete:")
    print(f"  - Time step: {config.dt/60:.1f} minutes")
    print(f"  - Maximum steps: {config.max_steps}")
    print(f"  - Celestial bodies: {len(celestial_bodies)}")
    print(f"  - Comet mass: {comet.mass:.2e} kg")
    print(f"  - Comet radius: {comet.radius:.0f} meters") 
    print(f"  - Structural integrity: {comet.structural_integrity:.1e} Pa (fragile)")
    
    # Distance analysis
    jupiter_distance_km = np.linalg.norm(comet.position - jupiter.position) / 1000
    jupiter_radii = jupiter_distance_km / (jupiter.radius / 1000)
    print(f"  - Distance to Jupiter: {jupiter_distance_km:.0f} km ({jupiter_radii:.1f} Jupiter radii)")
    print(f"  - Distance to Sun: {np.linalg.norm(comet.position)/1.496e11:.2f} AU")
    
    # Calculate Roche limit 
    jupiter_roche_limit = jupiter.roche_limit(comet.density)
    roche_radii = jupiter_roche_limit / jupiter.radius
    print(f"  - Jupiter's Roche limit: {jupiter_roche_limit/1000:.0f} km ({roche_radii:.1f} Jupiter radii)")
    
    current_jupiter_distance = np.linalg.norm(comet.position - jupiter.position)
    if current_jupiter_distance < jupiter_roche_limit:
        print("  ⚠️  COMET IS WITHIN ROCHE LIMIT - FRAGMENTATION IMMINENT!")
    elif current_jupiter_distance < jupiter_roche_limit * 1.5:
        print("  ⚡ COMET IS APPROACHING ROCHE LIMIT - TIDAL STRESS INCREASING!")
    else:
        print("  ✅ Comet is outside immediate tidal danger zone")
    
    # Run simulation
    print("\n🚀 Running Shoemaker-Levy 9 simulation...")
    print("Monitoring approach to Jupiter and fragmentation...")
    try:
        results = simulator.run_simulation()
        print("Simulation completed successfully!")
        
        # Print results summary with focus on fragmentation
        print(f"\n📊 Shoemaker-Levy 9 Results Summary:")
        print(f"  - Simulation time: {results['time'][-1]/(24*3600):.1f} days")
        print(f"  - Data points: {len(results['time'])}")
        print(f"  - Distance traveled: {np.linalg.norm(np.array(results['positions'][-1]) - np.array(results['positions'][0]))/1.496e11:.3f} AU")
        print(f"  - Final distance to Jupiter: {np.linalg.norm(np.array(results['positions'][-1]) - jupiter.position)/1.496e11:.3f} AU")
        
        # Fragmentation analysis
        frag_events = len(results.get('fragmentation_events', []))
        print(f"  - 💥 FRAGMENTATION EVENTS: {frag_events}")
        
        if frag_events > 0:
            first_frag_time = results['fragmentation_events'][0]['time'] / (24*3600)
            print(f"  - First fragmentation at: {first_frag_time:.2f} days")
            print("  - 🎯 SUCCESS: Comet fragmented as expected!")
        else:
            print("  - ❓ No fragmentation occurred - may need closer approach to Jupiter")
        
        if results['energy']:
            energy_conservation = abs(results['energy'][-1] - results['energy'][0])/abs(results['energy'][0])*100
            print(f"  - Energy conservation: {energy_conservation:.3f}% error")
        
        return results, simulator
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None, simulator


def visualize_results(results, simulator):
    """Create visualizations of simulation results."""
    
    if results is None:
        print("No results to visualize")
        return
    
    print("\nCreating visualizations...")
    
    try:
        # Create 3D trajectory plot
        plotter = TrajectoryPlotter(backend='plotly')
        plotter.create_figure("Comet Fragmentation Simulation")
        
        # Add celestial bodies
        for body in simulator.celestial_bodies:
            plotter.add_celestial_body(body.position, body.name)
        
        # Add comet trajectory
        positions = np.array(results['comet_positions'])
        times = np.array(results['times'])
        plotter.add_comet_trajectory(positions, times, name="Comet Trajectory")
        
        # Add fragmentation events
        for i, event in enumerate(results.get('fragmentation_events', [])):
            plotter.add_fragmentation_event(
                event.position, event.time, event.fragments, event_id=i
            )
        
        # Add Roche limit for Jupiter
        jupiter = next(body for body in simulator.celestial_bodies if body.name == "Jupiter")
        roche_distance = jupiter.roche_limit(500)  # Assuming comet density 500 kg/m³
        plotter.add_roche_limit(jupiter.position, roche_distance, "Jupiter Roche Limit")
        
        # Set view
        plotter.set_equal_aspect(range_au=8)
        
        # Save and show
        plotter.save("comet_simulation_3d")
        print("3D visualization saved as 'comet_simulation_3d.html'")
        
        # Show plot
        plotter.show()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("This might be due to missing plotly/matplotlib packages")


def analyze_results(results):
    """Perform analysis of simulation results."""
    
    if results is None:
        return
    
    print("\nAnalyzing results...")
    
    # Time series analysis
    times = np.array(results['time'])
    positions = np.array(results['positions'])
    velocities = np.array(results['velocities'])
    energies = np.array(results['energy'])
    
    # Calculate derived quantities
    distances_from_sun = np.linalg.norm(positions, axis=1)
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Print analysis
    print(f"Trajectory Analysis:")
    print(f"  - Minimum distance from Sun: {distances_from_sun.min()/1.496e11:.3f} AU")
    print(f"  - Maximum distance from Sun: {distances_from_sun.max()/1.496e11:.3f} AU")
    print(f"  - Maximum speed: {speeds.max()/1000:.1f} km/s")
    if energies.size > 0:
        print(f"  - Energy conservation: {abs(energies[-1] - energies[0])/abs(energies[0])*100:.6f}% change")
    
    # Fragmentation analysis
    fragmentation_events = results.get('fragmentation_events', [])
    if fragmentation_events:
        print(f"\nFragmentation Analysis:")
        print(f"  - Number of events: {len(fragmentation_events)}")
        
        total_fragments = sum(event.get('fragments_created', 5) for event in fragmentation_events)
        print(f"  - Total fragments created: {total_fragments}")
        
        # Event timing
        event_times = [event['time']/(24*3600) for event in fragmentation_events]
        print(f"  - First event at: {min(event_times):.1f} days")
        print(f"  - Last event at: {max(event_times):.1f} days")
        
        # Fragmentation summary (simplified since we don't track causes in current implementation)
        print(f"  - Fragmentation occurred throughout approach")


def main():
    """Main function to run the Shoemaker-Levy 9 example."""
    
    print("🌌 Comet Shoemaker-Levy 9 Fragmentation Simulation")
    print("=" * 55)
    print("Recreating the famous comet that broke apart near Jupiter")
    print()
    
    # Run simulation
    results, simulator = run_shoemaker_levy_simulation()
    
    if results is not None:
        # Analyze results
        analyze_results(results)
        
        # Create visualizations
        visualize_results(results, simulator)
        
        print("\nSimulation example completed!")
        print("Check the generated files for detailed results.")
    else:
        print("Simulation failed - check error messages above")


if __name__ == "__main__":
    main()
