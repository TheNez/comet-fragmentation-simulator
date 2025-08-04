#!/usr/bin/env python3
"""
Focused Shoemaker-Levy 9 fragmentation simulation.
This creates a scenario where we focus on the final approach and fragmentation.
"""

import sys
import numpy as np
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from comet_sim.core import CometSimulator, SimulationConfig
from comet_sim.bodies import CometBody, CometComposition, create_jupiter
from comet_sim.physics import TidalForces

def create_sl9_close_approach():
    """Create SL9 on final approach to Jupiter for fragmentation study."""
    
    # Create a Jupiter-only system (no Sun) to focus on tidal effects
    jupiter_mass = 1.898e27
    jupiter_radius = 7.149e7
    
    jupiter = create_jupiter()
    jupiter.position = np.array([0.0, 0.0, 0.0])  # Put Jupiter at origin
    jupiter.velocity = np.array([0.0, 0.0, 0.0])  # Stationary
    
    # Comet starting at 8 Jupiter radii, approaching to 1.5 Jupiter radii
    initial_distance = 8.0 * jupiter_radius
    perijove_distance = 1.5 * jupiter_radius  # Well inside Roche limit
    
    # Calculate orbital elements for hyperbolic/highly elliptical approach
    G = 6.67430e-11
    
    # Comet composition - very fragile like SL9
    composition = CometComposition(ice_fraction=0.5, rock_fraction=0.3, dust_fraction=0.2)
    
    comet = CometBody(
        name="Shoemaker-Levy 9 Fragment",
        mass=1e11,  # 100 billion kg (fragment size)
        radius=400.0,  # 400 meters radius
        density=300.0,  # Very low density (fluffy)
        composition={'ice': 0.5, 'rock': 0.3, 'dust': 0.2},
        position=np.array([initial_distance, 0.0, 0.0]),
        velocity=np.array([0.0, 5000.0, 0.0]),  # 5 km/s approach velocity
        structural_integrity=5e4  # Very weak (50 kPa)
    )
    
    return comet, [jupiter]

def run_fragmentation_simulation():
    """Run focused fragmentation simulation."""
    
    print("🌌 Shoemaker-Levy 9 Tidal Fragmentation Study")
    print("=" * 50)
    print("Simulating close approach and fragmentation near Jupiter")
    print()
    
    # Create the scenario
    comet, celestial_bodies = create_sl9_close_approach()
    jupiter = celestial_bodies[0]
    
    # High-resolution simulation for fragmentation capture
    config = SimulationConfig(
        dt=300.0,  # 5 minute timesteps
        max_steps=2880,  # 10 days
        adaptive_timestep=True,
        tolerance=1e-12
    )
    
    # Calculate important distances
    jupiter_distance = np.linalg.norm(comet.position - jupiter.position)
    roche_limit = jupiter.roche_limit(comet.density)
    
    print(f"Initial Conditions:")
    print(f"  - Comet mass: {comet.mass:.1e} kg")
    print(f"  - Comet radius: {comet.radius:.0f} m")
    print(f"  - Comet density: {comet.density:.0f} kg/m³")
    print(f"  - Structural integrity: {comet.structural_integrity:.0e} Pa")
    print(f"  - Initial distance: {jupiter_distance/jupiter.radius:.1f} Jupiter radii")
    print(f"  - Roche limit: {roche_limit/jupiter.radius:.1f} Jupiter radii")
    print(f"  - Time to Roche limit: ~{(jupiter_distance - roche_limit)/5000/3600:.1f} hours")
    
    if jupiter_distance > roche_limit:
        print("  ✅ Starting outside Roche limit - fragmentation expected during approach")
    else:
        print("  ⚠️  Already within Roche limit!")
    
    # Run simulation
    simulator = CometSimulator(comet=comet, celestial_bodies=celestial_bodies, config=config)
    
    print(f"\n🚀 Running high-resolution fragmentation simulation...")
    results = simulator.run_simulation()
    
    # Analyze results
    print(f"\n📊 Fragmentation Results:")
    print(f"  - Simulation time: {results['time'][-1]/3600:.1f} hours")
    print(f"  - Data points: {len(results['time'])}")
    
    # Calculate minimum approach distance
    positions = np.array(results['positions'])
    distances = np.linalg.norm(positions - jupiter.position.reshape(1, 3), axis=1)
    min_distance = np.min(distances)
    
    print(f"  - Closest approach: {min_distance/jupiter.radius:.2f} Jupiter radii")
    print(f"  - Closest approach: {min_distance/1000:.0f} km")
    
    # Check for fragmentation
    frag_events = len(results.get('fragmentation_events', []))
    print(f"  - 💥 FRAGMENTATION EVENTS: {frag_events}")
    
    if frag_events > 0:
        first_frag = results['fragmentation_events'][0]
        frag_distance = np.linalg.norm(np.array(first_frag['position']) - jupiter.position)
        print(f"  - First fragmentation at: {frag_distance/jupiter.radius:.2f} Jupiter radii")
        print(f"  - First fragmentation time: {first_frag['time']/3600:.1f} hours")
        print(f"  - 🎯 SUCCESS: Comprehensive fragmentation occurred!")
        
        # Fragment analysis
        total_fragments = sum(event.get('fragments_created', 5) for event in results['fragmentation_events'])
        print(f"  - Total fragments created: {total_fragments}")
    else:
        print(f"  - ❓ No fragmentation detected")
        if min_distance > roche_limit:
            print(f"  - Comet did not penetrate Roche limit deeply enough")
        else:
            print(f"  - Comet reached {min_distance/roche_limit:.2f} × Roche limit")
    
    # Energy conservation check
    if results['energy']:
        energy_error = abs(results['energy'][-1] - results['energy'][0])/abs(results['energy'][0])*100
        print(f"  - Energy conservation: {energy_error:.3f}% error")
    
    return results, simulator

if __name__ == "__main__":
    results, simulator = run_fragmentation_simulation()
    
    print(f"\n💡 Summary:")
    if len(results.get('fragmentation_events', [])) > 0:
        print("✅ Successfully modeled comet fragmentation due to tidal forces!")
        print("This demonstrates how Shoemaker-Levy 9 broke apart approaching Jupiter.")
    else:
        print("⚠️  No fragmentation occurred in this scenario.")
        print("Try adjusting comet properties or approach trajectory.")
    
    print(f"\nSimulation complete! 🌌")
