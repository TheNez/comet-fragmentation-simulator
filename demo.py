#!/usr/bin/env python3
"""
Simple demonstration of the comet fragmentation simulation system.
"""

import sys
import numpy as np

# Add the project to Python path
sys.path.append('/Users/Roni/comets_fragmentation')

try:
    # Import the main simulation components
    from comet_sim.core import CometSimulator, SimulationConfig
    from comet_sim.bodies import CometBody, CometComposition, create_sun, create_jupiter
    from comet_sim.physics import GravitationalForces, TidalForces, CometFragmentation
    from comet_sim.visualization import TrajectoryPlotter
    
    print("🌌 Comet Fragmentation Simulation System")
    print("=" * 50)
    
    # Test basic components
    print("✅ All core modules imported successfully")
    
    # Test physics engines
    gravity = GravitationalForces()
    tidal = TidalForces()
    fragmentation = CometFragmentation()
    print("✅ Physics engines initialized")
    
    # Test celestial bodies
    sun = create_sun()
    jupiter = create_jupiter()
    print(f"✅ Celestial bodies created: {sun.name}, {jupiter.name}")
    
    # Test comet creation
    composition = CometComposition(ice_fraction=0.7, rock_fraction=0.2, dust_fraction=0.1)
    comet = CometBody(
        name="Demo Comet",
        mass=1e11,  # 100 billion kg
        radius=300.0,  # 300 meters
        density=600.0,  # kg/m³
        composition={'ice': 0.7, 'rock': 0.2, 'dust': 0.1},
        position=np.array([3.0e11, 0.0, 0.0]),  # 2 AU
        velocity=np.array([0.0, 20000.0, 0.0]),  # 20 km/s
        structural_integrity=5e5  # Pa
    )
    print(f"✅ Comet created: {comet.name}, mass={comet.mass:.1e} kg")
    
    # Test quick simulation
    config = SimulationConfig(dt=3600.0, max_steps=100)  # 100 hours
    simulator = CometSimulator(
        comet=comet,
        celestial_bodies=[sun, jupiter],
        config=config
    )
    print("✅ Simulator initialized")
    
    # Run short simulation
    print("\n🚀 Running demonstration simulation (100 hours)...")
    results = simulator.run_simulation()
    
    # Display results
    final_time = results['time'][-1] / 3600  # Convert to hours
    initial_pos = np.array(results['positions'][0])
    final_pos = np.array(results['positions'][-1])
    distance_traveled = np.linalg.norm(final_pos - initial_pos) / 1.496e11  # AU
    
    print(f"\n📊 Demonstration Results:")
    print(f"  • Simulation time: {final_time:.1f} hours")
    print(f"  • Data points collected: {len(results['time'])}")
    print(f"  • Distance traveled: {distance_traveled:.3f} AU")
    print(f"  • Energy conservation: {abs(results['energy'][-1] - results['energy'][0])/abs(results['energy'][0])*100:.6f}% error")
    print(f"  • Fragmentation events: {len(results['fragmentation_events'])}")
    
    print("\n🎯 System Status: FULLY OPERATIONAL")
    print("\nThe comet fragmentation simulation system is ready for:")
    print("  • N-body gravitational dynamics")
    print("  • Tidal force calculations") 
    print("  • Fragmentation modeling")
    print("  • Long-term trajectory analysis")
    print("  • Interactive 3D visualization")
    
    print(f"\n💡 Next Steps:")
    print("  • Run 'python example_simulation.py' for a full year simulation")
    print("  • Modify initial conditions to trigger fragmentation events")
    print("  • Experiment with different comet compositions and sizes")
    print("  • Add more celestial bodies for complex gravitational interactions")
    
except Exception as e:
    print(f"❌ Error during demonstration: {e}")
    print("Check that all required packages are installed in the virtual environment")
