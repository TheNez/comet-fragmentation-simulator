#!/usr/bin/env python3
"""
Shoemaker-Levy 9 Comprehensive Fragmentation Analysis with Advanced Visualization
=================================================================================

This script performs a detailed simulation of comet SL9's fragmentation and generates
comprehensive interactive visualizations showing the trajectory, force analysis,
and fragmentation timeline.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import simulation components
from comet_sim.core import CometSimulator, SimulationConfig
from comet_sim.bodies import CometBody, CometComposition, create_jupiter
from comet_sim.physics import TidalForces

def create_sl9_close_approach():
    """Create SL9 on final approach to Jupiter for fragmentation study."""
    
    # Create Jupiter
    jupiter = create_jupiter()
    jupiter.position = np.array([0.0, 0.0, 0.0])
    jupiter.velocity = np.array([0.0, 0.0, 0.0])
    
    # Create comet composition (icy, fluffy structure like SL9)
    composition = CometComposition(ice_fraction=0.7, rock_fraction=0.2, dust_fraction=0.1)
    
    # SL9 physical parameters
    comet_mass = 1e11  # kg (estimated)
    comet_radius = 400  # m (estimated)
    
    # Create comet at close approach distance
    initial_distance = 8.0 * jupiter.radius  # Start outside Roche limit
    approach_velocity = 5000.0  # m/s (5 km/s approach)
    
    position = np.array([initial_distance, 0.0, 0.0])
    velocity = np.array([0.0, approach_velocity, 0.0])
    
    comet = CometBody(
        name="Shoemaker-Levy 9 Fragment",
        mass=comet_mass,
        radius=comet_radius,
        density=300.0,  # kg/m³ - very low density (fluffy)
        composition={'ice': 0.7, 'rock': 0.2, 'dust': 0.1},
        position=position,
        velocity=velocity,
        structural_integrity=5e4  # Pa - very weak structure
    )
    
    return comet, [jupiter]

def main():
    """Run complete SL9 fragmentation analysis with visualization."""
    
    print("🌌 Shoemaker-Levy 9 Comprehensive Fragmentation Analysis")
    print("=" * 65)
    print("Simulating fragmentation with advanced visualization system")
    print()
    
    # Create simulation objects
    comet, celestial_bodies = create_sl9_close_approach()
    jupiter = celestial_bodies[0]
    
    print("Initial Conditions:")
    print(f"  - Comet mass: {comet.mass:.1e} kg")
    print(f"  - Comet radius: {comet.radius} m")
    print(f"  - Structural integrity: {comet.structural_integrity:.0e} Pa")
    print(f"  - Density: {comet.density:.0f} kg/m³")
    
    initial_distance = np.linalg.norm(comet.position)
    roche_limit = jupiter.roche_limit(comet.density)
    
    print(f"  - Initial distance: {initial_distance/jupiter.radius:.1f} Jupiter radii")
    print(f"  - Roche limit: {roche_limit/jupiter.radius:.1f} Jupiter radii")
    
    if initial_distance > roche_limit:
        print("  ✅ Starting outside Roche limit - fragmentation expected during approach")
    else:
        print("  ⚠️ Starting inside Roche limit - immediate fragmentation expected")
    print()
    
    # Create simulation configuration
    config = SimulationConfig(
        dt=300.0,  # 5 minutes
        max_steps=2880,  # 10 days max
        adaptive_timestep=True,
        tolerance=1e-12
    )
    
    # Initialize physics
    tidal_forces = TidalForces(jupiter_mass=jupiter.mass, jupiter_radius=jupiter.radius)
    
    # Create and configure simulator
    simulator = CometSimulator(comet=comet, celestial_bodies=celestial_bodies, config=config)
    
    print("🚀 Running high-resolution fragmentation simulation...")
    
    # Run simulation
    results = simulator.run_simulation()
    
    print(f"✅ Simulation completed")
    print(f"   Total steps: {len(results['time'])}")
    if results['time']:
        final_time = results['time'][-1] / 3600 / 24  # Convert to days
        print(f"   Final time: {final_time:.2f} days")
    print()
    
    # Process results for visualization
    trajectory_data = []
    fragmentation_events = []
    
    # Convert simulation results to visualization format
    for i, time in enumerate(results['time']):
        if i < len(results['position']):
            point = {
                'time': time / 3600 / 24,  # Convert to days
                'position': results['position'][i],
                'velocity': results['velocity'][i] if i < len(results['velocity']) else np.zeros(3),
                'total_stress': results.get('stress', [0]*len(results['time']))[i],
                'tidal_stress': results.get('tidal_stress', [0]*len(results['time']))[i],
                'thermal_stress': 0,  # Placeholder
                'gas_pressure': 0,    # Placeholder
                'material_strength': comet.structural_integrity
            }
            trajectory_data.append(point)
    
    # Process fragmentation events from results
    if 'fragmentation_events' in results and results['fragmentation_events']:
        for event in results['fragmentation_events']:
            frag_event = {
                'time': event['time'] / 3600 / 24,  # Convert to days
                'position': event['position'],
                'fragments_created': event.get('fragments_created', 5)
            }
            fragmentation_events.append(frag_event)
    
    print("📊 Fragmentation Results:")
    if trajectory_data:
        print(f"  - Simulation time: {trajectory_data[-1]['time']*24:.1f} hours")
        print(f"  - Data points: {len(trajectory_data)}")
        
        distances = [np.linalg.norm(point['position']) for point in trajectory_data]
        closest_distance = min(distances)
        print(f"  - Closest approach: {closest_distance/jupiter.radius:.2f} Jupiter radii")
        print(f"  - Closest approach: {closest_distance/1000:.0f} km")
    
    print(f"  - 💥 FRAGMENTATION EVENTS: {len(fragmentation_events)}")
    
    if fragmentation_events:
        first_frag = fragmentation_events[0]
        first_distance = np.linalg.norm(first_frag['position'])
        first_time = first_frag['time']
        
        print(f"  - First fragmentation at: {first_distance/jupiter.radius:.2f} Jupiter radii")
        print(f"  - First fragmentation time: {first_time*24:.1f} hours")
        print("  - 🎯 SUCCESS: Comprehensive fragmentation occurred!")
        
        total_fragments = sum(event.get('fragments_created', 5) for event in fragmentation_events)
        print(f"  - Total fragments created: {total_fragments}")
    else:
        print("  - ❌ No fragmentation events detected")
        print("  - Consider adjusting material strength or force parameters")
    
    print()
    
    # Generate visualizations
    if len(trajectory_data) > 0:
        print("🎨 Generating interactive visualizations...")
        
        try:
            # Import visualization module
            from comet_sim.visualization.plotter import create_comprehensive_visualization
            
            # Create comprehensive visualization suite
            trajectory_fig, force_fig, timeline_fig, report = create_comprehensive_visualization(
                trajectory_data=trajectory_data,
                fragmentation_events=fragmentation_events,
                save_html=True
            )
            
            print("✅ Visualization suite generated successfully!")
            print()
            
            # Display statistics report
            print(report)
            
            print()
            print("🌐 Open the HTML files in your browser to explore:")
            print("   • comet_trajectory_3d.html - Interactive 3D trajectory")
            print("   • force_analysis_dashboard.html - Comprehensive force analysis")
            print("   • fragmentation_timeline.html - Fragmentation event timeline")
            
        except ImportError as e:
            print(f"⚠️ Visualization requires additional packages: {e}")
            print("   Install with: pip install plotly matplotlib")
            
            # Fallback: basic analysis
            analyze_results(trajectory_data, fragmentation_events, jupiter.radius)
    
    print()
    print("💡 Summary:")
    if fragmentation_events:
        print("✅ Successfully modeled comet fragmentation due to comprehensive forces!")
        print("This demonstrates how Shoemaker-Levy 9 broke apart approaching Jupiter.")
        print("The simulation shows realistic force hierarchies and fragmentation cascades.")
    else:
        print("❌ Fragmentation modeling needs adjustment.")
        print("Consider stronger thermal/gas forces or weaker material strength.")
    
    print()
    print("Simulation complete! 🌌")

def analyze_results(trajectory_data, fragmentation_events, jupiter_radius):
    """Basic analysis fallback when visualization packages not available."""
    
    print("📈 BASIC ANALYSIS (fallback):")
    print()
    
    if not trajectory_data:
        print("❌ No trajectory data available")
        return
    
    # Distance analysis
    distances = [np.linalg.norm(point['position']) for point in trajectory_data]
    times = [point['time'] for point in trajectory_data]
    
    print(f"📍 Trajectory Analysis:")
    print(f"   • Data points: {len(trajectory_data)}")
    print(f"   • Time span: {max(times):.3f} days")
    print(f"   • Closest approach: {min(distances)/1000:.0f} km")
    print(f"   • Starting distance: {max(distances)/1000:.0f} km")
    print()
    
    # Fragmentation analysis
    if fragmentation_events:
        print(f"💥 Fragmentation Analysis:")
        print(f"   • Total events: {len(fragmentation_events)}")
        
        frag_times = [event['time'] for event in fragmentation_events]
        frag_distances = [np.linalg.norm(event['position']) for event in fragmentation_events]
        
        print(f"   • First fragmentation: {min(frag_times):.3f} days")
        print(f"   • Last fragmentation: {max(frag_times):.3f} days")
        print(f"   • Fragmentation range: {min(frag_distances)/1000:.0f} - {max(frag_distances)/1000:.0f} km")
        
        # Fragment count
        total_fragments = sum(event.get('fragments_created', 5) for event in fragmentation_events)
        print(f"   • Total fragments: {total_fragments}")
        print(f"   • Average per event: {total_fragments/len(fragmentation_events):.1f}")
    
    print()
    
    # Force analysis (if available)
    if trajectory_data and 'total_stress' in trajectory_data[0]:
        stresses = [point.get('total_stress', 0) for point in trajectory_data if point.get('total_stress', 0) > 0]
        if stresses:
            print(f"💪 Force Analysis:")
            print(f"   • Max stress: {max(stresses):.2e} Pa")
            print(f"   • Mean stress: {np.mean(stresses):.2e} Pa")
            
            # Material strength comparison
            material_strength = 5e4  # Pa
            max_ratio = max(stresses) / material_strength
            print(f"   • Max stress ratio: {max_ratio:.1f}x material strength")
            print()

if __name__ == "__main__":
    main()
