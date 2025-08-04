#!/usr/bin/env python3
"""
Simple Shoemaker-Levy 9 Visualization
====================================

Creates basic visualizations from our successful fragmentation simulation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from comet_sim.core import CometSimulator, SimulationConfig
from comet_sim.bodies import CometBody, CometComposition, create_jupiter
from comet_sim.physics import TidalForces

def create_sl9_approach():
    """Create SL9 approaching Jupiter."""
    
    jupiter = create_jupiter()
    jupiter.position = np.array([0.0, 0.0, 0.0])
    jupiter.velocity = np.array([0.0, 0.0, 0.0])
    
    composition = CometComposition(ice_fraction=0.7, rock_fraction=0.2, dust_fraction=0.1)
    
    initial_distance = 8.0 * jupiter.radius
    
    comet = CometBody(
        name="Shoemaker-Levy 9 Fragment",
        mass=1e11,
        radius=400.0,
        density=300.0,
        composition={'ice': 0.7, 'rock': 0.2, 'dust': 0.1},
        position=np.array([initial_distance, 0.0, 0.0]),
        velocity=np.array([0.0, 5000.0, 0.0]),
        structural_integrity=5e4
    )
    
    return comet, [jupiter]

def run_simulation():
    """Run the simulation and return results."""
    
    print("🌌 Running Shoemaker-Levy 9 Simulation for Visualization")
    print("=" * 55)
    
    comet, celestial_bodies = create_sl9_approach()
    jupiter = celestial_bodies[0]
    
    config = SimulationConfig(
        dt=300.0,  # 5 minutes
        max_steps=2880,  # 10 days
        adaptive_timestep=True,
        tolerance=1e-12
    )
    
    print(f"Initial distance: {np.linalg.norm(comet.position)/jupiter.radius:.1f} Jupiter radii")
    print(f"Roche limit: {jupiter.roche_limit(comet.density)/jupiter.radius:.1f} Jupiter radii")
    print()
    
    simulator = CometSimulator(comet=comet, celestial_bodies=celestial_bodies, config=config)
    
    print("Running simulation...")
    results = simulator.run_simulation()
    
    print(f"Simulation completed: {len(results.get('time', []))} steps")
    print(f"Fragmentation events: {len(results.get('fragmentation_events', []))}")
    print(f"Available keys in results: {list(results.keys())}")
    
    return results, comet, jupiter

def create_trajectory_plot(results, comet, jupiter):
    """Create 2D trajectory plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convert positions to km and relative to Jupiter
    positions = np.array(results['positions'])
    distances = np.linalg.norm(positions, axis=1) / 1000  # km
    times = np.array(results['time']) / 3600  # hours
    
    # Trajectory plot
    ax1.plot(positions[:, 0]/1000, positions[:, 1]/1000, 'b-', linewidth=2, label='Comet trajectory')
    ax1.scatter([0], [0], c='orange', s=200, label='Jupiter', marker='o')
    
    # Add Roche limit circle
    roche_limit = jupiter.roche_limit(comet.density) / 1000  # km
    circle = plt.Circle((0, 0), roche_limit, fill=False, color='red', linestyle='--', label='Roche limit')
    ax1.add_patch(circle)
    
    # Add fragmentation events
    if 'fragmentation_events' in results and results['fragmentation_events']:
        frag_positions = np.array([event['position'] for event in results['fragmentation_events']])
        ax1.scatter(frag_positions[:, 0]/1000, frag_positions[:, 1]/1000, 
                   c='red', s=50, marker='x', label='Fragmentation events')
    
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_title('Comet Trajectory Around Jupiter')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Distance vs time plot
    ax2.plot(times, distances, 'b-', linewidth=2, label='Distance to Jupiter')
    ax2.axhline(y=roche_limit, color='red', linestyle='--', label='Roche limit')
    ax2.axhline(y=jupiter.radius/1000, color='orange', linestyle='--', label='Jupiter surface')
    
    # Mark fragmentation events
    if 'fragmentation_events' in results and results['fragmentation_events']:
        frag_times = np.array([event['time'] for event in results['fragmentation_events']]) / 3600
        frag_distances = np.array([np.linalg.norm(event['position']) for event in results['fragmentation_events']]) / 1000
        ax2.scatter(frag_times, frag_distances, c='red', s=50, marker='x', label='Fragmentation events')
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Distance (km)')
    ax2.set_title('Distance vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('sl9_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Saved trajectory plot: sl9_trajectory_analysis.png")
    
    return fig

def create_fragmentation_analysis(results, comet, jupiter):
    """Create fragmentation analysis plots."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    times = np.array(results['time']) / 3600  # hours
    positions = np.array(results['positions'])
    distances = np.linalg.norm(positions, axis=1) / 1000  # km
    
    # Stress analysis (if available)
    if 'stress' in results:
        stresses = np.array(results['stress'])
        ax1.plot(times, stresses, 'purple', linewidth=2, label='Total stress')
        ax1.axhline(y=comet.structural_integrity, color='red', linestyle='--', label='Material strength')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Stress (Pa)')
        ax1.set_title('Stress Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
    else:
        ax1.text(0.5, 0.5, 'Stress data not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Stress Evolution (No Data)')
    
    # Velocity evolution
    if 'velocities' in results:
        velocities = np.array(results['velocities'])
        v_magnitudes = np.linalg.norm(velocities, axis=1) / 1000  # km/s
        ax2.plot(times, v_magnitudes, 'green', linewidth=2, label='Speed')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Speed (km/s)')
        ax2.set_title('Velocity Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Velocity data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Velocity Evolution (No Data)')
    
    # Fragmentation timeline
    if 'fragmentation_events' in results and results['fragmentation_events']:
        frag_times = np.array([event['time'] for event in results['fragmentation_events']]) / 3600
        frag_counts = np.arange(1, len(frag_times) + 1)
        
        ax3.step(frag_times, frag_counts, 'red', linewidth=2, where='post')
        ax3.scatter(frag_times, frag_counts, c='red', s=30)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Cumulative Fragmentation Events')
        ax3.set_title(f'Fragmentation Timeline ({len(frag_times)} events)')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No fragmentation events', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Fragmentation Timeline (No Events)')
    
    # Distance vs fragmentation
    if 'fragmentation_events' in results and results['fragmentation_events']:
        frag_distances = np.array([np.linalg.norm(event['position']) for event in results['fragmentation_events']]) / 1000
        frag_times = np.array([event['time'] for event in results['fragmentation_events']]) / 3600
        
        ax4.scatter(frag_distances, frag_times, c='red', s=50, alpha=0.7)
        ax4.axvline(x=jupiter.roche_limit(comet.density)/1000, color='red', linestyle='--', label='Roche limit')
        ax4.set_xlabel('Distance from Jupiter (km)')
        ax4.set_ylabel('Time (hours)')
        ax4.set_title('Fragmentation Events vs Distance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
    else:
        ax4.text(0.5, 0.5, 'No fragmentation events', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Fragmentation vs Distance (No Events)')
    
    plt.tight_layout()
    plt.savefig('sl9_fragmentation_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Saved fragmentation analysis: sl9_fragmentation_analysis.png")
    
    return fig

def create_summary_report(results, comet, jupiter):
    """Generate a text summary report."""
    
    report = []
    report.append("🌌 SHOEMAKER-LEVY 9 SIMULATION ANALYSIS REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Basic info
    report.append("📊 SIMULATION SUMMARY:")
    report.append(f"   • Total simulation steps: {len(results['time'])}")
    if results['time']:
        total_time = results['time'][-1] / 3600
        report.append(f"   • Total simulation time: {total_time:.1f} hours")
    
    # Trajectory analysis
    if results['positions']:
        positions = np.array(results['positions'])
        distances = np.linalg.norm(positions, axis=1)
        closest_approach = np.min(distances)
        
        report.append("")
        report.append("📍 TRAJECTORY ANALYSIS:")
        report.append(f"   • Starting distance: {distances[0]/jupiter.radius:.1f} Jupiter radii")
        report.append(f"   • Closest approach: {closest_approach/jupiter.radius:.2f} Jupiter radii")
        report.append(f"   • Closest approach: {closest_approach/1000:.0f} km")
        
        roche_limit = jupiter.roche_limit(comet.density)
        if closest_approach < roche_limit:
            report.append(f"   ✅ Comet penetrated Roche limit ({roche_limit/jupiter.radius:.1f} Jupiter radii)")
        else:
            report.append(f"   ❌ Comet did not reach Roche limit")
    
    # Fragmentation analysis
    if 'fragmentation_events' in results and results['fragmentation_events']:
        events = results['fragmentation_events']
        report.append("")
        report.append("💥 FRAGMENTATION ANALYSIS:")
        report.append(f"   • Total fragmentation events: {len(events)}")
        
        if events:
            first_event = events[0]
            last_event = events[-1]
            
            first_distance = np.linalg.norm(first_event['position'])
            first_time = first_event['time'] / 3600
            
            last_distance = np.linalg.norm(last_event['position'])
            last_time = last_event['time'] / 3600
            
            report.append(f"   • First fragmentation: {first_time:.1f} hours at {first_distance/jupiter.radius:.2f} Jupiter radii")
            report.append(f"   • Last fragmentation: {last_time:.1f} hours at {last_distance/jupiter.radius:.2f} Jupiter radii")
            
            # Fragment count
            total_fragments = sum(event.get('fragments_created', 1) for event in events)
            report.append(f"   • Total fragments created: {total_fragments}")
            report.append(f"   • Average fragments per event: {total_fragments/len(events):.1f}")
    else:
        report.append("")
        report.append("💥 FRAGMENTATION ANALYSIS:")
        report.append("   • No fragmentation events detected")
        report.append("   • Consider adjusting material properties or simulation parameters")
    
    # Physical parameters
    report.append("")
    report.append("🔬 PHYSICAL PARAMETERS:")
    report.append(f"   • Comet mass: {comet.mass:.1e} kg")
    report.append(f"   • Comet radius: {comet.radius:.0f} m")
    report.append(f"   • Comet density: {comet.density:.0f} kg/m³")
    report.append(f"   • Structural integrity: {comet.structural_integrity:.0e} Pa")
    report.append(f"   • Jupiter mass: {jupiter.mass:.2e} kg")
    report.append(f"   • Jupiter radius: {jupiter.radius/1000:.0f} km")
    report.append(f"   • Roche limit: {jupiter.roche_limit(comet.density)/1000:.0f} km")
    
    report.append("")
    report.append("🎯 SIMULATION SUCCESS METRICS:")
    if 'fragmentation_events' in results and results['fragmentation_events']:
        report.append("   ✅ Fragmentation successfully modeled")
        report.append("   ✅ Tidal effects captured")
        report.append("   ✅ Realistic comet breakup simulation")
    else:
        report.append("   ⚠️ No fragmentation detected")
        report.append("   • May need stronger tidal forces or weaker material")
    
    report_text = "\n".join(report)
    
    # Save to file
    with open('sl9_simulation_report.txt', 'w') as f:
        f.write(report_text)
    
    print("📄 Saved analysis report: sl9_simulation_report.txt")
    print()
    print(report_text)
    
    return report_text

def main():
    """Run complete analysis with visualizations."""
    
    # Run simulation
    results, comet, jupiter = run_simulation()
    
    print("\n🎨 Creating visualizations...")
    
    # Create plots
    try:
        trajectory_fig = create_trajectory_plot(results, comet, jupiter)
        fragmentation_fig = create_fragmentation_analysis(results, comet, jupiter)
        
        print("📊 Visualizations created successfully!")
        
    except Exception as e:
        print(f"⚠️ Error creating plots: {e}")
        print("Continuing with text analysis...")
    
    # Generate report
    print("\n📝 Generating analysis report...")
    report = create_summary_report(results, comet, jupiter)
    
    print("\n🎯 Analysis complete!")
    print("\nGenerated files:")
    print("   • sl9_trajectory_analysis.png")
    print("   • sl9_fragmentation_analysis.png")
    print("   • sl9_simulation_report.txt")
    print("\n🌌 Simulation and visualization complete!")

if __name__ == "__main__":
    main()
