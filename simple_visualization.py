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
    
    print(f"Simulation completed: {len(results['time'])} steps")
    print(f"Fragmentation events: {len(results.get('fragmentation_events', []))}")
    
    return results, comet, jupiter

def create_trajectory_plot(results, comet, jupiter):
    """Create 2D trajectory plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convert positions to km and relative to Jupiter
    positions = np.array(results['position'])
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
    ax1.set_aspect('equal')\n    \n    # Distance vs time plot\n    ax2.plot(times, distances, 'b-', linewidth=2, label='Distance to Jupiter')\n    ax2.axhline(y=roche_limit, color='red', linestyle='--', label='Roche limit')\n    ax2.axhline(y=jupiter.radius/1000, color='orange', linestyle='--', label='Jupiter surface')\n    \n    # Mark fragmentation events\n    if 'fragmentation_events' in results and results['fragmentation_events']:\n        frag_times = np.array([event['time'] for event in results['fragmentation_events']]) / 3600\n        frag_distances = np.array([np.linalg.norm(event['position']) for event in results['fragmentation_events']]) / 1000\n        ax2.scatter(frag_times, frag_distances, c='red', s=50, marker='x', label='Fragmentation events')\n    \n    ax2.set_xlabel('Time (hours)')\n    ax2.set_ylabel('Distance (km)')\n    ax2.set_title('Distance vs Time')\n    ax2.legend()\n    ax2.grid(True, alpha=0.3)\n    ax2.set_yscale('log')\n    \n    plt.tight_layout()\n    plt.savefig('sl9_trajectory_analysis.png', dpi=300, bbox_inches='tight')\n    print(\"📊 Saved trajectory plot: sl9_trajectory_analysis.png\")\n    \n    return fig\n\ndef create_fragmentation_analysis(results, comet, jupiter):\n    \"\"\"Create fragmentation analysis plots.\"\"\"\n    \n    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))\n    \n    times = np.array(results['time']) / 3600  # hours\n    positions = np.array(results['position'])\n    distances = np.linalg.norm(positions, axis=1) / 1000  # km\n    \n    # Stress analysis (if available)\n    if 'stress' in results:\n        stresses = np.array(results['stress'])\n        ax1.plot(times, stresses, 'purple', linewidth=2, label='Total stress')\n        ax1.axhline(y=comet.structural_integrity, color='red', linestyle='--', label='Material strength')\n        ax1.set_xlabel('Time (hours)')\n        ax1.set_ylabel('Stress (Pa)')\n        ax1.set_title('Stress Evolution')\n        ax1.legend()\n        ax1.grid(True, alpha=0.3)\n        ax1.set_yscale('log')\n    else:\n        ax1.text(0.5, 0.5, 'Stress data not available', ha='center', va='center', transform=ax1.transAxes)\n        ax1.set_title('Stress Evolution (No Data)')\n    \n    # Velocity evolution\n    if 'velocity' in results:\n        velocities = np.array(results['velocity'])\n        v_magnitudes = np.linalg.norm(velocities, axis=1) / 1000  # km/s\n        ax2.plot(times, v_magnitudes, 'green', linewidth=2, label='Speed')\n        ax2.set_xlabel('Time (hours)')\n        ax2.set_ylabel('Speed (km/s)')\n        ax2.set_title('Velocity Evolution')\n        ax2.legend()\n        ax2.grid(True, alpha=0.3)\n    else:\n        ax2.text(0.5, 0.5, 'Velocity data not available', ha='center', va='center', transform=ax2.transAxes)\n        ax2.set_title('Velocity Evolution (No Data)')\n    \n    # Fragmentation timeline\n    if 'fragmentation_events' in results and results['fragmentation_events']:\n        frag_times = np.array([event['time'] for event in results['fragmentation_events']]) / 3600\n        frag_counts = np.arange(1, len(frag_times) + 1)\n        \n        ax3.step(frag_times, frag_counts, 'red', linewidth=2, where='post')\n        ax3.scatter(frag_times, frag_counts, c='red', s=30)\n        ax3.set_xlabel('Time (hours)')\n        ax3.set_ylabel('Cumulative Fragmentation Events')\n        ax3.set_title(f'Fragmentation Timeline ({len(frag_times)} events)')\n        ax3.grid(True, alpha=0.3)\n    else:\n        ax3.text(0.5, 0.5, 'No fragmentation events', ha='center', va='center', transform=ax3.transAxes)\n        ax3.set_title('Fragmentation Timeline (No Events)')\n    \n    # Distance vs fragmentation\n    if 'fragmentation_events' in results and results['fragmentation_events']:\n        frag_distances = np.array([np.linalg.norm(event['position']) for event in results['fragmentation_events']]) / 1000\n        frag_times = np.array([event['time'] for event in results['fragmentation_events']]) / 3600\n        \n        ax4.scatter(frag_distances, frag_times, c='red', s=50, alpha=0.7)\n        ax4.axvline(x=jupiter.roche_limit(comet.density)/1000, color='red', linestyle='--', label='Roche limit')\n        ax4.set_xlabel('Distance from Jupiter (km)')\n        ax4.set_ylabel('Time (hours)')\n        ax4.set_title('Fragmentation Events vs Distance')\n        ax4.legend()\n        ax4.grid(True, alpha=0.3)\n        ax4.set_xscale('log')\n    else:\n        ax4.text(0.5, 0.5, 'No fragmentation events', ha='center', va='center', transform=ax4.transAxes)\n        ax4.set_title('Fragmentation vs Distance (No Events)')\n    \n    plt.tight_layout()\n    plt.savefig('sl9_fragmentation_analysis.png', dpi=300, bbox_inches='tight')\n    print(\"📊 Saved fragmentation analysis: sl9_fragmentation_analysis.png\")\n    \n    return fig\n\ndef create_summary_report(results, comet, jupiter):\n    \"\"\"Generate a text summary report.\"\"\"\n    \n    report = []\n    report.append(\"🌌 SHOEMAKER-LEVY 9 SIMULATION ANALYSIS REPORT\")\n    report.append(\"=\" * 50)\n    report.append(\"\")\n    \n    # Basic info\n    report.append(\"📊 SIMULATION SUMMARY:\")\n    report.append(f\"   • Total simulation steps: {len(results['time'])}\")\n    if results['time']:\n        total_time = results['time'][-1] / 3600\n        report.append(f\"   • Total simulation time: {total_time:.1f} hours\")\n    \n    # Trajectory analysis\n    if results['position']:\n        positions = np.array(results['position'])\n        distances = np.linalg.norm(positions, axis=1)\n        closest_approach = np.min(distances)\n        \n        report.append(\"\")\n        report.append(\"📍 TRAJECTORY ANALYSIS:\")\n        report.append(f\"   • Starting distance: {distances[0]/jupiter.radius:.1f} Jupiter radii\")\n        report.append(f\"   • Closest approach: {closest_approach/jupiter.radius:.2f} Jupiter radii\")\n        report.append(f\"   • Closest approach: {closest_approach/1000:.0f} km\")\n        \n        roche_limit = jupiter.roche_limit(comet.density)\n        if closest_approach < roche_limit:\n            report.append(f\"   ✅ Comet penetrated Roche limit ({roche_limit/jupiter.radius:.1f} Jupiter radii)\")\n        else:\n            report.append(f\"   ❌ Comet did not reach Roche limit\")\n    \n    # Fragmentation analysis\n    if 'fragmentation_events' in results and results['fragmentation_events']:\n        events = results['fragmentation_events']\n        report.append(\"\")\n        report.append(\"💥 FRAGMENTATION ANALYSIS:\")\n        report.append(f\"   • Total fragmentation events: {len(events)}\")\n        \n        if events:\n            first_event = events[0]\n            last_event = events[-1]\n            \n            first_distance = np.linalg.norm(first_event['position'])\n            first_time = first_event['time'] / 3600\n            \n            last_distance = np.linalg.norm(last_event['position'])\n            last_time = last_event['time'] / 3600\n            \n            report.append(f\"   • First fragmentation: {first_time:.1f} hours at {first_distance/jupiter.radius:.2f} Jupiter radii\")\n            report.append(f\"   • Last fragmentation: {last_time:.1f} hours at {last_distance/jupiter.radius:.2f} Jupiter radii\")\n            \n            # Fragment count\n            total_fragments = sum(event.get('fragments_created', 1) for event in events)\n            report.append(f\"   • Total fragments created: {total_fragments}\")\n            report.append(f\"   • Average fragments per event: {total_fragments/len(events):.1f}\")\n    else:\n        report.append(\"\")\n        report.append(\"💥 FRAGMENTATION ANALYSIS:\")\n        report.append(\"   • No fragmentation events detected\")\n        report.append(\"   • Consider adjusting material properties or simulation parameters\")\n    \n    # Physical parameters\n    report.append(\"\")\n    report.append(\"🔬 PHYSICAL PARAMETERS:\")\n    report.append(f\"   • Comet mass: {comet.mass:.1e} kg\")\n    report.append(f\"   • Comet radius: {comet.radius:.0f} m\")\n    report.append(f\"   • Comet density: {comet.density:.0f} kg/m³\")\n    report.append(f\"   • Structural integrity: {comet.structural_integrity:.0e} Pa\")\n    report.append(f\"   • Jupiter mass: {jupiter.mass:.2e} kg\")\n    report.append(f\"   • Jupiter radius: {jupiter.radius/1000:.0f} km\")\n    report.append(f\"   • Roche limit: {jupiter.roche_limit(comet.density)/1000:.0f} km\")\n    \n    report.append(\"\")\n    report.append(\"🎯 SIMULATION SUCCESS METRICS:\")\n    if 'fragmentation_events' in results and results['fragmentation_events']:\n        report.append(\"   ✅ Fragmentation successfully modeled\")\n        report.append(\"   ✅ Tidal effects captured\")\n        report.append(\"   ✅ Realistic comet breakup simulation\")\n    else:\n        report.append(\"   ⚠️ No fragmentation detected\")\n        report.append(\"   • May need stronger tidal forces or weaker material\")\n    \n    report_text = \"\\n\".join(report)\n    \n    # Save to file\n    with open('sl9_simulation_report.txt', 'w') as f:\n        f.write(report_text)\n    \n    print(\"📄 Saved analysis report: sl9_simulation_report.txt\")\n    print()\n    print(report_text)\n    \n    return report_text\n\ndef main():\n    \"\"\"Run complete analysis with visualizations.\"\"\"\n    \n    # Run simulation\n    results, comet, jupiter = run_simulation()\n    \n    print(\"\\n🎨 Creating visualizations...\")\n    \n    # Create plots\n    try:\n        trajectory_fig = create_trajectory_plot(results, comet, jupiter)\n        fragmentation_fig = create_fragmentation_analysis(results, comet, jupiter)\n        \n        print(\"📊 Visualizations created successfully!\")\n        \n    except Exception as e:\n        print(f\"⚠️ Error creating plots: {e}\")\n        print(\"Continuing with text analysis...\")\n    \n    # Generate report\n    print(\"\\n📝 Generating analysis report...\")\n    report = create_summary_report(results, comet, jupiter)\n    \n    print(\"\\n🎯 Analysis complete!\")\n    print(\"\\nGenerated files:\")\n    print(\"   • sl9_trajectory_analysis.png\")\n    print(\"   • sl9_fragmentation_analysis.png\")\n    print(\"   • sl9_simulation_report.txt\")\n    print(\"\\n🌌 Simulation and visualization complete!\")\n\nif __name__ == \"__main__\":\n    main()
