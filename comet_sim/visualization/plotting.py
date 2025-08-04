"""
3D visualization components for comet simulation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class TrajectoryPlotter:
    """
    Creates 3D visualizations of comet trajectories and fragmentation events.
    
    Supports both matplotlib and plotly backends for different use cases.
    """
    
    def __init__(self, backend: str = 'plotly'):
        """
        Initialize plotter.
        
        Args:
            backend: 'plotly' for interactive plots, 'matplotlib' for static
        """
        self.backend = backend
        self.fig = None
        self.traces = []
        
        # Color schemes
        self.comet_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        self.body_colors = {
            'Sun': 'gold',
            'Jupiter': 'brown',
            'Saturn': 'tan',
            'Earth': 'blue',
            'Mars': 'red'
        }
    
    def create_figure(self, title: str = "Comet Simulation") -> None:
        """
        Create a new 3D figure.
        
        Args:
            title: Title for the plot
        """
        if self.backend == 'plotly':
            self.fig = go.Figure()
            self.fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="X (AU)",
                    yaxis_title="Y (AU)",
                    zaxis_title="Z (AU)",
                    aspectmode='cube'
                ),
                showlegend=True
            )
        else:
            self.fig = plt.figure(figsize=(12, 9))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_title(title)
            self.ax.set_xlabel('X (AU)')
            self.ax.set_ylabel('Y (AU)')
            self.ax.set_zlabel('Z (AU)')
    
    def add_celestial_body(self,
                          position: np.ndarray,
                          name: str,
                          radius: float = None) -> None:
        """
        Add a celestial body to the plot.
        
        Args:
            position: Position in meters [x, y, z]
            name: Name of the body
            radius: Radius for visualization (optional)
        """
        # Convert to AU
        au = 1.496e11
        pos_au = position / au
        
        color = self.body_colors.get(name, 'gray')
        
        # Scale radius for visibility
        if radius is None:
            if name == 'Sun':
                plot_radius = 0.1
            elif name in ['Jupiter', 'Saturn']:
                plot_radius = 0.05
            else:
                plot_radius = 0.02
        else:
            plot_radius = max(radius / au, 0.01)  # Minimum visible size
        
        if self.backend == 'plotly':
            # Create sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = plot_radius * np.outer(np.cos(u), np.sin(v)) + pos_au[0]
            y = plot_radius * np.outer(np.sin(u), np.sin(v)) + pos_au[1]
            z = plot_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos_au[2]
            
            self.fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.8,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=name,
                hovertemplate=f"{name}<br>Position: ({pos_au[0]:.2f}, {pos_au[1]:.2f}, {pos_au[2]:.2f}) AU"
            ))
        else:
            self.ax.scatter(pos_au[0], pos_au[1], pos_au[2],
                          c=color, s=plot_radius*1000, label=name, alpha=0.8)
    
    def add_comet_trajectory(self,
                           positions: np.ndarray,
                           times: np.ndarray,
                           comet_id: int = 0,
                           name: str = "Comet") -> None:
        """
        Add a comet trajectory to the plot.
        
        Args:
            positions: Array of positions [N, 3] in meters
            times: Array of times [N] in seconds
            comet_id: Identifier for color coding
            name: Name for legend
        """
        # Convert to AU
        au = 1.496e11
        pos_au = positions / au
        
        color = self.comet_colors[comet_id % len(self.comet_colors)]
        
        if self.backend == 'plotly':
            # Color by time for gradient effect
            time_normalized = (times - times[0]) / (times[-1] - times[0])
            
            self.fig.add_trace(go.Scatter3d(
                x=pos_au[:, 0],
                y=pos_au[:, 1],
                z=pos_au[:, 2],
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(
                    size=3,
                    color=time_normalized,
                    colorscale='Viridis',
                    showscale=False
                ),
                name=name,
                hovertemplate="Time: %{text:.1f} days<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f}) AU",
                text=times / (24 * 3600)  # Convert to days
            ))
        else:
            self.ax.plot(pos_au[:, 0], pos_au[:, 1], pos_au[:, 2],
                        color=color, linewidth=2, label=name)
            
            # Mark start and end
            self.ax.scatter(pos_au[0, 0], pos_au[0, 1], pos_au[0, 2],
                          color=color, s=50, marker='o', alpha=0.8)
            self.ax.scatter(pos_au[-1, 0], pos_au[-1, 1], pos_au[-1, 2],
                          color=color, s=50, marker='s', alpha=0.8)
    
    def add_fragmentation_event(self,
                              position: np.ndarray,
                              time: float,
                              fragments: List,
                              event_id: int = 0) -> None:
        """
        Add visualization of fragmentation event.
        
        Args:
            position: Position where fragmentation occurred
            time: Time of fragmentation
            fragments: List of fragment objects
            event_id: Event identifier
        """
        # Convert to AU
        au = 1.496e11
        pos_au = position / au
        
        if self.backend == 'plotly':
            # Explosion marker
            self.fig.add_trace(go.Scatter3d(
                x=[pos_au[0]],
                y=[pos_au[1]],
                z=[pos_au[2]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='x',
                    opacity=0.8
                ),
                name=f"Fragmentation {event_id}",
                hovertemplate=f"Fragmentation Event {event_id}<br>Time: {time/(24*3600):.1f} days<br>Fragments: {len(fragments)}"
            ))
            
            # Fragment positions
            if fragments:
                frag_pos = np.array([f.position for f in fragments]) / au
                frag_masses = [f.mass for f in fragments]
                max_mass = max(frag_masses)
                sizes = [5 + 15 * (mass / max_mass) for mass in frag_masses]
                
                self.fig.add_trace(go.Scatter3d(
                    x=frag_pos[:, 0],
                    y=frag_pos[:, 1],
                    z=frag_pos[:, 2],
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color='orange',
                        opacity=0.6
                    ),
                    name=f"Fragments {event_id}",
                    hovertemplate="Fragment Mass: %{text:.2e} kg",
                    text=frag_masses
                ))
        else:
            # Mark fragmentation point
            self.ax.scatter(pos_au[0], pos_au[1], pos_au[2],
                          color='red', s=200, marker='x', alpha=0.8,
                          label=f"Fragmentation {event_id}")
            
            # Fragment positions
            if fragments:
                frag_pos = np.array([f.position for f in fragments]) / au
                self.ax.scatter(frag_pos[:, 0], frag_pos[:, 1], frag_pos[:, 2],
                              color='orange', s=20, alpha=0.6)
    
    def add_roche_limit(self,
                       primary_position: np.ndarray,
                       roche_distance: float,
                       name: str = "Roche Limit") -> None:
        """
        Add Roche limit visualization.
        
        Args:
            primary_position: Position of primary body
            roche_distance: Roche limit distance in meters
            name: Name for legend
        """
        au = 1.496e11
        pos_au = primary_position / au
        radius_au = roche_distance / au
        
        if self.backend == 'plotly':
            # Create sphere for Roche limit
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = radius_au * np.outer(np.cos(u), np.sin(v)) + pos_au[0]
            y = radius_au * np.outer(np.sin(u), np.sin(v)) + pos_au[1]
            z = radius_au * np.outer(np.ones(np.size(u)), np.cos(v)) + pos_au[2]
            
            self.fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.2,
                colorscale=[[0, 'red'], [1, 'red']],
                showscale=False,
                name=name,
                hovertemplate=f"{name}<br>Distance: {radius_au:.3f} AU"
            ))
        else:
            # Draw sphere wireframe
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = radius_au * np.outer(np.cos(u), np.sin(v)) + pos_au[0]
            y = radius_au * np.outer(np.sin(u), np.sin(v)) + pos_au[1]
            z = radius_au * np.outer(np.ones(np.size(u)), np.cos(v)) + pos_au[2]
            
            self.ax.plot_wireframe(x, y, z, alpha=0.3, color='red', linewidth=0.5)
    
    def set_equal_aspect(self, range_au: float = None) -> None:
        """
        Set equal aspect ratio for the plot.
        
        Args:
            range_au: Range in AU to display (if None, auto-calculate)
        """
        if self.backend == 'plotly':
            if range_au is not None:
                self.fig.update_layout(
                    scene=dict(
                        xaxis=dict(range=[-range_au, range_au]),
                        yaxis=dict(range=[-range_au, range_au]),
                        zaxis=dict(range=[-range_au, range_au]),
                        aspectmode='cube'
                    )
                )
        else:
            if range_au is not None:
                self.ax.set_xlim(-range_au, range_au)
                self.ax.set_ylim(-range_au, range_au)
                self.ax.set_zlim(-range_au, range_au)
            
            # Equal aspect
            self.ax.set_box_aspect([1,1,1])
    
    def show(self) -> None:
        """Display the plot."""
        if self.backend == 'plotly':
            self.fig.show()
        else:
            self.ax.legend()
            plt.tight_layout()
            plt.show()
    
    def save(self, filename: str, **kwargs) -> None:
        """
        Save the plot to file.
        
        Args:
            filename: Output filename
            **kwargs: Additional arguments for saving
        """
        if self.backend == 'plotly':
            if filename.endswith('.html'):
                self.fig.write_html(filename, **kwargs)
            elif filename.endswith('.png'):
                self.fig.write_image(filename, **kwargs)
            else:
                self.fig.write_html(filename + '.html', **kwargs)
        else:
            plt.savefig(filename, **kwargs)


class AnalysisPlotter:
    """
    Creates analysis plots for comet simulation results.
    
    Provides charts for energy, stress, fragmentation statistics, etc.
    """
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set1
    
    def plot_energy_conservation(self,
                               times: np.ndarray,
                               kinetic_energy: np.ndarray,
                               potential_energy: np.ndarray,
                               total_energy: np.ndarray) -> go.Figure:
        """
        Plot energy conservation over time.
        
        Args:
            times: Time array in seconds
            kinetic_energy: Kinetic energy array
            potential_energy: Potential energy array
            total_energy: Total energy array
            
        Returns:
            Plotly figure
        """
        # Convert to days
        times_days = times / (24 * 3600)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times_days,
            y=kinetic_energy,
            mode='lines',
            name='Kinetic Energy',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=times_days,
            y=potential_energy,
            mode='lines',
            name='Potential Energy',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=times_days,
            y=total_energy,
            mode='lines',
            name='Total Energy',
            line=dict(color='black', dash='dash')
        ))
        
        fig.update_layout(
            title="Energy Conservation",
            xaxis_title="Time (days)",
            yaxis_title="Energy (J)",
            showlegend=True
        )
        
        return fig
    
    def plot_stress_evolution(self,
                            times: np.ndarray,
                            stress_components: Dict[str, np.ndarray]) -> go.Figure:
        """
        Plot stress evolution over time.
        
        Args:
            times: Time array in seconds
            stress_components: Dictionary of stress arrays
            
        Returns:
            Plotly figure
        """
        times_days = times / (24 * 3600)
        
        fig = go.Figure()
        
        for i, (component, values) in enumerate(stress_components.items()):
            fig.add_trace(go.Scatter(
                x=times_days,
                y=values,
                mode='lines',
                name=component.capitalize() + ' Stress',
                line=dict(color=self.colors[i % len(self.colors)])
            ))
        
        fig.update_layout(
            title="Stress Evolution",
            xaxis_title="Time (days)",
            yaxis_title="Stress (Pa)",
            showlegend=True
        )
        
        return fig
    
    def plot_distance_to_primary(self,
                               times: np.ndarray,
                               distances: np.ndarray,
                               roche_limit: float = None) -> go.Figure:
        """
        Plot distance to primary body over time.
        
        Args:
            times: Time array in seconds
            distances: Distance array in meters
            roche_limit: Roche limit distance (optional)
            
        Returns:
            Plotly figure
        """
        times_days = times / (24 * 3600)
        distances_au = distances / 1.496e11
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times_days,
            y=distances_au,
            mode='lines',
            name='Distance to Primary',
            line=dict(color='blue')
        ))
        
        if roche_limit is not None:
            roche_au = roche_limit / 1.496e11
            fig.add_hline(
                y=roche_au,
                line_dash="dash",
                line_color="red",
                annotation_text="Roche Limit"
            )
        
        fig.update_layout(
            title="Distance to Primary Body",
            xaxis_title="Time (days)",
            yaxis_title="Distance (AU)",
            showlegend=True
        )
        
        return fig
    
    def plot_fragmentation_statistics(self,
                                    fragmentation_events: List) -> go.Figure:
        """
        Plot fragmentation statistics.
        
        Args:
            fragmentation_events: List of fragmentation events
            
        Returns:
            Plotly figure
        """
        if not fragmentation_events:
            # Empty plot
            fig = go.Figure()
            fig.update_layout(title="No Fragmentation Events")
            return fig
        
        # Extract data
        times = [event.time / (24 * 3600) for event in fragmentation_events]
        fragment_counts = [event.get_fragment_count() for event in fragmentation_events]
        causes = [event.cause for event in fragmentation_events]
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Fragmentation Timeline', 'Fragment Count Distribution')
        )
        
        # Timeline
        fig.add_trace(go.Scatter(
            x=times,
            y=fragment_counts,
            mode='markers',
            marker=dict(
                size=10,
                color=[self.colors[i % len(self.colors)] for i in range(len(times))],
                opacity=0.7
            ),
            text=causes,
            hovertemplate="Time: %{x:.1f} days<br>Fragments: %{y}<br>Cause: %{text}",
            name="Events"
        ), row=1, col=1)
        
        # Distribution
        fig.add_trace(go.Histogram(
            x=fragment_counts,
            nbinsx=10,
            name="Distribution",
            marker_color='lightblue'
        ), row=2, col=1)
        
        fig.update_xaxes(title_text="Time (days)", row=1, col=1)
        fig.update_yaxes(title_text="Fragment Count", row=1, col=1)
        fig.update_xaxes(title_text="Fragment Count", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_layout(title="Fragmentation Analysis", showlegend=False)
        
        return fig
