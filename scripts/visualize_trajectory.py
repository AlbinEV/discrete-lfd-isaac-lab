#!/usr/bin/env python3
"""Generate HTML visualization of trajectory."""

import argparse
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_trajectory(trajectory_path: str, output_html: str):
    """Create interactive HTML plot of trajectory.
    
    Args:
        trajectory_path: Path to trajectory JSON file
        output_html: Path to output HTML file
    """
    # Load trajectory
    with open(trajectory_path, 'r') as f:
        traj_data = json.load(f)
    
    # Extract positions
    positions = np.array([[wp['x'], wp['y'], wp['z']] for wp in traj_data['waypoints']])
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('3D Trajectory', 'XY Plane', 'Height Profile', 'Distance Between Waypoints'),
        specs=[[{'type': 'scatter3d', 'rowspan': 2}, {'type': 'scatter'}],
               [None, {'type': 'scatter'}]],
        horizontal_spacing=0.12,
        vertical_spacing=0.1
    )
    
    # 3D trajectory
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            marker=dict(size=2, color=np.arange(len(positions)), colorscale='Viridis', showscale=True),
            line=dict(color='gray', width=2),
            name='Trajectory',
            hovertemplate='<b>Waypoint %{text}</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>',
            text=[f"{i}" for i in range(len(positions))]
        ),
        row=1, col=1
    )
    
    # XY plane
    fig.add_trace(
        go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode='lines+markers',
            marker=dict(size=3, color=np.arange(len(positions)), colorscale='Viridis'),
            line=dict(color='gray', width=1),
            name='XY Path',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Height profile
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(positions)),
            y=positions[:, 2],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Z Height',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Trajectory Visualization<br><sub>{len(positions)} waypoints</sub>",
        height=800,
        showlegend=True,
        hovermode='closest'
    )
    
    # Update axes
    fig.update_scenes(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data'
    )
    fig.update_xaxes(title_text='X (m)', row=1, col=2)
    fig.update_yaxes(title_text='Y (m)', row=1, col=2)
    fig.update_xaxes(title_text='Waypoint Index', row=2, col=2)
    fig.update_yaxes(title_text='Z (m)', row=2, col=2)
    
    # Save HTML
    fig.write_html(output_html)
    print(f"✓ Saved interactive plot to {output_html}")
    print(f"  Open in browser to explore {len(positions)} waypoints")

def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory as interactive HTML")
    parser.add_argument("trajectory", help="Path to trajectory JSON file")
    parser.add_argument("--output", "-o", default="trajectory.html", 
                        help="Output HTML file (default: trajectory.html)")
    args = parser.parse_args()
    
    visualize_trajectory(args.trajectory, args.output)

if __name__ == "__main__":
    main()
