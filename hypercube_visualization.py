#!/usr/bin/env python3
"""
n-dimensional Hypercube Visualization

This script provides a 3D visualization of an n-dimensional hypercube rotating
through its various rotation planes. The dimension 'n' is provided as a command-line
argument (n >= 4). The visualization includes interactive controls for pausing,
resuming, and restarting the animation.

Usage:
    python hypercube_visualization.py n

Example:
    python hypercube_visualization.py 5

Dependencies:
    - Python 3.8
    - NumPy
    - Matplotlib

Author:
    Your Name (your.email@example.com)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation
from matplotlib.widgets import Button
from itertools import combinations, product
import argparse
import sys

def get_coord_labels(n):
    """
    Generate coordinate labels for n dimensions.

    Parameters:
        n (int): The number of dimensions.

    Returns:
        list of str: A list of coordinate labels.
    """
    return [f'X{i}' for i in range(n)]

def generate_vertices(n):
    """
    Generate the vertices of an n-dimensional hypercube.

    Parameters:
        n (int): The number of dimensions.

    Returns:
        np.ndarray: An array of vertices coordinates.
    """
    # Each vertex coordinate is either -1 or 1 in each dimension
    return np.array(list(product([-1, 1], repeat=n)))

def generate_edges(vertices):
    """
    Generate the edges of the hypercube by connecting vertices that differ by
    exactly one coordinate.

    Parameters:
        vertices (np.ndarray): Array of vertex coordinates.

    Returns:
        list of tuple: A list of edges represented as pairs of vertex indices.
    """
    edges = []
    n_vertices = len(vertices)
    for i in range(n_vertices):
        for j in range(i+1, n_vertices):
            # Edges connect vertices differing in exactly one coordinate
            if np.sum(np.abs(vertices[i] - vertices[j])) == 2:
                edges.append((i, j))
    return edges

def generate_rotation_planes(n):
    """
    Generate all possible rotation planes (pairs of axes) in n dimensions.

    Parameters:
        n (int): The number of dimensions.

    Returns:
        list of tuple: A list of axis index pairs representing rotation planes.
    """
    return list(combinations(range(n), 2))

def get_rotation_matrix(angle, axis1, axis2, n):
    """
    Generate an n-dimensional rotation matrix for rotation in the plane defined
    by axis1 and axis2.

    Parameters:
        angle (float): The rotation angle in radians.
        axis1 (int): The first axis index.
        axis2 (int): The second axis index.
        n (int): The number of dimensions.

    Returns:
        np.ndarray: An n x n rotation matrix.
    """
    # Initialize an identity matrix
    matrix = np.eye(n)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    # Apply rotation in the specified plane
    matrix[axis1, axis1] = cos_angle
    matrix[axis2, axis2] = cos_angle
    matrix[axis1, axis2] = -sin_angle
    matrix[axis2, axis1] = sin_angle
    return matrix

def project_nd_to_3d(vertices_nd, d=2, epsilon=1e-5):
    """
    Project n-dimensional vertices down to 3D using perspective projection.

    Parameters:
        vertices_nd (np.ndarray): Array of n-dimensional vertex coordinates.
        d (float): The distance from the projection center to the hyperplane.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        np.ndarray: Array of 3D projected vertex coordinates.
    """
    vertices_projected = vertices_nd.copy()
    n = vertices_nd.shape[1]
    # Iteratively project from nD to 3D
    for coord_index in reversed(range(3, n)):
        denominator = d - vertices_projected[:, coord_index]
        denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
        scale = 1 / denominator
        # Scale coordinates up to the current dimension
        vertices_projected = vertices_projected[:, :coord_index] * scale[:, np.newaxis]
    return vertices_projected[:, :3]

def draw_edges(ax, projected_vertices, edges):
    """
    Draw the edges of the hypercube in the 3D plot.

    Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes to draw on.
        projected_vertices (np.ndarray): Array of 3D projected vertex coordinates.
        edges (list of tuple): List of edges represented as pairs of vertex indices.
    """
    lines = []
    for edge in edges:
        start, end = edge
        line = [projected_vertices[start], projected_vertices[end]]
        lines.append(line)
    edge_collection = Line3DCollection(lines, colors='white', linewidths=1)
    ax.add_collection3d(edge_collection)

def draw_vertices(ax, projected_vertices):
    """
    Draw the vertices of the hypercube in the 3D plot.

    Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes to draw on.
        projected_vertices (np.ndarray): Array of 3D projected vertex coordinates.
    """
    ax.scatter(projected_vertices[:, 0], projected_vertices[:, 1], projected_vertices[:, 2],
               color='white', s=20)

def update(manager, frame):
    """
    Update function for the animation.

    Parameters:
        manager (AnimationManager): The animation manager instance.
        frame (int): The current frame number.

    Returns:
        matplotlib.axes._subplots.Axes3DSubplot: The updated 3D axes.
    """
    vertices_nd = manager.original_vertices
    ax = manager.ax
    n = manager.n
    coord_labels = manager.coord_labels
    rotation_planes = manager.rotation_planes

    total_planes = len(rotation_planes)
    plane_total_frames = 360
    total_frames = plane_total_frames * total_planes
    current_frame = frame % total_frames
    current_plane_idx = current_frame // plane_total_frames
    current_plane = rotation_planes[current_plane_idx]
    angle = (current_frame % plane_total_frames) * np.pi / 180

    # Generate rotation matrix and rotate vertices
    rotation_matrix = get_rotation_matrix(angle, *current_plane, n)
    rotated_vertices = np.dot(vertices_nd, rotation_matrix.T)

    # Project the rotated vertices down to 3D
    projected_vertices = project_nd_to_3d(rotated_vertices)

    # Clear the axes and redraw
    ax.clear()
    draw_edges(ax, projected_vertices, manager.edges)
    draw_vertices(ax, projected_vertices)

    # Set plot parameters
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.axis('off')  # Hide axes for better visualization

    # Display rotation information
    rotation_info = f"Rotating {n}-dimensional hypercube in {coord_labels[current_plane[0]]}{coord_labels[current_plane[1]]}-plane"
    frame_info = f"Frame: {frame}"
    ax.text2D(0.05, 0.95, rotation_info, transform=ax.transAxes, color='white')
    ax.text2D(0.05, 0.90, frame_info, transform=ax.transAxes, color='white')

    return ax

class AnimationManager:
    """
    Class to manage the animation, including pausing, resuming, and restarting.

    Attributes:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes for plotting.
        original_vertices (np.ndarray): The original n-dimensional vertices.
        edges (list of tuple): List of edges represented as pairs of vertex indices.
        n (int): The number of dimensions.
        coord_labels (list of str): The coordinate labels.
        rotation_planes (list of tuple): The rotation planes.
        paused (bool): Indicates whether the animation is paused.
        anim (matplotlib.animation.FuncAnimation): The animation object.
    """

    def __init__(self, fig, ax, vertices_nd, edges, n, coord_labels, rotation_planes):
        """
        Initialize the AnimationManager.

        Parameters:
            fig (matplotlib.figure.Figure): The figure object.
            ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes for plotting.
            vertices_nd (np.ndarray): The n-dimensional vertices.
            edges (list of tuple): List of edges.
            n (int): The number of dimensions.
            coord_labels (list of str): The coordinate labels.
            rotation_planes (list of tuple): The rotation planes.
        """
        self.fig = fig
        self.ax = ax
        self.original_vertices = vertices_nd.copy()
        self.edges = edges
        self.n = n
        self.coord_labels = coord_labels
        self.rotation_planes = rotation_planes
        self.paused = False
        self.anim = None

    def animate(self, frame):
        """
        Animation function called for each frame.

        Parameters:
            frame (int): The current frame number.

        Returns:
            matplotlib.axes._subplots.Axes3DSubplot: The updated axes.
        """
        if self.paused:
            return self.ax

        # Update the rotation and redraw the plot
        update(self, frame)

        return self.ax

    def pause_resume(self, event):
        """
        Toggle between pausing and resuming the animation.

        Parameters:
            event: The button click event.
        """
        self.paused = not self.paused
        if self.paused:
            self.anim.event_source.stop()
        else:
            self.anim.event_source.start()

    def restart(self, event):
        """
        Restart the animation from the beginning.

        Parameters:
            event: The button click event.
        """
        self.anim.frame_seq = self.anim.new_frame_seq()
        self.paused = False
        self.anim.event_source.start()

    def start_animation(self):
        """
        Initialize and start the animation.
        """
        total_frames = 360 * len(self.rotation_planes)
        self.anim = animation.FuncAnimation(self.fig, self.animate,
                                            frames=total_frames,
                                            interval=50, blit=False, repeat=True)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="n-dimensional hypercube visualization")
    parser.add_argument('n', type=int, help='Dimension of the hypercube (n >= 4)')
    args = parser.parse_args()
    n = args.n

    if n < 4:
        print("Error: Dimension n must be 4 or greater.")
        sys.exit(1)

    # Generate coordinate labels, vertices, edges, and rotation planes
    coord_labels = get_coord_labels(n)
    vertices = generate_vertices(n)
    edges = generate_edges(vertices)
    rotation_planes = generate_rotation_planes(n)

    # Set up the plot
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    # Add interactive buttons
    pause_ax = fig.add_axes([0.7, 0.05, 0.1, 0.075])
    restart_ax = fig.add_axes([0.81, 0.05, 0.1, 0.075])

    pause_button = Button(pause_ax, 'Pause/Resume', color='gray', hovercolor='lightgray')
    restart_button = Button(restart_ax, 'Restart', color='gray', hovercolor='lightgray')

    # Initialize the animation manager
    manager = AnimationManager(fig, ax, vertices, edges, n, coord_labels, rotation_planes)

    # Connect buttons to their functions
    pause_button.on_clicked(manager.pause_resume)
    restart_button.on_clicked(manager.restart)

    # Start the animation
    manager.start_animation()

    plt.show()
