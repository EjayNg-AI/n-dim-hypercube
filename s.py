#!/usr/bin/env python3
"""
n-dimensional Hypersphere Visualization

This script provides a 3D visualization of an n-dimensional hypersphere (n-sphere)
rotating through its various rotation planes. The dimension 'n' is provided as a command-line
argument (n >= 3). The visualization includes interactive controls for pausing,
resuming, and restarting the animation.

Usage:
    python hypersphere_visualization.py n [--fixed_angles NUM] [--points NUM]

Example:
    python hypersphere_visualization.py 3 --fixed_angles 10 --points 100

Dependencies:
    - Python 3.x
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
    return [f'X{i+1}' for i in range(n)]

def spherical_to_cartesian(angles, radius=1):
    """
    Convert n-dimensional spherical coordinates to Cartesian coordinates.

    Parameters:
        angles (list or np.ndarray): List of n-1 angles [θ1, θ2, ..., θ_{n-1}].
        radius (float): The radius of the n-sphere.

    Returns:
        np.ndarray: Array of n Cartesian coordinates [x1, x2, ..., xn].
    """
    n = len(angles) + 1
    coords = np.zeros(n)
    sin_prod = radius
    for i in range(n - 1):
        coords[i] = sin_prod * np.cos(angles[i])
        sin_prod *= np.sin(angles[i])
    coords[-1] = sin_prod
    return coords

def generate_parallels_and_meridians(n, num_points, num_fixed_angles):
    """
    Generate points representing the parallels and meridians on an n-sphere.

    Parameters:
        n (int): The number of dimensions.
        num_points (int): Number of points to sample along each line.
        num_fixed_angles (int): Number of fixed angles to use for generating parallels and meridians.

    Returns:
        list of tuple: List of tuples containing arrays of points and their corresponding dimension index.
    """
    lines = []
    angles_ranges = []
    for i in range(n - 1):
        angles_ranges.append(np.linspace(0, 2* np.pi, num_points))
        
    fixed_angles_list = np.linspace(0, np.pi, num_fixed_angles)

    for i in range(n - 1):
        other_indices = list(range(n - 1))
        other_indices.remove(i)
        fixed_angles_combinations = list(product(fixed_angles_list, repeat=len(other_indices)))

        for fixed_values in fixed_angles_combinations:
            fixed_angles = [0] * (n - 1)
            for idx, angle_idx in enumerate(other_indices):
                fixed_angles[angle_idx] = fixed_values[idx]
            theta_i_range = angles_ranges[i]
            line_points = []
            for theta_i in theta_i_range:
                angles = fixed_angles.copy()
                angles[i] = theta_i
                point = spherical_to_cartesian(angles)
                line_points.append(point)
            lines.append((np.array(line_points), i))  # Include dimension index
    return lines

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
    if n <= 3:
        return vertices_projected
    # Iteratively project from nD to 3D
    for coord_index in reversed(range(3, n)):
        denominator = d - vertices_projected[:, coord_index]
        denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
        scale = 1 / denominator
        # Scale coordinates up to the current dimension
        vertices_projected = vertices_projected[:, :coord_index] * scale[:, np.newaxis]
    return vertices_projected[:, :3]

def draw_lines(ax, projected_lines, colors):
    """
    Draw the lines in the 3D plot.

    Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes to draw on.
        projected_lines (list of tuple): List of tuples containing arrays of 3D projected points and dimension index.
        colors (list of str): List of colors for each dimension.
    """
    for line, dim_index in projected_lines:
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=colors[dim_index % len(colors)], linewidth=1)

def update(manager, frame):
    """
    Update function for the animation.

    Parameters:
        manager (AnimationManager): The animation manager instance.
        frame (int): The current frame number.

    Returns:
        matplotlib.axes._subplots.Axes3DSubplot: The updated 3D axes.
    """
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

    # Generate rotation matrix
    rotation_matrix = get_rotation_matrix(angle, *current_plane, n)

    # Rotate the lines
    rotated_lines = []
    for line, dim_index in manager.original_lines:
        rotated_line = np.dot(line, rotation_matrix.T)
        rotated_lines.append((rotated_line, dim_index))

    # Project the rotated lines down to 3D
    projected_lines = []
    for rotated_line, dim_index in rotated_lines:
        projected_line = project_nd_to_3d(rotated_line)
        projected_lines.append((projected_line, dim_index))

    # Clear the axes and redraw
    ax.clear()
    draw_lines(ax, projected_lines, manager.colors)

    # Set plot parameters
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.axis('off')  # Hide axes for better visualization

    # Display rotation information
    rotation_info = f"Rotating in {coord_labels[current_plane[0]]}{coord_labels[current_plane[1]]}-plane"
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
        original_lines (list of tuple): The original n-dimensional lines with dimension indices.
        n (int): The number of dimensions.
        coord_labels (list of str): The coordinate labels.
        rotation_planes (list of tuple): The rotation planes.
        colors (list of str): Colors for each dimension.
        paused (bool): Indicates whether the animation is paused.
        anim (matplotlib.animation.FuncAnimation): The animation object.
    """

    def __init__(self, fig, ax, lines, n, coord_labels, rotation_planes):
        """
        Initialize the AnimationManager.

        Parameters:
            fig (matplotlib.figure.Figure): The figure object.
            ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes for plotting.
            lines (list of tuple): The n-dimensional lines with dimension indices.
            n (int): The number of dimensions.
            coord_labels (list of str): The coordinate labels.
            rotation_planes (list of tuple): The rotation planes.
        """
        self.fig = fig
        self.ax = ax
        self.original_lines = [(line.copy(), dim_index) for line, dim_index in lines]
        self.n = n
        self.coord_labels = coord_labels
        self.rotation_planes = rotation_planes
        self.colors = plt.cm.rainbow(np.linspace(0, 1, n - 1))
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
    parser = argparse.ArgumentParser(description="n-dimensional hypersphere visualization")
    parser.add_argument('n', type=int, help='Dimension of the hypersphere (n >= 3)')
    parser.add_argument('--fixed_angles', type=int, default=5, help='Number of fixed angles for parallels and meridians')
    parser.add_argument('--points', type=int, default=100, help='Number of points along each line')
    args = parser.parse_args()
    n = args.n
    num_fixed_angles = args.fixed_angles
    num_points = args.points

    if n < 3:
        print("Error: Dimension n must be 3 or greater.")
        sys.exit(1)

    # Generate coordinate labels, lines, and rotation planes
    coord_labels = get_coord_labels(n)
    lines = generate_parallels_and_meridians(n, num_points=num_points, num_fixed_angles=num_fixed_angles)
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
    manager = AnimationManager(fig, ax, lines, n, coord_labels, rotation_planes)

    # Connect buttons to their functions
    pause_button.on_clicked(manager.pause_resume)
    restart_button.on_clicked(manager.restart)

    # Start the animation
    manager.start_animation()

    plt.show()
