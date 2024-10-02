
# n-dimensional Hypercube Visualization

This project provides a 3D visualization of an n-dimensional hypercube rotating through its various rotation planes. The dimension `n` is provided as a command-line argument (n ≥ 4). The visualization includes interactive controls for pausing, resuming, and restarting the animation.

![Hypercube Visualization Screenshot](screenshot.png)

## Features

- **Dynamic Visualization:** Visualize hypercubes in any dimension `n` (n ≥ 4).
- **Rotational Animation:** Rotates through all possible rotation planes in n dimensions.
- **Interactive Controls:**
  - **Pause/Resume:** Pause or resume the animation.
  - **Restart:** Restart the animation from the beginning.
  - **Mouse Interaction:** Rotate the 3D view using your mouse.
- **Perspective Projection:** Projects from nD to 3D for intuitive visualization.
- **Consistent Output Size:** Maintains consistent output size for visualization.

## Requirements

- **Python >= 3.8 **
- **NumPy >=1.26.4 **
- **Matplotlib >=3.9.1 **

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/EjayNg-AI/n-dim-hypercube.git
   cd hypercube-visualization
   ```

2. **Install Dependencies**

   Use `pip` to install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script with the desired dimension `n` (n ≥ 4):

```bash
python hypercube_visualization.py n
```

Replace `n` with the desired dimension.

### Examples

- **Visualize a 4D hypercube (tesseract):**

  ```bash
  python hypercube_visualization.py 4
  ```

- **Visualize a 5D hypercube (penteract):**

  ```bash
  python hypercube_visualization.py 5
  ```

- **Visualize a 6D hypercube:**

  ```bash
  python hypercube_visualization.py 6
  ```

## How It Works

### Overview

The script generates an n-dimensional hypercube (where `n` is specified by the user) and animates its rotation through all possible rotation planes, projecting it down to 3D for visualization.

### **1. What is a Hypercube?**

A hypercube is a generalization of a 3-dimensional cube to higher dimensions. In *n* dimensions, a hypercube (also known as an *n*-cube) has the following properties:

- **Vertices:** Each vertex is represented by an *n*-tuple where each coordinate is either -1 or 1.
- **Edges:** Edges connect pairs of vertices that differ in exactly one coordinate.
- **Faces, Cells, etc.:** Higher-dimensional analogs of faces (2D), cells (3D), etc.

For example:
- **1D:** A line segment with 2 vertices.
- **2D:** A square with 4 vertices and 4 edges.
- **3D:** A cube with 8 vertices and 12 edges.
- **4D:** A tesseract with 16 vertices and 32 edges.

### **2. Constructing the n-Dimensional Hypercube**

#### **a. Generating Vertices**

**Function:** `generate_vertices(n)`

```python
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
```

**Explanation:**

- **Cartesian Product:** The `product([-1, 1], repeat=n)` generates all possible combinations of -1 and 1 across *n* dimensions. This results in \( 2^n \) vertices.
  
- **Example:** For \( n = 3 \), the vertices would be:
  \[
  \begin{align*}
  (-1, -1, -1),\ (-1, -1, 1),\ (-1, 1, -1),\ (-1, 1, 1),\\
  (1, -1, -1),\ (1, -1, 1),\ (1, 1, -1),\ (1, 1, 1)
  \end{align*}
  \]
  
- **Representation:** Each vertex is an *n*-dimensional point where each coordinate is either -1 or 1, centered at the origin.

#### **b. Generating Edges**

**Function:** `generate_edges(vertices)`

```python
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
```

**Explanation:**

- **Edge Definition:** In an *n*-dimensional hypercube, an edge exists between two vertices if they differ in exactly one coordinate.
  
- **Difference Calculation:** The difference between two vertices is calculated as `vertices[i] - vertices[j]`. Taking the absolute value and summing across all coordinates (`np.sum(np.abs(vertices[i] - vertices[j]))`) gives the total coordinate-wise difference.
  
- **Condition:** If the sum equals 2, it implies that exactly one coordinate differs by 2 (from -1 to 1 or vice versa), meaning they are connected by an edge.
  
- **Edge List:** The function returns a list of tuples, each representing a pair of vertex indices that form an edge.

- **Example:** In 3D, the cube has 12 edges, each connecting vertices that differ in exactly one coordinate.

#### **c. Generating Rotation Planes**

**Function:** `generate_rotation_planes(n)`

```python
def generate_rotation_planes(n):
    """
    Generate all possible rotation planes (pairs of axes) in n dimensions.

    Parameters:
        n (int): The number of dimensions.

    Returns:
        list of tuple: A list of axis index pairs representing rotation planes.
    """
    return list(combinations(range(n), 2))
```

**Explanation:**

- **Rotation Planes:** In *n*-dimensions, a rotation occurs within a 2D plane defined by any pair of axes. For example, in 4D, possible rotation planes include (X0, X1), (X0, X2), (X0, X3), (X1, X2), (X1, X3), and (X2, X3).

- **Combinations:** The `combinations(range(n), 2)` generates all possible unique pairs of axes indices without repetition.

- **Purpose:** These rotation planes are used to apply rotation transformations to the hypercube, allowing it to rotate through all possible 2D planes in *n*-dimensional space.

### **3. Projecting the Hypercube from n-Dimensions to 3D**

#### **Why Project?**

Directly visualizing higher-dimensional objects is not feasible, so we need a method to represent the n-dimensional hypercube in 3D space. The projection reduces the dimensionality while preserving as much of the structure as possible, enabling us to create meaningful visualizations.

#### **Projection Method Used:**

**Function:** `project_nd_to_3d(vertices_nd, d=2, epsilon=1e-5)`

```python
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
```

**Explanation:**

The projection process involves reducing the dimensionality from *n* to 3 in a stepwise manner using **perspective projection**. Here's a breakdown of the steps:

1. **Initial Setup:**
   - `vertices_projected` is a copy of the original n-dimensional vertices.
   - `n` is the number of dimensions.
   
2. **Iterative Projection:**
   - The projection is performed iteratively from the highest dimension down to the 4th dimension, reducing one dimension at each step until only 3 dimensions remain.
   - The loop iterates over `coord_index` from \( n-1 \) down to 3.

3. **Perspective Projection Formula:**
   - For each dimension being projected out, the perspective projection formula is applied:
     \[
     \text{scale} = \frac{1}{d - x_k}
     \]
     where \( x_k \) is the coordinate in the current dimension being projected.
     
   - **Scaling:** The remaining coordinates (up to `coord_index - 1`) are scaled by this factor:
     \[
     \text{new\_coords} = \text{old\_coords} \times \text{scale}
     \]
     
   - **Preventing Division by Zero:** The `epsilon` ensures that the denominator never becomes zero, avoiding infinite scaling.
   
4. **Final Projection:**
   - After all higher dimensions are projected out, only the first three coordinates remain, representing the 3D position.

**Mathematical Justification:**

Perspective projection simulates the way the human eye perceives depth, where objects further away appear smaller. By applying this iteratively, the script effectively "peels away" higher dimensions, projecting the remaining structure into 3D space while maintaining a sense of depth and perspective.

**Example:**

- **4D to 3D:**
  - Suppose a 4D vertex has coordinates \((x, y, z, w)\).
  - First, project the 4th dimension (w) onto the 3D space using the formula above.
  - The resulting 3D coordinates will be influenced by the value of w, creating a sense of depth.

- **5D and Beyond:**
  - The process extends similarly, sequentially projecting out each higher dimension into the 3D space.

### **4. Applying Rotations to the Hypercube**

#### **a. Rotation Matrix Generation**

**Function:** `get_rotation_matrix(angle, axis1, axis2, n)`

```python
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
```

**Explanation:**

- **Rotation Plane:** Defined by two axes, `axis1` and `axis2`. The rotation occurs within this 2D plane embedded in the n-dimensional space.

- **Identity Matrix:** Starts with an identity matrix of size \( n \times n \), representing no rotation.

- **Rotation in the Specified Plane:**
  - **Diagonal Elements:** The entries corresponding to `axis1` and `axis2` on the diagonal are set to \( \cos(\theta) \).
  - **Off-Diagonal Elements:** The cross terms are set to \( -\sin(\theta) \) and \( \sin(\theta) \), creating the rotation effect.
  
- **Result:** The matrix rotates any vector in the hyperplane defined by `axis1` and `axis2` by the specified angle while leaving other dimensions unchanged.

**Mathematical Basis:**

A rotation in the plane defined by axes \( i \) and \( j \) in n-dimensional space can be represented by an n x n matrix where the only non-identity components are in the \( i \)-th and \( j \)-th rows and columns, forming a 2D rotation matrix:
\[
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
\]
embedded within the n x n identity matrix.

#### **b. Applying the Rotation**

**Function:** `update(manager, frame)`

```python
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
```

**Explanation:**

- **Frame Calculation:**
  - The animation cycles through all rotation planes, spending 360 frames (degrees) on each plane.
  - `current_frame` determines the position within the total animation cycle.
  - `current_plane_idx` identifies which rotation plane is currently being animated.
  - `angle` is the rotation angle in radians for the current frame.

- **Applying Rotation:**
  - A rotation matrix is generated for the current plane and angle.
  - The original vertices are rotated using matrix multiplication (`np.dot(vertices_nd, rotation_matrix.T)`).

- **Projection:**
  - The rotated vertices are projected from n-dimensional space to 3D using the previously explained `project_nd_to_3d` function.

- **Rendering:**
  - The 3D plot (`ax`) is cleared and updated with the new set of edges and vertices.
  - Plot limits are set to \([-2, 2]\) in all three axes to maintain a consistent view.
  - Axes are hidden for aesthetic purposes.
  - Informational text about the current rotation plane and frame number is added.

- **Animation Flow:**
  - By iteratively rotating through all rotation planes and angles, the hypercube appears to rotate seamlessly in 3D space.

### **5. Visualization Components**

#### **a. Drawing Edges and Vertices**

**Functions:** `draw_edges(ax, projected_vertices, edges)` and `draw_vertices(ax, projected_vertices)`

```python
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
```

**Explanation:**

- **Edges:**
  - Each edge is a line connecting two vertices.
  - `Line3DCollection` is used to efficiently render all edges at once.
  - Edges are colored white with a linewidth of 1 for visibility against the black background.

- **Vertices:**
  - Vertices are rendered as scatter points.
  - They are colored white and sized appropriately (`s=20`) to stand out.

#### **b. Animation Manager**

**Class:** `AnimationManager`

```python
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
```

**Explanation:**

- **Purpose:** Manages the animation lifecycle, handling frame updates, pausing, resuming, and restarting.

- **Attributes:**
  - **Figure and Axes:** References to the Matplotlib figure and 3D axes.
  - **Vertices and Edges:** The hypercube's structure.
  - **Rotation Planes:** All possible planes through which the hypercube will rotate.
  - **Paused State:** A boolean indicating whether the animation is paused.
  - **Animation Object:** The `FuncAnimation` object controlling the animation.

- **Methods:**
  - **`animate`:** Called for each frame. It updates the hypercube's rotation unless paused.
  - **`pause_resume`:** Toggles the animation's paused state when the pause/resume button is clicked.
  - **`restart`:** Resets the animation to the beginning when the restart button is clicked.
  - **`start_animation`:** Initializes the `FuncAnimation` with the total number of frames and starts the animation loop.

#### **c. Interactive Controls**

**Buttons:** Pause/Resume and Restart

```python
# Add interactive buttons
pause_ax = fig.add_axes([0.7, 0.05, 0.1, 0.075])
restart_ax = fig.add_axes([0.81, 0.05, 0.1, 0.075])

pause_button = Button(pause_ax, 'Pause/Resume', color='gray', hovercolor='lightgray')
restart_button = Button(restart_ax, 'Restart', color='gray', hovercolor='lightgray')

# Connect buttons to their functions
pause_button.on_clicked(manager.pause_resume)
restart_button.on_clicked(manager.restart)
```

**Explanation:**

- **Button Placement:** The buttons are placed at specific positions within the figure using normalized coordinates.

- **Functionality:**
  - **Pause/Resume Button:** Allows the user to pause the animation, halting the rotation, and resume it.
  - **Restart Button:** Resets the animation to the starting frame, beginning the rotation sequence anew.

- **Interactivity:** Enhances user experience by providing control over the visualization, allowing exploration at the user's own pace.

### **6. Putting It All Together**

#### **Main Execution Flow**

```python
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
```

**Explanation:**

1. **Argument Parsing:**
   - The script expects a single command-line argument: the dimension *n* of the hypercube (with \( n \geq 4 \)).
   - If \( n < 4 \), the script exits with an error message.

2. **Generation of Hypercube Components:**
   - **Coordinate Labels:** Generated for informational purposes (e.g., X0, X1, X2, ...).
   - **Vertices and Edges:** Created using the aforementioned functions.
   - **Rotation Planes:** All possible pairs of axes are determined for rotation.

3. **Plot Setup:**
   - A Matplotlib figure is created with a black background.
   - A 3D subplot is added to the figure for rendering the hypercube.

4. **Interactive Buttons:**
   - Pause/Resume and Restart buttons are added to the figure.
   - These buttons are connected to the corresponding methods in the `AnimationManager`.

5. **Animation Initialization:**
   - An instance of `AnimationManager` is created, encapsulating all necessary components.
   - The animation is started using `manager.start_animation()`, which sets up the `FuncAnimation` loop.

6. **Display:**
   - `plt.show()` launches the interactive visualization window, displaying the rotating hypercube.

### **7. Summary of the Visualization Process**

1. **Initialization:**
   - The script begins by parsing the input dimension *n* and validating it.
   - It then generates all necessary components of the hypercube: vertices, edges, and rotation planes.

2. **Rotation:**
   - The hypercube is rotated through all possible 2D planes in the *n*-dimensional space.
   - For each rotation plane, the hypercube is rotated incrementally by 1 degree (converted to radians).

3. **Projection:**
   - After each rotation, the hypercube's vertices are projected from *n*-dimensional space to 3D using iterative perspective projection.
   - This projection maintains depth and perspective, allowing for a coherent 3D visualization.

4. **Rendering:**
   - The rotated and projected hypercube is rendered in the 3D subplot, with edges and vertices drawn in white against a black background.
   - Informational text displays the current rotation plane and frame number.

5. **Animation:**
   - The entire process is animated using Matplotlib's `FuncAnimation`, creating a smooth rotation effect.
   - Interactive buttons allow users to pause/resume or restart the animation at any time.

### Limitations

- **Performance:** The number of vertices and edges grows exponentially with `n` (`2^n` vertices and `n * 2^(n-1)` edges). This may affect performance for large `n`.
- **Visualization Clarity:** Higher-dimensional hypercubes may appear cluttered when projected to 3D due to overlapping edges.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Inspired by mathematical visualizations of higher-dimensional objects.
- Built using [Matplotlib](https://matplotlib.org/) and [NumPy](https://numpy.org/).