# Standard library imports
import sys
import time
import gc
import random
import os
import math
# Scientific computing and numerical operations
import numpy as np
import cupy as cp
import cupyx.scipy.linalg as cpx_la
from scipy import ndimage
import torch

import pandas as pd
import numba as nb
#from numba import njit





# Visualization and plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
import plotly.graph_objects as go


from IPython.display import display, clear_output



# System monitoring
import psutil

# Configuration
matplotlib.use('QtAgg')  # Set the matplotlib backend
plt.close('all')  # Close any existing plots
os.environ['CUPY_ACCELERATORS'] = 'cub,cutensor'  # Set CuPy accelerators

def free_gpu_memory(func):
    def wrapper_func(*args, **kwargs):
        retval = func(*args, **kwargs)
        cp._default_memory_pool.free_all_blocks()
        return retval
    return wrapper_func    


# Base Shape class
class Shape:
    def __init__(self, shape_id, n, group=None, priority=0):
        self.id = shape_id
        self.n = n
        self.group = group
        self.priority = priority
        self.value = None  # For calculated value (e.g., alpha)
        self.center = [0, 0, 0]  # Default center

    def to_tree_item(self):
        return f"{self.__class__.__name__.capitalize()} {self.id.split('_')[1]}"


# Sphere shape
class Sphere(Shape):
    def __init__(self, shape_id, center, radius, n, group=None, priority=0):
        super().__init__(shape_id, n, group, priority)
        self.center = center  # [x, y, z]
        self.radius = radius

# Cylinder shape
class Cylinder(Shape):
    def __init__(self, shape_id, center, radius, height, axis='z', n=1.5, group=None, priority=0):
        super().__init__(shape_id, n, group, priority)
        self.center = center
        self.radius = radius
        self.height = height
        self.axis = axis

# Ellipsoid shape
class Ellipsoid(Shape):
    def __init__(self, shape_id, center, semi_axes, n, axis='z', group=None, priority=0):
        super().__init__(shape_id, n, group, priority)
        self.center = center
        self.semi_axes = semi_axes  # [semi_axes_x, semi_axes_y, semi_axes_z]
        self.axis = axis

# Rectangle shape
class Rectangle(Shape):
    def __init__(self, shape_id, center, dimensions, n, group=None, priority=0):
        super().__init__(shape_id, n, group, priority)
        self.center = center
        self.dimensions = dimensions  # [dimensions_x, dimensions_y, dimensions_z]

# Prism shape
class Prism(Shape):
    def __init__(self, shape_id, center, radius, sides, height, n, axis='z', group=None, priority=0):
        super().__init__(shape_id, n, group, priority)
        self.center = center
        self.radius = radius
        self.sides = sides
        self.height = height
        self.axis = axis

# ShapeManager class
class ShapeManager:
    def __init__(self):
        self.shapes = {}  # {shape_id: shape_object}
        self.shape_counters = {}  # {shape_type: counter}
        self.available_ids = set()
        self.available_groups = set()
        self.next_group = 1

    def get_next_id(self, shape_type):
        if shape_type not in self.shape_counters:
            self.shape_counters[shape_type] = 1
        else:
            self.shape_counters[shape_type] += 1
        return f"{shape_type}_{self.shape_counters[shape_type]}"

    def get_next_group(self):
        if self.available_groups:
            return f"group_{self.available_groups.pop()}"
        else:
            group_num = self.next_group
            self.next_group += 1
            return f"group_{group_num}"

    def add_shape(self, shape_type, **params):
        shape_classes = {
            'sphere': Sphere,
            'cylinder': Cylinder,
            'ellipsoid': Ellipsoid,
            'rectangle': Rectangle,
            'prism': Prism
        }
        shape_class = shape_classes.get(shape_type)
        if not shape_class:
            raise ValueError(f"Unknown shape type: {shape_type}")
        shape_id = params.get('force_id', self.get_next_id(shape_type))
        shape = shape_class(shape_id, **params)
        self.shapes[shape_id] = shape
        return shape_id


    def remove_shape(self, shape_id):
        if shape_id in self.shapes:
            shape_type, shape_number = shape_id.rsplit('_', 1)
            del self.shapes[shape_id]
            self.available_ids.add(shape_id)
            self.shape_counters[shape_type].add(int(shape_number))
        else:
            raise ValueError(f"Shape ID {shape_id} not found")

            
    def remove_group(self, group_name):
        shapes_to_remove = [shape_id for shape_id, shape in self.shapes.items() if shape.group == group_name]
        
        for shape_id in shapes_to_remove:
            self.remove_shape(shape_id)
        
        if group_name in self.available_groups:
            self.available_groups.remove(group_name)
        
        # Reset group counter if needed
        group_number = int(group_name.split('_')[1])
        if group_number < self.next_group:
            self.next_group = group_number
        
        self.available_groups.add(group_number)
        
    def group_shapes(self, shape_ids, group_name):
        if group_name in self.available_groups:
            raise ValueError(f"Group {group_name} already exists")
        
        for shape_id in shape_ids:
            if shape_id not in self.shapes:
                raise ValueError(f"Shape ID {shape_id} not found")
            
            self.shapes[shape_id].group = group_name
        
        self.available_groups.add(group_name)
    
    def ungroup_shapes(self, group_name):
        if group_name not in self.available_groups:
            raise ValueError(f"Group {group_name} doesn't exist")
        
        for shape_id, shape in self.shapes.items():
            if shape.group == group_name:
                shape.group = None
        
        self.available_groups.remove(group_name)

    # Other methods like connect_shapes, disconnect_shapes can be added here

    def add_lattice(self, shape_type, shape_params, lattice_type='square',
                    spacing=100, size=(3,3,3),
                    x_offset=0, y_offset=0, z_offset=0, force_group=None, priority=0):
        if isinstance(spacing, (int, float)):
            spacing = (spacing, spacing, spacing)
        elif len(spacing) != 3:
            raise ValueError("Spacing must be single number or (dx,dy,dz) tuple")
        base_pos = shape_params.get('center', [0, 0, 0])
        group_name = force_group if force_group else self.get_next_group()
    
        # Get all coordinates from the helper function
        coords = self._generate_lattice_coords(
            lattice_type, spacing, size, base_pos, x_offset, y_offset, z_offset
        )
    
        # Create shapes at each coordinate
        shape_classes = {
            'sphere': Sphere,
            'cylinder': Cylinder,
            'ellipsoid': Ellipsoid,
            'rectangle': Rectangle,
            'prism': Prism
        }
        shape_class = shape_classes.get(shape_type)
        if not shape_class:
            raise ValueError(f"Unknown shape type: {shape_type}")
    
        for coord in coords:
            params = shape_params.copy()
            params['center'] = coord.tolist()
            params['group'] = group_name
            shape_id = self.get_next_id(shape_type)  # Pass shape_type here
            shape = shape_class(shape_id, **params)
            self.shapes[shape_id] = shape


    def _generate_lattice_coords(self, lattice_type, spacing, size,
                                 base_pos, x_offset, y_offset, z_offset):
        nx, ny, nz = size
        dx, dy, dz = spacing
        base_x, base_y, base_z = base_pos

        if lattice_type.lower() == 'square':
            x = np.arange(nx) * dx + base_x + x_offset
            y = np.arange(ny) * dy + base_y + y_offset
            z = np.arange(nz) * dz + base_z + z_offset
            x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        else:  # hexagonal
            x = np.arange(nx) * dx + base_x + x_offset
            y = np.arange(ny) * dy * 0.866 + base_y + y_offset
            z = np.arange(nz) * dz + base_z + z_offset
            x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
            # Adjust x for each alternate y row
            y_indices = np.arange(ny)
            x_adjustments = ((y_indices % 2) * dx/2)
            x_grid += x_adjustments[np.newaxis, :, np.newaxis]

        coords = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)
        return coords

    def add_grating(self, base_length, base_width, base_height,
                    grate_length, grate_width, grate_height,
                    period, num_grates, n_material=1.5, force_group=None, priority=0):

        group_name = force_group if force_group else self.get_next_group()

        # Base platform
        shape_id = self.get_next_id()
        rectangle = Rectangle(
            shape_id=shape_id,
            center=[0, 0, base_height/2],
            dimensions=[base_length, base_width, base_height],
            n=n_material,
            group=group_name,
            priority=priority
        )
        self.shapes[shape_id] = rectangle

        # Grating bars
        start_y = -period*(num_grates-1)/2
        for i in range(num_grates):
            shape_id = self.get_next_id()
            rectangle = Rectangle(
                shape_id=shape_id,
                center=[0, start_y + i*period, base_height + grate_height/2],
                dimensions=[grate_length, grate_width, grate_height],
                n=n_material,
                group=group_name,
                priority=priority
            )
            self.shapes[shape_id] = rectangle

    def process_shapes(self, k, E_direction, E_polarization, lattice_spacing, adjust_all=True):
        if not self.shapes:
            print("No shapes exist to process.")
            return
        
        # Sort shapes by priority (higher priority comes last)
        sorted_shapes = sorted(self.shapes.values(), key=lambda s: s.priority)
        
        # Calculate unique ns
        unique_ns = set(shape.n for shape in sorted_shapes)
        alpha_map = {}
        for n in unique_ns:
            alpha_value = calculate_alpha(n, lattice_spacing, k, E_direction, E_polarization, method='LDR')
            print(f"n = {n:.3f}: α = {alpha_value:.3e}")
            alpha_map[n] = alpha_value

        
        # Update shapes
        for shape in sorted_shapes:
            shape.value = alpha_map[shape.n]
        
        # Adjust centers
        centers = np.array([shape.center for shape in sorted_shapes])
        if len(sorted_shapes) == 1 or adjust_all:
            # Use ceil instead of round to always go up
            adjusted_centers = lattice_spacing * np.ceil(centers / lattice_spacing)
        else:
            # Find largest shape
            max_size = 0
            largest_shape = None
            for shape in sorted_shapes:
                size = getattr(shape, 'radius', 0)
                if hasattr(shape, 'dimensions'):
                    size = max(shape.dimensions)
                elif hasattr(shape, 'semi_axes'):
                    size = max(shape.semi_axes)
                if size > max_size:
                    max_size = size
                    largest_shape = shape
            if largest_shape is not None:
                base_center = np.array(largest_shape.center)
                # Use ceil instead of round to always go up
                adjustment = lattice_spacing * np.ceil(base_center / lattice_spacing) - base_center
                adjusted_centers = centers + adjustment
            else:
                adjusted_centers = centers  # No adjustment
        
        # Update centers
        for shape, new_center in zip(sorted_shapes, adjusted_centers):
            old_center = np.array(shape.center)
            shape.center = new_center.tolist()
            #if not np.array_equal(old_center, new_center):
            #    print(f"Shape center adjusted: Old center: {old_center}, New center: {new_center}")


    def calculate_grid_extents(self, lattice_spacing, max_x=1e9, max_y=1e9, max_z=1e9):
        if not self.shapes:
            return 0, 0, 0
        
        max_coords = np.zeros(3)
        
        for shape in self.shapes.values():
            center = np.array(shape.center)
            
            # Calculate the maximum extent in any direction from the center
            max_extent = max(
                getattr(shape, 'radius', 0),
                getattr(shape, 'height', 0)/2,
                max(getattr(shape, 'semi_axes', [0, 0, 0])),
                max([dim/2 for dim in getattr(shape, 'dimensions', [0, 0, 0])])
            )
            
            # Special handling for prisms with many sides
            if isinstance(shape, Prism) and shape.sides > 2:
                # For regular polygons, the distance from center to vertex can be larger than radius
                # The correction factor is 1/cos(π/sides)
                correction = 1.0 / np.cos(np.pi / shape.sides)
                prism_extent = shape.radius * correction
                max_extent = max(max_extent, prism_extent)
            
            # For non-spherical shapes, we need to consider the orientation
            if isinstance(shape, Cylinder) or isinstance(shape, Prism):
                # Create an extent vector with the appropriate dimensions
                axis_index = {'x': 0, 'y': 1, 'z': 2}[shape.axis]
                extent_vector = np.zeros(3)
                extent_vector[axis_index] = shape.height / 2
                
                # For the perpendicular directions, use the calculated max_extent
                perp_indices = [(axis_index + 1) % 3, (axis_index + 2) % 3]
                for idx in perp_indices:
                    extent_vector[idx] = max_extent
                    
                # Add this to the center to get the maximum coordinates
                shape_max_coords = center + extent_vector
            else:
                # For other shapes, use the max_extent in all directions
                shape_max_coords = center + max_extent
            
            max_coords = np.maximum(max_coords, shape_max_coords)
        
        max_coords = np.minimum(max_coords, [max_x, max_y, max_z])
        
        # Add a small buffer (1 lattice spacing) to ensure we capture the entire shape
        max_coords += lattice_spacing
        
        nx = min(int(np.ceil(max_coords[0] / lattice_spacing)), int(np.ceil(max_x / lattice_spacing)))
        ny = min(int(np.ceil(max_coords[1] / lattice_spacing)), int(np.ceil(max_y / lattice_spacing)))
        nz = min(int(np.ceil(max_coords[2] / lattice_spacing)), int(np.ceil(max_z / lattice_spacing)))
        
        return nx, ny, nz


def calculate_alpha(refractive_index, lattice_spacing, wave_number, incident_direction=None, polarization_direction=None, method='LDR'):
    # Ensure proper types
    n = np.complex128(refractive_index)
    d = np.float64(lattice_spacing)
    k = np.float64(wave_number)
    i_dir = np.array(incident_direction, dtype=np.float64) if incident_direction is not None else None
    p_dir = np.array(polarization_direction, dtype=np.float64) if polarization_direction is not None else None
    
    # Calculate base CM polarizability
    eps = n**2
    alpha_cm = (3 * d**3 / (4 * np.pi)) * (eps - 1) / (eps + 2)
    
    if method == 'CM':
        return alpha_cm
        
    # RR term used in both RR and LDR
    rr_term = (2/3) * 1j * (k*d)**3
    
    if method == 'RR':
        return alpha_cm / (1 + (alpha_cm/d**3) * (-rr_term))
        
    if method == 'LDR':
        if i_dir is None or p_dir is None:
            raise ValueError("incident_direction and polarization_direction required for LDR method")
            
        b1 = -1.891531
        b2 = 0.1648469
        b3 = -1.7700004
        
        S = np.sum(np.dot(i_dir, p_dir)**2)
        ldr_term = (b1 + n**2*b2 + n**2*b3*S) * (k*d)**2
        
        return alpha_cm / (1 + (alpha_cm/d**3) * (ldr_term - rr_term))
        
    raise ValueError("method must be 'CM', 'RR', or 'LDR'")

@nb.jit(nopython=True, parallel=True, cache=True)
def sphere(grid_points, center_x, center_y, center_z, radius, epsilon):
    result = np.zeros(len(grid_points), dtype=np.bool_)
    for i in range(len(grid_points)):
        dx = grid_points[i,0] - center_x
        dy = grid_points[i,1] - center_y
        dz = grid_points[i,2] - center_z
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        result[i] = (distance < radius) or (abs(distance - radius) < epsilon)
    return result

@nb.jit(nopython=True, parallel=True, cache=True)
def ellipsoid(grid_points, center_x, center_y, center_z, semi_axes_x, semi_axes_y, semi_axes_z, epsilon):
    result = np.zeros(len(grid_points), dtype=np.bool_)
    for i in range(len(grid_points)):
        dx = (grid_points[i,0] - center_x) / semi_axes_x
        dy = (grid_points[i,1] - center_y) / semi_axes_y
        dz = (grid_points[i,2] - center_z) / semi_axes_z
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        result[i] = distance <= 1.0 + epsilon
    return result

@nb.jit(nopython=True, parallel=True, cache=True)
def rectangle(grid_points, center_x, center_y, center_z, dimensions_x, dimensions_y, dimensions_z, epsilon):
    result = np.zeros(len(grid_points), dtype=np.bool_)
    half_x, half_y, half_z = dimensions_x / 2, dimensions_y / 2, dimensions_z / 2
    for i in range(len(grid_points)):
        dx = abs(grid_points[i,0] - center_x)
        dy = abs(grid_points[i,1] - center_y)
        dz = abs(grid_points[i,2] - center_z)
        inside = (dx <= half_x) and (dy <= half_y) and (dz <= half_z)
        on_edge = (abs(dx - half_x) < epsilon and dy <= half_y and dz <= half_z) or \
                 (abs(dy - half_y) < epsilon and dx <= half_x and dz <= half_z) or \
                 (abs(dz - half_z) < epsilon and dx <= half_x and dy <= half_y)
        result[i] = inside or on_edge
    return result

@nb.jit(nopython=True, parallel=True, cache=True)
def cylinder(grid_points, center_x, center_y, center_z, radius, height, axis, epsilon):
    result = np.zeros(len(grid_points), dtype=np.bool_)
    for i in range(len(grid_points)):
        if axis == 'z':
            dx = grid_points[i,0] - center_x
            dy = grid_points[i,1] - center_y
            dz = abs(grid_points[i,2] - center_z)
        elif axis == 'y':
            dx = grid_points[i,0] - center_x
            dy = abs(grid_points[i,1] - center_y)
            dz = grid_points[i,2] - center_z
        else:  # x axis
            dx = abs(grid_points[i,0] - center_x)
            dy = grid_points[i,1] - center_y
            dz = grid_points[i,2] - center_z
        
        radius_dist = np.sqrt(dx*dx + dy*dy)
        half_height = height / 2
        
        inside_radius = radius_dist <= radius
        on_radius = abs(radius_dist - radius) < epsilon
        within_height = dz <= half_height
        on_cap = abs(dz - half_height) < epsilon
        
        result[i] = (inside_radius and within_height) or \
                   (on_radius and within_height) or \
                   (inside_radius and on_cap)
    return result

@nb.jit(nopython=True, parallel=True, cache=True)
def prism(grid_points, center_x, center_y, center_z, radius, height, sides, axis, epsilon):
    result = np.zeros(len(grid_points), dtype=np.bool_)
    
    vertices = np.zeros((int(sides), 2))
    for i in range(int(sides)):
        angle = 2 * np.pi * i / sides
        if axis == 'z':
            vertices[i,0] = center_x + radius * np.cos(angle)
            vertices[i,1] = center_y + radius * np.sin(angle)
        elif axis == 'y':
            vertices[i,0] = center_x + radius * np.cos(angle)
            vertices[i,1] = center_z + radius * np.sin(angle)
        else:  # x axis
            vertices[i,0] = center_y + radius * np.cos(angle)
            vertices[i,1] = center_z + radius * np.sin(angle)
    
    for i in range(len(grid_points)):
        if axis == 'z':
            point_2d = np.array([grid_points[i,0], grid_points[i,1]])
            height_coord = grid_points[i,2]
            center_height = center_z
        elif axis == 'y':
            point_2d = np.array([grid_points[i,0], grid_points[i,2]])
            height_coord = grid_points[i,1]
            center_height = center_y
        else:  # x axis
            point_2d = np.array([grid_points[i,1], grid_points[i,2]])
            height_coord = grid_points[i,0]
            center_height = center_x
        
        half_height = height / 2
        height_dist = abs(height_coord - center_height)
        within_height = height_dist <= half_height
        on_cap = abs(height_dist - half_height) < epsilon
    
        inside = False
        on_edge = False
        n_vertices = len(vertices)
        
        for j in range(n_vertices):
            j2 = (j + 1) % n_vertices
            xi, yi = vertices[j]
            xj, yj = vertices[j2]
            
            dx = xj - xi
            dy = yj - yi
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > epsilon:
                dx = dx / length
                dy = dy / length
                vx = point_2d[0] - xi
                vy = point_2d[1] - yi
                proj = vx*dx + vy*dy
                dist = abs(vx*dy - vy*dx)
                if dist < epsilon and proj >= -epsilon and proj <= length + epsilon:
                    on_edge = True
                    break
            
            if ((yi > point_2d[1]) != (yj > point_2d[1])) and \
               (point_2d[0] < (xj - xi) * (point_2d[1] - yi) / (yj - yi) + xi):
                inside = not inside
    
        result[i] = (inside or on_edge) and (within_height or on_cap)
    
    return result

def create_shape_data(shape):
    return {
        'center_x': float(shape.center[0]),
        'center_y': float(shape.center[1]),
        'center_z': float(shape.center[2]),
        'radius': float(getattr(shape, 'radius', 0)),
        'height': float(getattr(shape, 'height', 0)),
        'sides': float(getattr(shape, 'sides', 0)),
        'semi_axes_x': float(getattr(shape, 'semi_axes', [0,0,0])[0]),
        'semi_axes_y': float(getattr(shape, 'semi_axes', [0,0,0])[1]),
        'semi_axes_z': float(getattr(shape, 'semi_axes', [0,0,0])[2]),
        'dimensions_x': float(getattr(shape, 'dimensions', [0,0,0])[0]),
        'dimensions_y': float(getattr(shape, 'dimensions', [0,0,0])[1]),
        'dimensions_z': float(getattr(shape, 'dimensions', [0,0,0])[2]),
        'axis': getattr(shape, 'axis', '')
    }


shape_calculators = {
    Sphere: lambda grid, **data: sphere(
        grid, data['center_x'], data['center_y'], data['center_z'], 
        data['radius'], 1e-7
    ),
    Ellipsoid: lambda grid, **data: ellipsoid(
        grid, data['center_x'], data['center_y'], data['center_z'],
        data['semi_axes_x'], data['semi_axes_y'], data['semi_axes_z'], 1e-7
    ),
    Rectangle: lambda grid, **data: rectangle(
        grid, data['center_x'], data['center_y'], data['center_z'],
        data['dimensions_x'], data['dimensions_y'], data['dimensions_z'], 1e-7
    ),
    Cylinder: lambda grid, **data: cylinder(
        grid, data['center_x'], data['center_y'], data['center_z'],
        data['radius'], data['height'], data['axis'], 1e-7
    ),
    Prism: lambda grid, **data: prism(
        grid, data['center_x'], data['center_y'], data['center_z'],
        data['radius'], data['height'], data['sides'], data['axis'], 1e-7
    )
}

def mark_shapes(grid_shape, lattice_spacing, shapes, voxel_center=True):
    nx, ny, nz = grid_shape
    offset = 0.5 if voxel_center else 0
    value_array = np.zeros(grid_shape, dtype=np.complex128)  # Changed from complex64
    epsilon = 1e-7
    buffer = 1

    for shape in shapes:
        shape_data = create_shape_data(shape)
        center = np.array([shape_data['center_x'], shape_data['center_y'], shape_data['center_z']], dtype=np.float64)
        
        max_dist = max(
            getattr(shape, 'radius', 0),
            getattr(shape, 'height', 0)/2,
            max(getattr(shape, 'semi_axes', [0,0,0])),
            max(getattr(shape, 'dimensions', [0,0,0]))/2
        )
        
        max_grid_dist = int(float(max_dist + epsilon) / lattice_spacing) + buffer
        center_idx = (center / lattice_spacing).astype(int)
        
        x_min = max(0, center_idx[0] - max_grid_dist)
        x_max = min(nx, center_idx[0] + max_grid_dist + 1)
        y_min = max(0, center_idx[1] - max_grid_dist)
        y_max = min(ny, center_idx[1] + max_grid_dist + 1)
        z_min = max(0, center_idx[2] - max_grid_dist)
        z_max = min(nz, center_idx[2] + max_grid_dist + 1)
        
        x = (np.arange(x_min, x_max) + offset) * lattice_spacing
        y = (np.arange(y_min, y_max) + offset) * lattice_spacing
        z = (np.arange(z_min, z_max) + offset) * lattice_spacing
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        local_grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        calculator = shape_calculators[type(shape)]
        inside = calculator(local_grid, **shape_data)
            
        local_value_array = np.zeros(len(local_grid), dtype=np.complex128)  # Changed from complex64
        local_value_array[inside] = shape.value
        
        local_value_array_reshaped = local_value_array.reshape(x_max-x_min, y_max-y_min, z_max-z_min)
        
        value_array_slice = value_array[x_min:x_max, y_min:y_max, z_min:z_max]
        mask = local_value_array_reshaped != 0
        value_array_slice[mask] = local_value_array_reshaped[mask]
    
    return value_array

def mark_shapes_n(grid_shape, lattice_spacing, shapes, voxel_center=True):
    nx, ny, nz = grid_shape
    offset = 0.5 if voxel_center else 0
    n_array = np.zeros(grid_shape, dtype=np.complex128)  # Array for refractive indices
    epsilon = 1e-7
    buffer = 1

    for shape in shapes:
        shape_data = create_shape_data(shape)
        center = np.array([shape_data['center_x'], shape_data['center_y'], shape_data['center_z']], dtype=np.float64)
        
        max_dist = max(
            getattr(shape, 'radius', 0),
            getattr(shape, 'height', 0)/2,
            max(getattr(shape, 'semi_axes', [0,0,0])),
            max(getattr(shape, 'dimensions', [0,0,0]))/2
        )
        
        max_grid_dist = int(float(max_dist + epsilon) / lattice_spacing) + buffer
        center_idx = (center / lattice_spacing).astype(int)
        
        x_min = max(0, center_idx[0] - max_grid_dist)
        x_max = min(nx, center_idx[0] + max_grid_dist + 1)
        y_min = max(0, center_idx[1] - max_grid_dist)
        y_max = min(ny, center_idx[1] + max_grid_dist + 1)
        z_min = max(0, center_idx[2] - max_grid_dist)
        z_max = min(nz, center_idx[2] + max_grid_dist + 1)
        
        x = (np.arange(x_min, x_max) + offset) * lattice_spacing
        y = (np.arange(y_min, y_max) + offset) * lattice_spacing
        z = (np.arange(z_min, z_max) + offset) * lattice_spacing
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        local_grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        calculator = shape_calculators[type(shape)]
        inside = calculator(local_grid, **shape_data)
            
        local_n_array = np.zeros(len(local_grid), dtype=np.complex128)
        local_n_array[inside] = shape.n  # Using n instead of value
        
        local_n_array_reshaped = local_n_array.reshape(x_max-x_min, y_max-y_min, z_max-z_min)
        
        n_array_slice = n_array[x_min:x_max, y_min:y_max, z_min:z_max]
        mask = local_n_array_reshaped != 0
        n_array_slice[mask] = local_n_array_reshaped[mask]
    
    return n_array

def mark_shapes_gpu(grid_shape, lattice_spacing, shapes, voxel_center=True):
    nx, ny, nz = grid_shape
    offset = 0.5 if voxel_center else 0
    
    x = (cp.arange(nx) + offset) * lattice_spacing
    y = (cp.arange(ny) + offset) * lattice_spacing
    z = (cp.arange(nz) + offset) * lattice_spacing
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    grid_points = cp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    value_array = cp.zeros(nx * ny * nz, dtype=cp.complex64)
    epsilon = 1e-7
    
    for shape in shapes:
        if isinstance(shape, Sphere):
            displacement = grid_points - cp.asarray(shape.center)
            distances = cp.sqrt((displacement ** 2).sum(axis=1))
            inside_points = distances < shape.radius
            surface_points = cp.abs(distances - shape.radius) < epsilon
            inside = inside_points | surface_points
            
        elif isinstance(shape, Ellipsoid):
            scaled_points = (grid_points - cp.asarray(shape.center)) / cp.asarray(shape.radii)
            distances = cp.sqrt((scaled_points ** 2).sum(axis=1))
            inside = distances <= 1.0 + epsilon
            
        elif isinstance(shape, Rectangle):
            inside = cp.all((grid_points >= cp.asarray(shape.min_corner)) & 
                          (grid_points <= cp.asarray(shape.max_corner)), axis=1)
            on_edge = cp.zeros_like(inside)
            for dim in range(3):
                min_face = cp.abs(grid_points[:,dim] - shape.min_corner[dim]) < epsilon
                max_face = cp.abs(grid_points[:,dim] - shape.max_corner[dim]) < epsilon
                other_dims = cp.all((grid_points >= cp.asarray(shape.min_corner) - epsilon) & 
                                  (grid_points <= cp.asarray(shape.max_corner) + epsilon), axis=1)
                on_edge |= (min_face | max_face) & other_dims
            inside |= on_edge
            
        elif isinstance(shape, Cylinder):
            axis_index = {'x': 0, 'y': 1, 'z': 2}[shape.axis]
            other_axes = [i for i in range(3) if i != axis_index]
            points_2d = grid_points[:, other_axes]
            center_2d = cp.asarray([shape.base_center[i] for i in other_axes])
            vx = points_2d[:,0] - center_2d[0]
            vy = points_2d[:,1] - center_2d[1]
            distances = cp.sqrt(vx*vx + vy*vy)
            edge_points = cp.abs(distances - shape.radius) < epsilon
            inside_base = distances <= shape.radius
            inside_base |= edge_points
            axis_coordinate = grid_points[:, axis_index]
            axis_start = shape.base_center[axis_index]
            within_height = (axis_coordinate >= axis_start - epsilon) & (axis_coordinate <= axis_start + shape.height + epsilon)
            inside = inside_base & within_height
            
        elif isinstance(shape, ArbitraryPrism):
            axis_index = {'x': 0, 'y': 1, 'z': 2}[shape.axis]
            other_axes = [i for i in range(3) if i != axis_index]
            points_2d = grid_points[:, other_axes]
            vertices = cp.asarray(shape.generate_base_vertices())
            base_vertices_2d = vertices[:, other_axes]
            inside_base = cp.zeros(len(points_2d), dtype=bool)
            edge_points = cp.zeros(len(points_2d), dtype=bool)
            n_vertices = len(base_vertices_2d)
            
            for i in range(n_vertices):
                j = (i + 1) % n_vertices
                xi, yi = base_vertices_2d[i]
                xj, yj = base_vertices_2d[j]
                dx = xj - xi
                dy = yj - yi
                length = cp.sqrt(dx*dx + dy*dy)
                
                if length > epsilon:
                    dx = dx / length
                    dy = dy / length
                    vx = points_2d[:,0] - xi
                    vy = points_2d[:,1] - yi
                    proj = vx*dx + vy*dy
                    dist = cp.abs(vx*dy - vy*dx)
                    on_edge = (dist < epsilon) & (proj >= -epsilon) & (proj <= length + epsilon)
                    edge_points |= on_edge
                
                intersect = ((yi > points_2d[:,1]) != (yj > points_2d[:,1])) & (points_2d[:,0] < (xj - xi) * (points_2d[:,1] - yi) / (yj - yi) + xi)
                inside_base ^= intersect
            
            inside_base |= edge_points
            axis_coordinate = grid_points[:, axis_index]
            prism_start = shape.center[axis_index]
            prism_end = prism_start + shape.height
            within_height = (axis_coordinate >= prism_start - epsilon) & (axis_coordinate <= prism_end + epsilon)
            inside = inside_base & within_height
        
        else:
            raise ValueError(f"Unsupported shape type: {type(shape)}")
        
        value_array[inside] = shape.value
    
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(value_array.reshape(grid_shape))


def create_alpha_and_n_arrays(shape_manager, lattice_spacing, voxel_center=True):
    nx, ny, nz = shape_manager.calculate_grid_extents(lattice_spacing)
    grid_shape = (nx, ny, nz)
    print('start grid shape')
    print(grid_shape)
    
    alpha_array = mark_shapes(grid_shape, lattice_spacing, shape_manager.shapes.values(), voxel_center=voxel_center)
    n_array = mark_shapes_n(grid_shape, lattice_spacing, shape_manager.shapes.values(), voxel_center=voxel_center)
    
    alpha_array = trim_zero_faces(alpha_array)
    n_array = trim_zero_faces(n_array)
    
    # Print non-zero elements in each x-slice
    #for i in range(alpha_array.shape[0]):
    #    non_zero_count = np.count_nonzero(alpha_array[i])
    #    print(f"Slice {i}: {non_zero_count} non-zero elements")
    print('end grid shape')
    print(alpha_array.shape)    
    return alpha_array, n_array, alpha_array.shape
def trim_zero_faces(array):
    # Get mask of non-zero values along each axis
    x_mask = np.any(np.any(array != 0, axis=2), axis=1)
    y_mask = np.any(np.any(array != 0, axis=2), axis=0)
    z_mask = np.any(np.any(array != 0, axis=1), axis=0)
    
    # Find bounds using boolean masks
    x_start = np.argmax(x_mask)
    x_end = len(x_mask) - np.argmax(x_mask[::-1])
    y_start = np.argmax(y_mask)
    y_end = len(y_mask) - np.argmax(y_mask[::-1])
    z_start = np.argmax(z_mask)
    z_end = len(z_mask) - np.argmax(z_mask[::-1])
    
    # Extract trimmed region
    trimmed = array[x_start:x_end, y_start:y_end, z_start:z_end]
    
    # Force even dimensions by rounding up odd sizes
    shape = trimmed.shape
    new_shape = tuple(s + (s % 2) for s in shape)
    
    if new_shape != shape:
        padded = np.zeros(new_shape, dtype=array.dtype)  # Preserve original dtype
        slices = tuple(slice(0, s) for s in shape)
        padded[slices] = trimmed
        return padded
    return trimmed

def is_good_fft_size(n):
    """Check if size is composed ONLY of factors 2,3,5,7,11,13,17,19."""
    if n <= 0:
        return False
    
    for prime in [2, 3, 5, 7, 11, 13, 17, 19]:
        while n % prime == 0:
            n //= prime
    return n == 1

def find_next_even_good_size(n):
    """
    Find the next size that is:
    - At least n + 2
    - Even
    - Only composed of the prime factors [2,3,5,7,11,13,17,19]
    """
    # Must be at least n+2:
    n += 2
    # Ensure it is even:
    if n % 2 == 1:
        n += 1
    # Now increment in steps of 2 until it's a good FFT size:
    while not is_good_fft_size(n):
        n += 2
    return n

def optimize_array_size(array):
    """
    Takes a 3D NumPy array, finds the next 'friendly' FFT size for each dimension
    (which must be even and at least 2 greater), then pads the array with zeros
    around (both sides) without cutting off data.
    """
    nx, ny, nz = array.shape

    # Find the next even good size for each dimension
    next_x = find_next_even_good_size(nx)
    next_y = find_next_even_good_size(ny)
    next_z = find_next_even_good_size(nz)

    # Calculate how much we need to pad on each end
    diff_x = next_x - nx
    diff_y = next_y - ny
    diff_z = next_z - nz

    pad_x_left = diff_x // 2
    pad_x_right = diff_x - pad_x_left
    pad_y_left = diff_y // 2
    pad_y_right = diff_y - pad_y_left
    pad_z_left = diff_z // 2
    pad_z_right = diff_z - pad_z_left

    # Create padded array
    padded = np.pad(array, 
                    pad_width=((pad_x_left, pad_x_right),
                               (pad_y_left, pad_y_right),
                               (pad_z_left, pad_z_right)),
                    mode='constant', 
                    constant_values=0)

    return padded




def is_good_fft_size(n):
    """Check if size is composed ONLY of factors 2,3,5,7,11,13,17,19"""
    if n <= 0:
        return False
    
    for prime in [2,3,5,7,11,13,17,19]:
        while n % prime == 0:
            n //= prime
    return n == 1





# Single precision kernel
greens_kernel_single = cp.ElementwiseKernel(
    'int32 nx, int32 ny, int32 nz, float32 k, float32 lattice_spacing, int32 component',
    'raw complex64 interaction_matrix',
    '''
    int ix = i / (2 * ny * 2 * nz);
    int iy = (i / (2 * nz)) % (2 * ny);
    int iz = i % (2 * nz);
    float positions[3] = {
        ix * lattice_spacing,
        iy * lattice_spacing,
        iz * lattice_spacing
    };
    if (ix >= nx) positions[0] -= 2 * nx * lattice_spacing;
    if (iy >= ny) positions[1] -= 2 * ny * lattice_spacing;
    if (iz >= nz) positions[2] -= 2 * nz * lattice_spacing;
    
    if (ix == 0 && iy == 0 && iz == 0) {
        interaction_matrix[i] = complex<float>(0, 0);
        return;
    }
    
    const float r_squared = positions[0] * positions[0] +
                          positions[1] * positions[1] +
                          positions[2] * positions[2];
    const float r = sqrt(r_squared);
    const float inv_r = 1.0f / r;
    const float directions[3] = {
        positions[0] * inv_r,
        positions[1] * inv_r,
        positions[2] * inv_r
    };
    
    float normalized_component;
    if (component == 0) normalized_component = directions[0] * directions[0];      // xx
    else if (component == 1) normalized_component = directions[0] * directions[1]; // xy
    else if (component == 2) normalized_component = directions[0] * directions[2]; // xz
    else if (component == 3) normalized_component = directions[1] * directions[1]; // yy
    else if (component == 4) normalized_component = directions[1] * directions[2]; // yz
    else normalized_component = directions[2] * directions[2];                     // zz
    
    const complex<float> exp_term = exp(complex<float>(0, k * r)) * inv_r;
    const complex<float> term2_factor = (complex<float>(0, k * r) - complex<float>(1, 0)) / r_squared;
    const bool is_diagonal = (component == 0 || component == 3 || component == 5);
    float multiplier = 1.0f;
    if (ix == nx || iy == ny || iz == nz) multiplier = 0.0f;
    
    const float term1 = k * k * (normalized_component - (is_diagonal ? 1.0f : 0.0f));
    const complex<float> term2 = (3.0f * normalized_component - (is_diagonal ? 1.0f : 0.0f)) * term2_factor;
    interaction_matrix[i] = multiplier * exp_term * (term1 + term2);
    ''',
    'dipole_greens_single'
)

# Double precision kernel
greens_kernel_double = cp.ElementwiseKernel(
    'int32 nx, int32 ny, int32 nz, float64 k, float64 lattice_spacing, int32 component',
    'raw complex128 interaction_matrix',
    '''
    int ix = i / (2 * ny * 2 * nz);
    int iy = (i / (2 * nz)) % (2 * ny);
    int iz = i % (2 * nz);
    double positions[3] = {
        ix * lattice_spacing,
        iy * lattice_spacing,
        iz * lattice_spacing
    };
    if (ix >= nx) positions[0] -= 2 * nx * lattice_spacing;
    if (iy >= ny) positions[1] -= 2 * ny * lattice_spacing;
    if (iz >= nz) positions[2] -= 2 * nz * lattice_spacing;
    
    if (ix == 0 && iy == 0 && iz == 0) {
        interaction_matrix[i] = complex<double>(0, 0);
        return;
    }
    
    const double r_squared = positions[0] * positions[0] +
                           positions[1] * positions[1] +
                           positions[2] * positions[2];
    const double r = sqrt(r_squared);
    const double inv_r = 1.0 / r;
    const double directions[3] = {
        positions[0] * inv_r,
        positions[1] * inv_r,
        positions[2] * inv_r
    };
    
    double normalized_component;
    if (component == 0) normalized_component = directions[0] * directions[0];      // xx
    else if (component == 1) normalized_component = directions[0] * directions[1]; // xy
    else if (component == 2) normalized_component = directions[0] * directions[2]; // xz
    else if (component == 3) normalized_component = directions[1] * directions[1]; // yy
    else if (component == 4) normalized_component = directions[1] * directions[2]; // yz
    else normalized_component = directions[2] * directions[2];                     // zz
    
    const complex<double> exp_term = exp(complex<double>(0, k * r)) * inv_r;
    const complex<double> term2_factor = (complex<double>(0, k * r) - complex<double>(1, 0)) / r_squared;
    const bool is_diagonal = (component == 0 || component == 3 || component == 5);
    double multiplier = 1.0;
    if (ix == nx || iy == ny || iz == nz) multiplier = 0.0;
    
    const double term1 = k * k * (normalized_component - (is_diagonal ? 1.0 : 0.0));
    const complex<double> term2 = (3.0 * normalized_component - (is_diagonal ? 1.0 : 0.0)) * term2_factor;
    interaction_matrix[i] = multiplier * exp_term * (term1 + term2);
    ''',
    'dipole_greens_double'
)

@free_gpu_memory    
def generate_interaction_row(nx, ny, nz, k, lattice_spacing, reduced=True, double_precision=False):
    # Set precision-dependent variables
    if double_precision:
        dtype = cp.complex128
        kernel = greens_kernel_double
        k = float(k)
        lattice_spacing = float(lattice_spacing)
    else:
        dtype = cp.complex64
        kernel = greens_kernel_single
        k = np.float32(k)
        lattice_spacing = np.float32(lattice_spacing)

    # Initialize result array
    result = cp.zeros((2*nx, 2*ny, 2*nz, 6), dtype=dtype)
    
    # Compute for each component
    for component in range(6):
        kernel(nx, ny, nz, k, lattice_spacing, component,
               result[..., component], size=8*nx*ny*nz)
    
    # FFT transform
    if reduced:
        shape = (nx+1, ny+1, nz+1, 6)
    else:
        shape = result.shape
        
    interaction_matrix_fft = cp.zeros(shape, dtype=dtype)
    for i in range(6):
        interaction_matrix_fft[..., i] = cp.fft.fftn(-result[..., i], 
                                                    axes=(0,1,2))[:shape[0], :shape[1], :shape[2]]
    
    cp.get_default_memory_pool().free_all_blocks()
    return interaction_matrix_fft




precon_kernel_single = cp.ElementwiseKernel(
    'int32 nx, int32 ny, int32 nz, float32 k, float32 lattice_spacing, int32 component',
    'raw complex64 precon',
    '''
    int ix = i / (ny * nz);
    int iy = (i / nz) % ny;
    int iz = i % nz;

    float positions[3] = {
        ix * lattice_spacing,
        iy * lattice_spacing,
        iz * lattice_spacing
    };

    if (ix == 0 && iy == 0 && iz == 0) {
        precon[i] = complex<float>(0, 0);
        return;
    }

    const float r_squared = positions[0] * positions[0] + 
                          positions[1] * positions[1] + 
                          positions[2] * positions[2];
    const float r = sqrtf(r_squared);
    const float inv_r = 1.0f / r;

    const float directions[3] = {
        positions[0] * inv_r,
        positions[1] * inv_r,
        positions[2] * inv_r
    };

    float normalized_component;
    if (component == 0) normalized_component = directions[0] * directions[0];      // xx
    else if (component == 1) normalized_component = directions[0] * directions[1]; // xy
    else if (component == 2) normalized_component = directions[0] * directions[2]; // xz
    else if (component == 3) normalized_component = directions[1] * directions[1]; // yy
    else if (component == 4) normalized_component = directions[1] * directions[2]; // yz
    else normalized_component = directions[2] * directions[2];                     // zz

    const complex<float> exp_term = exp(complex<float>(0, k * r)) * inv_r;
    const complex<float> term2_factor = (complex<float>(0, k * r) - complex<float>(1, 0)) / r_squared;
    const bool is_diagonal = (component == 0 || component == 3 || component == 5);
    
    const float term1 = k * k * (normalized_component - (is_diagonal ? 1.0f : 0.0f));
    const complex<float> term2 = (3.0f * normalized_component - (is_diagonal ? 1.0f : 0.0f)) * term2_factor;
    
    precon[i] = exp_term * (term1 + term2);
    ''',
    'precon_gen_single'
)

# Double precision kernel
precon_kernel_double = cp.ElementwiseKernel(
    'int32 nx, int32 ny, int32 nz, float64 k, float64 lattice_spacing, int32 component',
    'raw complex128 precon',
    '''
    int ix = i / (ny * nz);
    int iy = (i / nz) % ny;
    int iz = i % nz;

    double positions[3] = {
        ix * lattice_spacing,
        iy * lattice_spacing,
        iz * lattice_spacing
    };

    if (ix == 0 && iy == 0 && iz == 0) {
        precon[i] = complex<double>(0, 0);
        return;
    }

    const double r_squared = positions[0] * positions[0] + 
                           positions[1] * positions[1] + 
                           positions[2] * positions[2];
    const double r = sqrt(r_squared);
    const double inv_r = 1.0 / r;

    const double directions[3] = {
        positions[0] * inv_r,
        positions[1] * inv_r,
        positions[2] * inv_r
    };

    double normalized_component;
    if (component == 0) normalized_component = directions[0] * directions[0];      // xx
    else if (component == 1) normalized_component = directions[0] * directions[1]; // xy
    else if (component == 2) normalized_component = directions[0] * directions[2]; // xz
    else if (component == 3) normalized_component = directions[1] * directions[1]; // yy
    else if (component == 4) normalized_component = directions[1] * directions[2]; // yz
    else normalized_component = directions[2] * directions[2];                     // zz

    const complex<double> exp_term = exp(complex<double>(0, k * r)) * inv_r;
    const complex<double> term2_factor = (complex<double>(0, k * r) - complex<double>(1, 0)) / r_squared;
    const bool is_diagonal = (component == 0 || component == 3 || component == 5);
    
    const double term1 = k * k * (normalized_component - (is_diagonal ? 1.0 : 0.0));
    const complex<double> term2 = (3.0 * normalized_component - (is_diagonal ? 1.0 : 0.0)) * term2_factor;
    
    precon[i] = exp_term * (term1 + term2);
    ''',
    'precon_gen_double'
)

@free_gpu_memory            
def pre_preconditioner(nx, ny, nz, k, lattice_spacing, x_expansion, y_expansion, z_expansion, double_precision=False):
    pad_x = x_expansion
    pad_y = y_expansion
    pad_z = z_expansion
    print(f"Original shape: ({nx}, {ny}, {nz})")
    print(f"Padded shape: ({pad_x}, {pad_y}, {pad_z})")

    if double_precision:
        dtype = cp.complex128
        kernel = precon_kernel_double
        k = float(k)
        lattice_spacing = float(lattice_spacing)
    else:
        dtype = cp.complex64
        kernel = precon_kernel_single
        k = np.float32(k)
        lattice_spacing = np.float32(lattice_spacing)

    precon = cp.zeros((pad_x, pad_y, pad_z, 6), dtype=dtype)
    
    for component in range(6):
        kernel(pad_x, pad_y, pad_z, k, lattice_spacing, component, 
               precon.reshape(-1, 6)[:, component], 
               size=pad_x*pad_y*pad_z)
        cp.get_default_memory_pool().free_all_blocks()        
    return precon



invert_3x3_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void invert_3x3_kernel(complex<float>* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int matrix_idx = idx * 6;
        complex<float> a = data[matrix_idx];
        complex<float> b = data[matrix_idx+1];
        complex<float> c = data[matrix_idx+2];
        complex<float> d = data[matrix_idx+3];
        complex<float> e = data[matrix_idx+4];
        complex<float> f = data[matrix_idx+5];

        complex<float> det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
        complex<float> invdet = 1.0f / det;

        complex<float> temp0 = (d * f - e * e) * invdet;
        complex<float> temp1 = (c * e - b * f) * invdet;
        complex<float> temp2 = (b * e - c * d) * invdet;
        complex<float> temp3 = (a * f - c * c) * invdet;
        complex<float> temp4 = (b * c - a * e) * invdet;
        complex<float> temp5 = (a * d - b * b) * invdet;

        data[matrix_idx]   = temp0;
        data[matrix_idx+1] = temp1;
        data[matrix_idx+2] = temp2;
        data[matrix_idx+3] = temp3;
        data[matrix_idx+4] = temp4;
        data[matrix_idx+5] = temp5;
    }
}
''', 'invert_3x3_kernel')

def invert_3x3_matrices_gpu(data):
    n = data.size // 6
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    invert_3x3_kernel((blocks,), (threads_per_block,), (data, n))

invert_3x3_kernel_double = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void invert_3x3_kernel_double(complex<double>* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int matrix_idx = idx * 6;
        complex<double> a = data[matrix_idx];
        complex<double> b = data[matrix_idx+1];
        complex<double> c = data[matrix_idx+2];
        complex<double> d = data[matrix_idx+3];
        complex<double> e = data[matrix_idx+4];
        complex<double> f = data[matrix_idx+5];

        complex<double> det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
        complex<double> invdet = 1.0 / det;

        complex<double> temp0 = (d * f - e * e) * invdet;
        complex<double> temp1 = (c * e - b * f) * invdet;
        complex<double> temp2 = (b * e - c * d) * invdet;
        complex<double> temp3 = (a * f - c * c) * invdet;
        complex<double> temp4 = (b * c - a * e) * invdet;
        complex<double> temp5 = (a * d - b * b) * invdet;

        data[matrix_idx]   = temp0;
        data[matrix_idx+1] = temp1;
        data[matrix_idx+2] = temp2;
        data[matrix_idx+3] = temp3;
        data[matrix_idx+4] = temp4;
        data[matrix_idx+5] = temp5;
    }
}
''', 'invert_3x3_kernel_double')

def invert_3x3_matrices_gpu_double(data):
    n = data.size // 6
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    invert_3x3_kernel_double((blocks,), (threads_per_block,), (data, n))    

x_average_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>

extern "C" __global__
void x_average(complex<float>* data, const float* sym_x, 
               const int nx, const int ny, const int nz, const int nv,
               const int stride_x, const int stride_y, const int stride_z) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_points = ny * nz * nv;
    
    if (idx < total_points) {
        int component = idx % nv;
        int z = (idx / nv) % nz;
        int y = (idx / (nv * nz));
        
        int points_to_average = (nx - 1) / 2;
        
        for (int x = 1; x <= points_to_average; x++) {
            float wx = float(x) / float(nx);
            float wnx = float(nx - x) / float(nx);
            
            int forward_idx = x * stride_x + y * stride_y + z * stride_z + component;
            int reverse_idx = (nx-x) * stride_x + y * stride_y + z * stride_z + component;
            
            complex<float> forward_val = data[forward_idx];
            complex<float> reverse_val = data[reverse_idx];
            complex<float> averaged = wnx * forward_val + sym_x[component] * wx * reverse_val;
            
            data[forward_idx] = averaged;
            data[reverse_idx] = sym_x[component] < 0 ? -averaged : averaged;
        }
        
        if (nx % 2 == 0) {
            int middle_x = nx / 2;
            int middle_idx = middle_x * stride_x + y * stride_y + z * stride_z + component;
            if (sym_x[component] < 0) {
                data[middle_idx] = 0;
            }
        }
    }
}
''', 'x_average')

# Y-axis averaging kernel
y_average_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>

extern "C" __global__
void y_average(complex<float>* data, const float* sym_y, 
               const int nx, const int ny, const int nz, const int nv,
               const int stride_x, const int stride_y, const int stride_z) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_points = nx * nz * nv;
    
    if (idx < total_points) {
        int component = idx % nv;
        int z = (idx / nv) % nz;
        int x = (idx / (nv * nz));
        
        int points_to_average = (ny - 1) / 2;
        
        for (int y = 1; y <= points_to_average; y++) {
            float wy = float(y) / float(ny);
            float wny = float(ny - y) / float(ny);
            
            int forward_idx = x * stride_x + y * stride_y + z * stride_z + component;
            int reverse_idx = x * stride_x + (ny-y) * stride_y + z * stride_z + component;
            
            complex<float> forward_val = data[forward_idx];
            complex<float> reverse_val = data[reverse_idx];
            complex<float> averaged = wny * forward_val + sym_y[component] * wy * reverse_val;
            
            data[forward_idx] = averaged;
            data[reverse_idx] = sym_y[component] < 0 ? -averaged : averaged;
        }
        
        if (ny % 2 == 0) {
            int middle_y = ny / 2;
            int middle_idx = x * stride_x + middle_y * stride_y + z * stride_z + component;
            if (sym_y[component] < 0) {
                data[middle_idx] = 0;
            }
        }
    }
}
''', 'y_average')

# Z-axis averaging kernel
z_average_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>

extern "C" __global__
void z_average(complex<float>* data, const float* sym_z, 
               const int nx, const int ny, const int nz, const int nv,
               const int stride_x, const int stride_y, const int stride_z) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_points = nx * ny * nv;
    
    if (idx < total_points) {
        int component = idx % nv;
        int y = (idx / nv) % ny;
        int x = (idx / (nv * ny));
        
        int points_to_average = (nz - 1) / 2;
        
        for (int z = 1; z <= points_to_average; z++) {
            float wz = float(z) / float(nz);
            float wnz = float(nz - z) / float(nz);
            
            int forward_idx = x * stride_x + y * stride_y + z * stride_z + component;
            int reverse_idx = x * stride_x + y * stride_y + (nz-z) * stride_z + component;
            
            complex<float> forward_val = data[forward_idx];
            complex<float> reverse_val = data[reverse_idx];
            complex<float> averaged = wnz * forward_val + sym_z[component] * wz * reverse_val;
            
            data[forward_idx] = averaged;
            data[reverse_idx] = sym_z[component] < 0 ? -averaged : averaged;
        }
        
        if (nz % 2 == 0) {
            int middle_z = nz / 2;
            int middle_idx = x * stride_x + y * stride_y + middle_z * stride_z + component;
            if (sym_z[component] < 0) {
                data[middle_idx] = 0;
            }
        }
    }
}
''', 'z_average')

x_average_kernel_double = cp.RawKernel(r'''
#include <cupy/complex.cuh>

extern "C" __global__
void x_average_double(complex<double>* data, const double* sym_x, 
               const int nx, const int ny, const int nz, const int nv,
               const int stride_x, const int stride_y, const int stride_z) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_points = ny * nz * nv;
    
    if (idx < total_points) {
        int component = idx % nv;
        int z = (idx / nv) % nz;
        int y = (idx / (nv * nz));
        
        int points_to_average = (nx - 1) / 2;
        
        for (int x = 1; x <= points_to_average; x++) {
            double wx = double(x) / double(nx);
            double wnx = double(nx - x) / double(nx);
            
            int forward_idx = x * stride_x + y * stride_y + z * stride_z + component;
            int reverse_idx = (nx-x) * stride_x + y * stride_y + z * stride_z + component;
            
            complex<double> forward_val = data[forward_idx];
            complex<double> reverse_val = data[reverse_idx];
            complex<double> averaged = wnx * forward_val + sym_x[component] * wx * reverse_val;
            
            data[forward_idx] = averaged;
            data[reverse_idx] = sym_x[component] < 0 ? -averaged : averaged;
        }
        
        if (nx % 2 == 0) {
            int middle_x = nx / 2;
            int middle_idx = middle_x * stride_x + y * stride_y + z * stride_z + component;
            if (sym_x[component] < 0) {
                data[middle_idx] = 0;
            }
        }
    }
}
''', 'x_average_double')

y_average_kernel_double = cp.RawKernel(r'''
#include <cupy/complex.cuh>

extern "C" __global__
void y_average_double(complex<double>* data, const double* sym_y, 
               const int nx, const int ny, const int nz, const int nv,
               const int stride_x, const int stride_y, const int stride_z) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_points = nx * nz * nv;
    
    if (idx < total_points) {
        int component = idx % nv;
        int z = (idx / nv) % nz;
        int x = (idx / (nv * nz));
        
        int points_to_average = (ny - 1) / 2;
        
        for (int y = 1; y <= points_to_average; y++) {
            double wy = double(y) / double(ny);
            double wny = double(ny - y) / double(ny);
            
            int forward_idx = x * stride_x + y * stride_y + z * stride_z + component;
            int reverse_idx = x * stride_x + (ny-y) * stride_y + z * stride_z + component;
            
            complex<double> forward_val = data[forward_idx];
            complex<double> reverse_val = data[reverse_idx];
            complex<double> averaged = wny * forward_val + sym_y[component] * wy * reverse_val;
            
            data[forward_idx] = averaged;
            data[reverse_idx] = sym_y[component] < 0 ? -averaged : averaged;
        }
        
        if (ny % 2 == 0) {
            int middle_y = ny / 2;
            int middle_idx = x * stride_x + middle_y * stride_y + z * stride_z + component;
            if (sym_y[component] < 0) {
                data[middle_idx] = 0;
            }
        }
    }
}
''', 'y_average_double')

z_average_kernel_double = cp.RawKernel(r'''
#include <cupy/complex.cuh>

extern "C" __global__
void z_average_double(complex<double>* data, const double* sym_z, 
               const int nx, const int ny, const int nz, const int nv,
               const int stride_x, const int stride_y, const int stride_z) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_points = nx * ny * nv;
    
    if (idx < total_points) {
        int component = idx % nv;
        int y = (idx / nv) % ny;
        int x = (idx / (nv * ny));
        
        int points_to_average = (nz - 1) / 2;
        
        for (int z = 1; z <= points_to_average; z++) {
            double wz = double(z) / double(nz);
            double wnz = double(nz - z) / double(nz);
            
            int forward_idx = x * stride_x + y * stride_y + z * stride_z + component;
            int reverse_idx = x * stride_x + y * stride_y + (nz-z) * stride_z + component;
            
            complex<double> forward_val = data[forward_idx];
            complex<double> reverse_val = data[reverse_idx];
            complex<double> averaged = wnz * forward_val + sym_z[component] * wz * reverse_val;
            
            data[forward_idx] = averaged;
            data[reverse_idx] = sym_z[component] < 0 ? -averaged : averaged;
        }
        
        if (nz % 2 == 0) {
            int middle_z = nz / 2;
            int middle_idx = x * stride_x + y * stride_y + middle_z * stride_z + component;
            if (sym_z[component] < 0) {
                data[middle_idx] = 0;
            }
        }
    }
}
''', 'z_average_double')


x_average_2d_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>

extern "C" __global__
void x_average_2d(complex<float>* data, const float* sym_x, 
               const int nx, const int ny, const int nz3, const int nz3_2,
               const int stride_x, const int stride_y, const int stride_z1, const int stride_z2) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_points = ny * nz3 * nz3_2;
    
    if (idx < total_points) {
        int z2 = idx % nz3_2;
        int z1 = (idx / nz3_2) % nz3;
        int y = idx / (nz3 * nz3_2);
        
        int points_to_average = (nx - 1) / 2;
        
        for (int x = 1; x <= points_to_average; x++) {
            float wx = float(x) / float(nx);
            float wnx = float(nx - x) / float(nx);
            
            int forward_idx = x * stride_x + y * stride_y + z1 * stride_z1 + z2 * stride_z2;
            int reverse_idx = (nx-x) * stride_x + y * stride_y + z1 * stride_z1 + z2 * stride_z2;
            
            complex<float> forward_val = data[forward_idx];
            complex<float> reverse_val = data[reverse_idx];
            complex<float> averaged = wnx * forward_val + sym_x[z1/3] * wx * reverse_val;
            
            data[forward_idx] = averaged;
            data[reverse_idx] = sym_x[z1/3] < 0 ? -averaged : averaged;
        }
        
        if (nx % 2 == 0) {
            int middle_x = nx / 2;
            int middle_idx = middle_x * stride_x + y * stride_y + z1 * stride_z1 + z2 * stride_z2;
            if (sym_x[z1/3] < 0) {
                data[middle_idx] = 0;
            }
        }
    }
}
''', 'x_average_2d')

y_average_2d_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>

extern "C" __global__
void y_average_2d(complex<float>* data, const float* sym_y, 
               const int nx, const int ny, const int nz3, const int nz3_2,
               const int stride_x, const int stride_y, const int stride_z1, const int stride_z2) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_points = nx * nz3 * nz3_2;
    
    if (idx < total_points) {
        int z2 = idx % nz3_2;
        int z1 = (idx / nz3_2) % nz3;
        int x = idx / (nz3 * nz3_2);
        
        int points_to_average = (ny - 1) / 2;
        
        for (int y = 1; y <= points_to_average; y++) {
            float wy = float(y) / float(ny);
            float wny = float(ny - y) / float(ny);
            
            int forward_idx = x * stride_x + y * stride_y + z1 * stride_z1 + z2 * stride_z2;
            int reverse_idx = x * stride_x + (ny-y) * stride_y + z1 * stride_z1 + z2 * stride_z2;
            
            complex<float> forward_val = data[forward_idx];
            complex<float> reverse_val = data[reverse_idx];
            complex<float> averaged = wny * forward_val + sym_y[z1/3] * wy * reverse_val;
            
            data[forward_idx] = averaged;
            data[reverse_idx] = sym_y[z1/3] < 0 ? -averaged : averaged;
        }
        
        if (ny % 2 == 0) {
            int middle_y = ny / 2;
            int middle_idx = x * stride_x + middle_y * stride_y + z1 * stride_z1 + z2 * stride_z2;
            if (sym_y[z1/3] < 0) {
                data[middle_idx] = 0;
            }
        }
    }
}
''', 'y_average_2d')

x_average_2d_kernel_double = cp.RawKernel(r'''
#include <cupy/complex.cuh>

extern "C" __global__
void x_average_2d_double(complex<double>* data, const double* sym_x, 
               const int nx, const int ny, const int nz3, const int nz3_2,
               const int stride_x, const int stride_y, const int stride_z1, const int stride_z2) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_points = ny * nz3 * nz3_2;
    
    if (idx < total_points) {
        int z2 = idx % nz3_2;
        int z1 = (idx / nz3_2) % nz3;
        int y = idx / (nz3 * nz3_2);
        
        int points_to_average = (nx - 1) / 2;
        
        for (int x = 1; x <= points_to_average; x++) {
            double wx = double(x) / double(nx);
            double wnx = double(nx - x) / double(nx);
            
            int forward_idx = x * stride_x + y * stride_y + z1 * stride_z1 + z2 * stride_z2;
            int reverse_idx = (nx-x) * stride_x + y * stride_y + z1 * stride_z1 + z2 * stride_z2;
            
            complex<double> forward_val = data[forward_idx];
            complex<double> reverse_val = data[reverse_idx];
            complex<double> averaged = wnx * forward_val + sym_x[z1/3] * wx * reverse_val;
            
            data[forward_idx] = averaged;
            data[reverse_idx] = sym_x[z1/3] < 0 ? -averaged : averaged;
        }
        
        if (nx % 2 == 0) {
            int middle_x = nx / 2;
            int middle_idx = middle_x * stride_x + y * stride_y + z1 * stride_z1 + z2 * stride_z2;
            if (sym_x[z1/3] < 0) {
                data[middle_idx] = 0;
            }
        }
    }
}
''', 'x_average_2d_double')

y_average_2d_kernel_double = cp.RawKernel(r'''
#include <cupy/complex.cuh>

extern "C" __global__
void y_average_2d_double(complex<double>* data, const double* sym_y, 
               const int nx, const int ny, const int nz3, const int nz3_2,
               const int stride_x, const int stride_y, const int stride_z1, const int stride_z2) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_points = nx * nz3 * nz3_2;
    
    if (idx < total_points) {
        int z2 = idx % nz3_2;
        int z1 = (idx / nz3_2) % nz3;
        int x = idx / (nz3 * nz3_2);
        
        int points_to_average = (ny - 1) / 2;
        
        for (int y = 1; y <= points_to_average; y++) {
            double wy = double(y) / double(ny);
            double wny = double(ny - y) / double(ny);
            
            int forward_idx = x * stride_x + y * stride_y + z1 * stride_z1 + z2 * stride_z2;
            int reverse_idx = x * stride_x + (ny-y) * stride_y + z1 * stride_z1 + z2 * stride_z2;
            
            complex<double> forward_val = data[forward_idx];
            complex<double> reverse_val = data[reverse_idx];
            complex<double> averaged = wny * forward_val + sym_y[z1/3] * wy * reverse_val;
            
            data[forward_idx] = averaged;
            data[reverse_idx] = sym_y[z1/3] < 0 ? -averaged : averaged;
        }
        
        if (ny % 2 == 0) {
            int middle_y = ny / 2;
            int middle_idx = x * stride_x + middle_y * stride_y + z1 * stride_z1 + z2 * stride_z2;
            if (sym_y[z1/3] < 0) {
                data[middle_idx] = 0;
            }
        }
    }
}
''', 'y_average_2d_double')


@free_gpu_memory    
def circulant_approximation(original_data, alpha_array, refract_mult, reduced=True, double_precision=False):
    cp.get_default_memory_pool().free_all_blocks()
    
    # Set precision-dependent variables
    if double_precision:
        dtype = cp.complex128
        x_kernel = x_average_kernel_double
        y_kernel = y_average_kernel_double
        z_kernel = z_average_kernel_double
        invert_kernel = invert_3x3_matrices_gpu_double
    else:
        dtype = cp.complex64
        x_kernel = x_average_kernel
        y_kernel = y_average_kernel
        z_kernel = z_average_kernel
        invert_kernel = invert_3x3_matrices_gpu
    
    nx, ny, nz, nv = original_data.shape
    
    # Initial setup
    raw_mean = cp.mean(alpha_array[alpha_array != 0])
    #alpha_avg = refract_mult * raw_mean.astype(dtype)
    alpha_avg = 1/refract_mult
    #alpha_avg =  raw_mean.astype(dtype)
    print(f"Mean before multiplication: {raw_mean}")
    print(f"Mean after multiplication with {refract_mult}: {alpha_avg}")
    
    # Define symmetry arrays with appropriate precision
    if double_precision:
        sym_x = cp.array([1, -1, -1, 1, 1, 1], dtype=cp.float64)
        sym_y = cp.array([1, -1, 1, 1, -1, 1], dtype=cp.float64)
        sym_z = cp.array([1, 1, -1, 1, -1, 1], dtype=cp.float64)
    else:
        sym_x = cp.array([1, -1, -1, 1, 1, 1], dtype=cp.float32)
        sym_y = cp.array([1, -1, 1, 1, -1, 1], dtype=cp.float32)
        sym_z = cp.array([1, 1, -1, 1, -1, 1], dtype=cp.float32)
    
    indices = cp.array([0, 3, 5])
    original_data[0, 0, 0, indices] += alpha_avg
    del indices, alpha_avg, raw_mean  # Clean up unused variables
    cp.get_default_memory_pool().free_all_blocks()
    
    # Get strides for kernel indexing
    strides = original_data.strides
    stride_x = strides[0] // (16 if double_precision else 8)
    stride_y = strides[1] // (16 if double_precision else 8)
    stride_z = strides[2] // (16 if double_precision else 8)
    
    # Set up kernel parameters
    threads_per_block = 256
    
    # X averaging
    blocks_per_grid = (ny * nz * nv + threads_per_block - 1) // threads_per_block
    x_kernel((blocks_per_grid,), (threads_per_block,),
            (original_data, sym_x, nx, ny, nz, nv,
             stride_x, stride_y, stride_z))
    cp.get_default_memory_pool().free_all_blocks()
    
    # Y averaging
    blocks_per_grid = (nx * nz * nv + threads_per_block - 1) // threads_per_block
    y_kernel((blocks_per_grid,), (threads_per_block,),
            (original_data, sym_y, nx, ny, nz, nv,
             stride_x, stride_y, stride_z))
    cp.get_default_memory_pool().free_all_blocks()
    
    # Z averaging
    blocks_per_grid = (nx * ny * nv + threads_per_block - 1) // threads_per_block
    z_kernel((blocks_per_grid,), (threads_per_block,),
            (original_data, sym_z, nx, ny, nz, nv,
             stride_x, stride_y, stride_z))
    cp.get_default_memory_pool().free_all_blocks()

    # Forward FFT
    original_data = torch.from_dlpack(original_data)
    for i in range(nv):
        original_data[:, :, :, i] = torch.fft.fftn(original_data[:, :, :, i], dim=(0, 1, 2))
    original_data = cp.from_dlpack(original_data)
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    
    # Matrix inversion
    invert_kernel(original_data)
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    #original_data[0, 0, 0, 0] *= refract_mult  # First element
    #original_data[0, 0, 0, 3] *= refract_mult  # Fourth element
    #original_data[0, 0, 0, 5] *= refract_mult  # Sixth element
    # Inverse FFT
    original_data = torch.from_dlpack(original_data)
    for i in range(nv):
        original_data[:, :, :, i] = torch.fft.ifftn(original_data[:, :, :, i], dim=(0, 1, 2))
    original_data = cp.from_dlpack(original_data)
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    
    # Dimension reduction
    current_nx, current_ny, current_nz = nx, ny, nz
    alpha_nx, alpha_ny, alpha_nz = alpha_array.shape
    
    while True:
        reduction_performed = False
        
        # X dimension reduction
        if current_nx >= 2 * alpha_nx:
            n2 = current_nx//2
            original_data = original_data[:n2, :, :]
            strides = original_data.strides
            stride_x = strides[0] // (16 if double_precision else 8)
            stride_y = strides[1] // (16 if double_precision else 8)
            stride_z = strides[2] // (16 if double_precision else 8)
            
            blocks_per_grid = (original_data.shape[1] * original_data.shape[2] * nv + threads_per_block - 1) // threads_per_block
            x_kernel((blocks_per_grid,), (threads_per_block,),
                    (original_data, sym_x, n2, original_data.shape[1], original_data.shape[2], nv,
                     stride_x, stride_y, stride_z))
            
            current_nx = original_data.shape[0]
            reduction_performed = True
        
        # Y dimension reduction
        if current_ny >= 2 * alpha_ny:
            n2 = current_ny//2
            original_data = original_data[:, :n2, :]
            strides = original_data.strides
            stride_x = strides[0] // (16 if double_precision else 8)
            stride_y = strides[1] // (16 if double_precision else 8)
            stride_z = strides[2] // (16 if double_precision else 8)
            
            blocks_per_grid = (original_data.shape[0] * original_data.shape[2] * nv + threads_per_block - 1) // threads_per_block
            y_kernel((blocks_per_grid,), (threads_per_block,),
                    (original_data, sym_y, original_data.shape[0], n2, original_data.shape[2], nv,
                     stride_x, stride_y, stride_z))
            
            current_ny = original_data.shape[1]
            reduction_performed = True
        
        # Z dimension reduction
        # Z dimension reduction
        if current_nz >= 2 * alpha_nz:
            n2 = current_nz//2
            original_data = original_data[:, :, :n2]
            strides = original_data.strides
            stride_x = strides[0] // (16 if double_precision else 8)
            stride_y = strides[1] // (16 if double_precision else 8)
            stride_z = strides[2] // (16 if double_precision else 8)
            
            blocks_per_grid = (original_data.shape[0] * original_data.shape[1] * nv + threads_per_block - 1) // threads_per_block
            z_kernel((blocks_per_grid,), (threads_per_block,),
                    (original_data, sym_z, original_data.shape[0], original_data.shape[1], n2, nv,
                     stride_x, stride_y, stride_z))
            
            current_nz = original_data.shape[2]
            reduction_performed = True
        
        # Free memory after each round of reductions
        cp.get_default_memory_pool().free_all_blocks()
        
        # If no reduction was performed in any dimension, we're done
        if not reduction_performed:
            break

    # Final FFT
    original_data = torch.from_dlpack(original_data)
    for i in range(nv):
        original_data[:, :, :, i] = torch.fft.fftn(original_data[:, :, :, i], dim=(0, 1, 2))
    original_data = cp.from_dlpack(original_data)
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    
    # Clean up variables that won't be used anymore
    del sym_x, sym_y, sym_z, strides, stride_x, stride_y, stride_z
    del threads_per_block, blocks_per_grid
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    if not reduced:
        print(f"Full size: {original_data.shape}")
        return original_data.astype(dtype)
    else:
        reduced_nx = current_nx//2 + 1
        reduced_ny = current_ny//2 + 1
        reduced_nz = current_nz//2 + 1
        
        # Check each dimension
        if reduced_nx == alpha_nx and (current_nx//2) % 2 != 0:
            reduced_nx += 1
        if reduced_ny == alpha_ny and (current_ny//2) % 2 != 0:
            reduced_ny += 1
        if reduced_nz == alpha_nz and (current_nz//2) % 2 != 0:
            reduced_nz += 1
            
        reduced_size = (reduced_nx, reduced_ny, reduced_nz)
        print(f"Full size: {original_data.shape}")
        print(f"Reduced size will be: {reduced_size}")
        result = original_data[:(reduced_nx), :(reduced_ny), :(reduced_nz)].astype(dtype)
        del original_data
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        return result

        
@free_gpu_memory          
def circulant_approximationbig(original_data, alpha_array, refract_mult, reduced=True, double_precision=False):
    print('shape')
    print(alpha_array.shape)
    
    # Set precision-dependent variables
    if double_precision:
        dtype = cp.complex128
        x_kernel = x_average_kernel_double
        y_kernel = y_average_kernel_double
        sym_dtype = cp.float64
        stride_div = 16
    else:
        dtype = cp.complex64
        x_kernel = x_average_kernel
        y_kernel = y_average_kernel
        sym_dtype = cp.float32
        stride_div = 8

    data = cp.copy(original_data).astype(dtype)
    nx, ny, nz, nv = data.shape
    alpha_nx, alpha_ny, alpha_nz = alpha_array.shape
    
    alpha_avg = refract_mult * cp.max(alpha_array)
    print(alpha_avg)
    
    # Define symmetries with appropriate precision
    sym_x = cp.array([1, -1, -1, 1, 1, 1], dtype=sym_dtype)
    sym_y = cp.array([1, -1, 1, 1, -1, 1], dtype=sym_dtype)
    sym_z = cp.array([1, 1, -1, 1, -1, 1], dtype=sym_dtype)
    
    indices = cp.array([0, 3, 5])    
    data[0, 0, 0, indices] += alpha_avg
    del indices, alpha_avg
    cp.get_default_memory_pool().free_all_blocks()

    # Get strides for kernel indexing
    strides = data.strides
    stride_x = strides[0] // stride_div
    stride_y = strides[1] // stride_div
    stride_z = strides[2] // stride_div
    
    # Set up kernel parameters
    threads_per_block = 256
    
    # X averaging
    blocks_per_grid = (ny * nz * nv + threads_per_block - 1) // threads_per_block
    x_kernel((blocks_per_grid,), (threads_per_block,),
             (data, sym_x, nx, ny, nz, nv,
              stride_x, stride_y, stride_z))
    cp.get_default_memory_pool().free_all_blocks()
    
    # Y averaging
    blocks_per_grid = (nx * nz * nv + threads_per_block - 1) // threads_per_block
    y_kernel((blocks_per_grid,), (threads_per_block,),
             (data, sym_y, nx, ny, nz, nv,
              stride_x, stride_y, stride_z))
    cp.get_default_memory_pool().free_all_blocks()

    # 2D FFT using PyTorch
    data = torch.from_dlpack(data)
    for i in range(nv):
        data[:, :, :, i] = torch.fft.fft2(data[:, :, :, i], dim=(0, 1))
    data = cp.from_dlpack(data)
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

    data = data.astype(dtype)
    cp.get_default_memory_pool().free_all_blocks()

    # Matrix building section
    indices = [(0, 1, 2), (1, 3, 4), (2, 4, 5)]
    big_toeplitz_matrices = cp.zeros((nx, ny, 3*nz, 3*nz), dtype=dtype)
    idx_i = cp.repeat(cp.arange(nz), nz).reshape(nz, nz)
    idx_j = idx_i.T
    diagonal_indices = cp.arange(3*nz)
    flat_indices = [idx for row in indices for idx in row]
    
    for block_pos, block_idx in enumerate(flat_indices):
        i, j = block_pos // 3, block_pos % 3
        start_i, start_j = i*nz, j*nz
        
        if block_idx in [0, 1, 3, 5]:
            big_toeplitz_matrices[:, :, start_i:start_i+nz, start_j:start_j+nz] = data[:, :, cp.abs(idx_i-idx_j), block_idx]
        else:
            big_toeplitz_matrices[:, :, start_i:start_i+nz, start_j:start_j+nz] = cp.where(
                idx_i >= idx_j,
                data[:, :, idx_i-idx_j, block_idx],
                -data[:, :, idx_j-idx_i, block_idx]
            )
        cp.get_default_memory_pool().free_all_blocks()
    
    del data, idx_i, idx_j, diagonal_indices, flat_indices, indices
    
    big_toeplitz_matrices *= -1
    cp.get_default_memory_pool().free_all_blocks()

    big_toeplitz_matrices = cp.linalg.inv(big_toeplitz_matrices.reshape(-1, 3*nz, 3*nz)).reshape(nx, ny, 3*nz, 3*nz)
    cp.get_default_memory_pool().free_all_blocks()

    # Dimension reduction section
    current_nx, current_ny = nx, ny
    
    if current_nx >= 2 * alpha_nx or current_ny >= 2 * alpha_ny:
        big_toeplitz_matrices = torch.from_dlpack(big_toeplitz_matrices)
        big_toeplitz_matrices = torch.fft.ifft2(big_toeplitz_matrices, dim=(0,1))
        big_toeplitz_matrices = cp.from_dlpack(big_toeplitz_matrices)
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        
        while current_nx >= 2 * alpha_nx:
            nx2 = current_nx//2
            wx = cp.arange(1, nx2, dtype=dtype)[:, None, None, None] / nx2
            
            if current_nx % 2:
                big_toeplitz_matrices[1:nx2] = (1 - wx) * big_toeplitz_matrices[1:nx2] + wx * big_toeplitz_matrices[nx2+2:2*nx2+1]
            else:
                big_toeplitz_matrices[1:nx2] = (1 - wx) * big_toeplitz_matrices[1:nx2] + wx * big_toeplitz_matrices[nx2+1:2*nx2]
            
            big_toeplitz_matrices = big_toeplitz_matrices[:nx2]
            current_nx = nx2
            cp.get_default_memory_pool().free_all_blocks()
        
        while current_ny >= 2 * alpha_ny:
            ny2 = current_ny//2
            wy = cp.arange(1, ny2, dtype=dtype)[None, :, None, None] / ny2
            
            if current_ny % 2:
                big_toeplitz_matrices[:, 1:ny2] = (1 - wy) * big_toeplitz_matrices[:, 1:ny2] + wy * big_toeplitz_matrices[:, ny2+2:2*ny2+1]
            else:
                big_toeplitz_matrices[:, 1:ny2] = (1 - wy) * big_toeplitz_matrices[:, 1:ny2] + wy * big_toeplitz_matrices[:, ny2+1:2*ny2]
            
            big_toeplitz_matrices = big_toeplitz_matrices[:, :ny2]
            current_ny = ny2
            cp.get_default_memory_pool().free_all_blocks()
        
        # FFT2 back using PyTorch
        big_toeplitz_matrices = torch.from_dlpack(big_toeplitz_matrices)
        big_toeplitz_matrices = torch.fft.fft2(big_toeplitz_matrices, dim=(0,1))
        big_toeplitz_matrices = cp.from_dlpack(big_toeplitz_matrices)
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
    
    # Final cleanup
    del sym_x, sym_y, sym_z, strides, stride_x, stride_y, stride_z
    del threads_per_block, blocks_per_grid
    if 'wx' in locals(): del wx
    if 'wy' in locals(): del wy
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    gc.collect()
    return big_toeplitz_matrices

@free_gpu_memory    
def prepare_interaction_matrices(grid_size, k, refract_mult, x_expansion, y_expansion, z_expansion, 
                               alpha_array, lattice_spacing, reduced=True, is_2d=False, 
                               cutoff=False, double_precision=False):
    nx, ny, nz = grid_size
    
    # Set precision type
    dtype = cp.complex128 if double_precision else cp.complex64
    
    # Convert alpha_array to cupy with appropriate precision
    if isinstance(alpha_array, np.ndarray):
        alpha_array = cp.asarray(alpha_array, dtype=dtype)
    else:
        alpha_array = alpha_array.astype(dtype)
    
    if cutoff:
        alpha_array = cp.where(cp.abs(alpha_array) < 1.0, 0.0, alpha_array)
    
    print('different alphas')
    print(cp.unique(alpha_array).get())
    inv_alpha = cp.where(alpha_array != 0, 1.0 / alpha_array, 0)
    print('different inverse alphas')        
    print(cp.unique(inv_alpha).get())
    mask = alpha_array != 0
    
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()        
    
    # Start timing
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    
    preconditioner = pre_preconditioner(nx, ny, nz, k, lattice_spacing, 
                                      x_expansion, y_expansion, z_expansion,
                                      double_precision=double_precision)
    if is_2d:
        preconditioner = circulant_approximationbig(
            preconditioner,
            inv_alpha,
            refract_mult,
            reduced=True,
            double_precision=double_precision
        )
    else:
        preconditioner = circulant_approximation(
            preconditioner,
            inv_alpha,
            refract_mult,
            reduced=True,
            double_precision=double_precision
        )   
    
    # End timing
    end_event.record()
    end_event.synchronize()
    time_taken = cp.cuda.get_elapsed_time(start_event, end_event)
    print(f"Time to build preconditioner: {time_taken:.2f} ms")
    
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    
    interaction_matrix = generate_interaction_row(
        grid_size[0], grid_size[1], grid_size[2], 
        k, lattice_spacing,
        reduced=reduced,
        double_precision=double_precision
    )

    # Cleanup
    del alpha_array, start_event, end_event, time_taken
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return inv_alpha, mask, interaction_matrix, preconditioner


import numpy as np
import torch
import cupy as cp
from Gpbicgstab import solve_gpbicgstab
from Gpbicgstabhybrid import solve_gpbicgstabhybrid
from Gpbicgstabdouble import solve_gpbicgstab_double
from Gpbicgstabmulti import solve_gpbicgstabmulti
    
################################################################################
# DDA Solver Wrapper
################################################################################

@free_gpu_memory
def run_solver(grid_size, k, alpha_array, lattice_spacing, x_expansion, y_expansion, 
               z_expansion, refract_mult, E_inc, precon=True, max_iter=20000, double=False, ratio = 0):
    try:
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()
        # Prepare interaction matrices (assumed defined elsewhere)
        inv_alpha, mask, interaction_matrix, preconditioner = prepare_interaction_matrices(
            grid_size, k, refract_mult,
            x_expansion, y_expansion, z_expansion,
            alpha_array, lattice_spacing,
            reduced=True, is_2d=False, cutoff=False, double_precision=double
        )
        
        # Set the data type based on double precision flag
        if double:
            interaction_matrix = interaction_matrix.astype(cp.complex128)
        else:
            interaction_matrix = interaction_matrix.astype(cp.complex64)
        ratio = 100  # Example ratio for solver
        if double:
            result = solve_gpbicgstab_double(
                grid_size, inv_alpha, interaction_matrix, preconditioner,
                E_inc, mask, ratio, max_iter, is_2d=False, precon=precon
            )
        else:
            result = solve_gpbicgstab(
                grid_size, inv_alpha, interaction_matrix, preconditioner,
                E_inc, mask, ratio, max_iter, is_2d=False, precon=precon
            )
        # Return: iterations, polarization_array, final_norm
        # result[0] = iterations
        # result[1] = polarization_array
        # result[2] = final_norm (scalar)
        return result[0], result[1], result[2]  # Fixed: correct order 0,1,2
    except Exception as e:
        raise RuntimeError(f"Solver failed: {str(e)}")


import numpy as np
import scipy.stats as stats
import math

def analyze_and_normalize11(problem):
    """
    Analyzes DDA problem parameters and normalizes them into a 76-feature vector.

    Key analysis aspects include:
    - Alpha array:
        - Magnitudes: Normalized by the maximum magnitude in the active set (values range [0,1]).
        - Phases: Original angle [0,π] (from arccos(Re/|alpha|)) mapped to [-1,1].
        - Statistical descriptions (mean, median, min, std, skew, kurtosis, IQR, CV, IQR-to-Range ratio) for both.
        - Cross-magnitude/phase features (e.g., phase at max magnitude).
        - Phase zone distributions.
        - Ratio of Re(alpha) > 0.
    - N array:
        - Magnitudes: Linearly normalized by a fixed divisor.
        - Phases: Original angle (from np.angle) clipped to [0,π/2] and then mapped to [0,1].
        - Statistical descriptions (similar to Alpha) for both.
        - Cross-magnitude/phase features.
        - Phase zone distributions.
    - Cross Alpha-N features:
        - Correlation of absolute magnitudes.
        - Correlation of normalized phases.
    - Geometric features: Ratios involving lattice spacing, wavelength, and grid dimensions.
    - FFT of alpha_array (if full grid):
        - DC component ratio.
        - AC component statistics (mean, std normalized by max FFT magnitude).
        - Ratios of AC energy along primary axes and in a local box around DC.
        - New: Spectral flatness, ratio of 2nd largest to largest AC peak, radial AC energy distribution.

    Skewness and Kurtosis are tanh-normalized. Mode calculations have been removed.
    """
    # --- Configuration ---
    N_MAG_NORM_DIVISOR = 6.0
    AXIS_ZONE_TOLERANCE_ANGLE = math.pi / 32.0
    SAFE_DIVISION_EPS = 1e-15 # Epsilon for safe division

    final_vector = np.zeros(76) # Updated feature vector size

    # --- Helper Functions ---
    def calculate_simple_shape_stats_reworked(data_array): # Tanh normalized skew/kurt
        if data_array.size < 2: return 0.0, 0.0
        is_constant = False
        # Check for constant array to avoid issues with skew/kurt
        if data_array.size > 1:
            # Check if all elements are close to the first element
            if np.all(np.abs(data_array - data_array[0]) < SAFE_DIVISION_EPS): is_constant = True
            # Check if peak-to-peak is very small relative to mean or absolute scale
            elif np.ptp(data_array) < SAFE_DIVISION_EPS * max(np.abs(np.mean(data_array)), SAFE_DIVISION_EPS): is_constant = True
            elif np.ptp(data_array) < SAFE_DIVISION_EPS: is_constant = True
        
        skewness, kurt = 0.0, 0.0
        if not is_constant:
            # Skewness requires at least 3 data points for scipy's default (bias=False).
            # For smaller arrays, or if issues, default to 0.
            if data_array.size > 2: # SciPy default skew needs > 2 for non-zero std
                try: skewness = np.tanh(stats.skew(data_array, nan_policy='omit'))
                except: skewness = 0.0
            # Kurtosis (Fisher) requires at least 4 data points for some definitions, 
            # SciPy's kurtosis needs > 3 for non-zero std for fisher.
            if data_array.size >= 4 : # SciPy default kurtosis needs > 3 for non-zero std for fisher
                 try: kurt = np.tanh(stats.kurtosis(data_array, fisher=True, nan_policy='omit'))
                 except: kurt = 0.0
        return float(skewness), float(kurt)

    def calculate_correlation(arr1, arr2):
        if arr1.size < 2 or arr2.size < 2 or arr1.size != arr2.size: return 0.0
        # Check for constant arrays
        if np.std(arr1) < SAFE_DIVISION_EPS or np.std(arr2) < SAFE_DIVISION_EPS: return 0.0
        try:
            valid_mask = ~np.isnan(arr1) & ~np.isinf(arr1) & ~np.isnan(arr2) & ~np.isinf(arr2)
            if np.sum(valid_mask) < 2: return 0.0 # Need at least 2 valid pairs
            corr = np.corrcoef(arr1[valid_mask], arr2[valid_mask])[0, 1]
            return float(corr) if np.isfinite(corr) else 0.0
        except: return 0.0

    def calculate_iqr(data_array):
        if data_array.size < 2: return 0.0
        data_array_clean = data_array[~np.isnan(data_array) & ~np.isinf(data_array)]
        if data_array_clean.size < 2: return 0.0
        try:
            # Ensure array isn't effectively constant for percentile to make sense for IQR
            # A minimum of 4 points is often implicitly assumed for robust Q1/Q3, but percentile works for fewer.
            # If too few points and constant, IQR is 0.
            if data_array_clean.size < 4 and np.all(np.abs(data_array_clean - data_array_clean[0]) < SAFE_DIVISION_EPS): return 0.0
            q1 = np.percentile(data_array_clean, 25)
            q3 = np.percentile(data_array_clean, 75)
            return float(q3 - q1)
        except Exception: return 0.0
        
    def calculate_cv(data_array): # Coefficient of Variation (std/mean)
        if data_array.size < 2: return 0.0
        data_array_clean = data_array[~np.isnan(data_array) & ~np.isinf(data_array)]
        if data_array_clean.size < 2: return 0.0
        mean_val = np.mean(data_array_clean)
        std_val = np.std(data_array_clean)
        if abs(mean_val) < SAFE_DIVISION_EPS:
            # If mean is near zero, CV is undefined or very large.
            # If std is also near zero, it's a constant near zero, CV can be 0.
            return 0.0 if std_val < SAFE_DIVISION_EPS else 0.0 # Or a large placeholder if preferred
        return float(std_val / mean_val)

    def calculate_iqr_to_range_ratio(data_array, iqr_val=None):
        if data_array.size < 2: return 0.0
        data_array_clean = data_array[~np.isnan(data_array) & ~np.isinf(data_array)]
        if data_array_clean.size < 2: return 0.0
        
        current_range = np.ptp(data_array_clean)
        if current_range < SAFE_DIVISION_EPS:
            # If range is zero (constant array), IQR is also zero. Ratio is ill-defined.
            # Can define as 0 (no spread within the range) or 1 (all data is within the zero range).
            # Let's say 0 if IQR is also 0, otherwise could be an issue if IQR > 0 and range=0 (should not happen).
            return 0.0 
            
        if iqr_val is None:
            iqr_val = calculate_iqr(data_array_clean)
            
        return float(iqr_val / current_range)

    # --- Problem Data Extraction ---
    try:
        a_arr_orig = problem['alpha_array']
        n_arr_orig = problem['n_array']
        lattice_spacing = float(problem['lattice_spacing'])
        wavelength = float(problem.get('wavelength', SAFE_DIVISION_EPS))
        if wavelength < SAFE_DIVISION_EPS: wavelength = SAFE_DIVISION_EPS # Avoid division by zero later
        
        grid_size_input = problem.get('grid_size', [1, 1, 1])
        if isinstance(grid_size_input, (list, tuple)) and len(grid_size_input) == 3:
            grid_size = np.array(grid_size_input, dtype=int)
        elif isinstance(grid_size_input, np.ndarray) and grid_size_input.shape == (3,):
            grid_size = grid_size_input.astype(int)
        else: grid_size = np.array([1,1,1], dtype=int) # Default if malformed
        if np.any(grid_size <= 0): grid_size = np.array([1,1,1], dtype=int) # Ensure positive dimensions
        total_grid_cells = np.prod(grid_size)
        if total_grid_cells == 0: total_grid_cells = 1 # Should not happen if grid_size corrected
    except KeyError: # If essential keys are missing
        return np.nan_to_num(final_vector, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception: # Catch any other unexpected errors during data extraction
        return np.nan_to_num(final_vector, nan=0.0, posinf=0.0, neginf=0.0)

    # Process alpha array (active elements based on non-zero magnitude)
    try:
        if not isinstance(a_arr_orig, np.ndarray): a_arr_orig = np.array(a_arr_orig, dtype=complex)
        else: a_arr_orig = a_arr_orig.astype(complex)
        # Identify active alpha values (non-zero)
        a_nonzero_mask = np.abs(a_arr_orig) > SAFE_DIVISION_EPS
        alpha_active = a_arr_orig[a_nonzero_mask]
        num_active_alpha = alpha_active.size
    except Exception:
        alpha_active = np.array([], dtype=complex); num_active_alpha = 0

    # Process n array (using the same mask as alpha for consistency in active sites)
    try:
        if not isinstance(n_arr_orig, np.ndarray): n_arr_orig = np.array(n_arr_orig, dtype=complex)
        else: n_arr_orig = n_arr_orig.astype(complex)
        n_active = n_arr_orig[a_nonzero_mask] # n_active uses same mask as alpha_active
        num_active_n = n_active.size
    except Exception:
        n_active = np.array([], dtype=complex); num_active_n = 0


    # --- I. Alpha Array Analysis Block (Indices 0-30, Total 31 features) ---
    if num_active_alpha > 0:
        # Index 0: Ratio of active alpha elements to total grid cells
        final_vector[0] = num_active_alpha / total_grid_cells
        unique_alpha_vals = np.unique(alpha_active)
        count_unique_alpha = len(unique_alpha_vals)
        # Index 1: Inverse of unique alpha count (diversity measure)
        final_vector[1] = 1.0 / count_unique_alpha if count_unique_alpha > 0 else 0.0

        # Alpha Magnitude (Normalized by Max Magnitude) - Stats (Indices 2-10, 9 features)
        alpha_mags_abs = np.abs(alpha_active)
        max_alpha_mag = np.max(alpha_mags_abs)
        # Normalize magnitudes to [0,1] by dividing by max_alpha_mag.
        # If max_alpha_mag is zero (e.g., all active alphas are zero), all normalized mags are zero.
        alpha_mags_norm = alpha_mags_abs / (max_alpha_mag + SAFE_DIVISION_EPS)
        
        final_vector[2] = np.mean(alpha_mags_norm)
        final_vector[3] = np.median(alpha_mags_norm)
        # Max is implicitly 1 (or 0 if all alpha_active were 0), so not stored as a separate feature.
        final_vector[4] = np.min(alpha_mags_norm)
        final_vector[5] = np.std(alpha_mags_norm)
        skew_a_mag, kurt_a_mag = calculate_simple_shape_stats_reworked(alpha_mags_norm)
        final_vector[6] = skew_a_mag
        final_vector[7] = kurt_a_mag
        iqr_a_mag = calculate_iqr(alpha_mags_norm)
        final_vector[8] = iqr_a_mag
        final_vector[9] = calculate_cv(alpha_mags_norm)
        final_vector[10] = calculate_iqr_to_range_ratio(alpha_mags_norm, iqr_a_mag)

        # Alpha Phase (Original [0,π] mapped to [-1,1]) - Stats (Indices 11-20, 10 features)
        # Phase from arccos(Re/|alpha|) gives [0, pi]
        # Max with SAFE_DIVISION_EPS to avoid division by zero if alpha_mags_abs contains zeros
        alpha_phase_0_pi = np.arccos(np.clip(np.real(alpha_active) / np.maximum(alpha_mags_abs, SAFE_DIVISION_EPS), -1.0, 1.0))
        # Normalize [0, pi] to [-1, 1]
        alpha_phase_norm = (alpha_phase_0_pi / math.pi) * 2.0 - 1.0
        
        final_vector[11] = np.mean(alpha_phase_norm)
        final_vector[12] = np.median(alpha_phase_norm)
        final_vector[13] = np.max(alpha_phase_norm)
        final_vector[14] = np.min(alpha_phase_norm)
        final_vector[15] = np.std(alpha_phase_norm)
        skew_a_phase, kurt_a_phase = calculate_simple_shape_stats_reworked(alpha_phase_norm)
        final_vector[16] = skew_a_phase
        final_vector[17] = kurt_a_phase
        iqr_a_phase = calculate_iqr(alpha_phase_norm)
        final_vector[18] = iqr_a_phase
        final_vector[19] = calculate_cv(alpha_phase_norm)
        final_vector[20] = calculate_iqr_to_range_ratio(alpha_phase_norm, iqr_a_phase)

        # Alpha Cross-features and other general alpha features
        # Index 21: Correlation between normalized alpha magnitude and normalized alpha phase
        final_vector[21] = calculate_correlation(alpha_mags_norm, alpha_phase_norm)
        # Index 22: Ratio of alpha elements with positive real part
        final_vector[22] = np.sum(np.real(alpha_active) > 0) / num_active_alpha
        
        # Indices 23-26: Cross-value features (phase at max/min magnitude, magnitude at max/min phase)
        idx_max_amag = np.argmax(alpha_mags_abs) # Index from original absolute magnitudes
        final_vector[23] = alpha_phase_norm[idx_max_amag]       # Phase at Max (abs) Mag
        idx_min_amag = np.argmin(alpha_mags_abs) # Index from original absolute magnitudes
        final_vector[24] = alpha_phase_norm[idx_min_amag]       # Phase at Min (abs) Mag
        
        idx_max_aphase = np.argmax(alpha_phase_norm)
        final_vector[25] = alpha_mags_norm[idx_max_aphase] # Normalized Mag at Max Phase
        idx_min_aphase = np.argmin(alpha_phase_norm)
        final_vector[26] = alpha_mags_norm[idx_min_aphase] # Normalized Mag at Min Phase

        # Alpha Phase Zones (Indices 27-30, 4 features)
        # Normalized phase [-1,1] maps to physical angle [0,π]. Tolerance angle defines zones.
        # Zone 1: Near 0 rad (norm_phase near -1)
        upper1_limit_norm = (AXIS_ZONE_TOLERANCE_ANGLE / math.pi) * 2.0 - 1.0
        final_vector[27] = np.sum((alpha_phase_norm >= -1.0) & (alpha_phase_norm < upper1_limit_norm)) / num_active_alpha
        
        # Zone 2: Near pi/2 rad, from below (norm_phase approaching 0 from negative side)
        lower2_limit_norm = ((math.pi/2.0 - AXIS_ZONE_TOLERANCE_ANGLE) / math.pi) * 2.0 - 1.0
        pi_half_norm_equiv = 0.0 # Corresponds to pi/2 physical angle
        final_vector[28] = np.sum((alpha_phase_norm >= lower2_limit_norm) & (alpha_phase_norm < pi_half_norm_equiv)) / num_active_alpha
        
        # Zone 3: Near pi/2 rad, from above (norm_phase moving from 0 towards positive side)
        upper3_limit_norm = ((math.pi/2.0 + AXIS_ZONE_TOLERANCE_ANGLE) / math.pi) * 2.0 - 1.0
        final_vector[29] = np.sum((alpha_phase_norm > pi_half_norm_equiv) & (alpha_phase_norm <= upper3_limit_norm)) / num_active_alpha
        
        # Zone 4: Near pi rad (norm_phase near 1)
        lower4_limit_norm = ((math.pi - AXIS_ZONE_TOLERANCE_ANGLE) / math.pi) * 2.0 - 1.0
        final_vector[30] = np.sum((alpha_phase_norm > lower4_limit_norm) & (alpha_phase_norm <= 1.0)) / num_active_alpha
    # End of Alpha Block

    # --- II. N Array Analysis Block (Indices 31-58, Total 28 features) ---
    if num_active_n > 0: # n_active has same size as alpha_active
        # Index 31: Inverse of unique n count
        unique_n_vals = np.unique(n_active)
        count_unique_n = len(unique_n_vals)
        final_vector[31] = 1.0 / count_unique_n if count_unique_n > 0 else 0.0

        # N Magnitude (Linearly-normalized by N_MAG_NORM_DIVISOR) - Stats (Indices 32-41, 10 features)
        n_mags_abs = np.abs(n_active)
        n_mags_lin_norm = n_mags_abs / N_MAG_NORM_DIVISOR # Linear normalization
        
        final_vector[32] = np.mean(n_mags_lin_norm)
        final_vector[33] = np.median(n_mags_lin_norm)
        final_vector[34] = np.max(n_mags_lin_norm)
        final_vector[35] = np.min(n_mags_lin_norm)
        final_vector[36] = np.std(n_mags_lin_norm)
        skew_n_mag, kurt_n_mag = calculate_simple_shape_stats_reworked(n_mags_lin_norm)
        final_vector[37] = skew_n_mag
        final_vector[38] = kurt_n_mag
        iqr_n_mag = calculate_iqr(n_mags_lin_norm)
        final_vector[39] = iqr_n_mag
        final_vector[40] = calculate_cv(n_mags_lin_norm)
        final_vector[41] = calculate_iqr_to_range_ratio(n_mags_lin_norm, iqr_n_mag)

        # N Phase (Original np.angle clipped to [0,π/2] mapped to [0,1]) - Stats (Indices 42-51, 10 features)
        raw_n_phases = np.angle(n_active) # Result is in (-pi, pi]
        n_phase_0_pi_half = np.clip(raw_n_phases, 0, math.pi/2.0) # Clip to physical range [0, pi/2]
        # Normalize [0, pi/2] to [0, 1]
        n_phase_norm = n_phase_0_pi_half / (math.pi / 2.0) 
        
        final_vector[42] = np.mean(n_phase_norm)
        final_vector[43] = np.median(n_phase_norm)
        final_vector[44] = np.max(n_phase_norm)
        final_vector[45] = np.min(n_phase_norm)
        final_vector[46] = np.std(n_phase_norm)
        skew_n_phase, kurt_n_phase = calculate_simple_shape_stats_reworked(n_phase_norm)
        final_vector[47] = skew_n_phase
        final_vector[48] = kurt_n_phase
        iqr_n_phase = calculate_iqr(n_phase_norm)
        final_vector[49] = iqr_n_phase
        final_vector[50] = calculate_cv(n_phase_norm)
        final_vector[51] = calculate_iqr_to_range_ratio(n_phase_norm, iqr_n_phase)

        # N Cross-features and other general N features
        # Index 52: Correlation between normalized N magnitude and normalized N phase
        final_vector[52] = calculate_correlation(n_mags_lin_norm, n_phase_norm)
        
        # Indices 53-56: Cross-value features for N
        idx_max_nmag = np.argmax(n_mags_abs) # Index from original absolute magnitudes
        final_vector[53] = n_phase_norm[idx_max_nmag]       # Phase at Max (abs) Mag
        idx_min_nmag = np.argmin(n_mags_abs) # Index from original absolute magnitudes
        final_vector[54] = n_phase_norm[idx_min_nmag]       # Phase at Min (abs) Mag
        
        idx_max_nphase = np.argmax(n_phase_norm)
        final_vector[55] = n_mags_lin_norm[idx_max_nphase] # Normalized Mag at Max Phase
        idx_min_nphase = np.argmin(n_phase_norm)
        final_vector[56] = n_mags_lin_norm[idx_min_nphase] # Normalized Mag at Min Phase
        
        # N Phase Zones (Indices 57-58, 2 features)
        # Normalized phase [0,1] maps to physical angle [0,pi/2].
        # Zone 1: Near 0 rad (norm_phase near 0)
        upperN1_limit_norm = AXIS_ZONE_TOLERANCE_ANGLE / (math.pi/2.0)
        final_vector[57] = np.sum((n_phase_norm >= 0.0) & (n_phase_norm < upperN1_limit_norm)) / num_active_n
        
        # Zone 2: Near pi/2 rad (norm_phase near 1)
        lowerN2_limit_norm = (math.pi/2.0 - AXIS_ZONE_TOLERANCE_ANGLE) / (math.pi/2.0)
        final_vector[58] = np.sum((n_phase_norm > lowerN2_limit_norm) & (n_phase_norm <= 1.0)) / num_active_n
    # End of N Block

    # --- III. Cross Alpha-N Features (Indices 59-60, Total 2 features) ---
    if num_active_alpha > 0 and num_active_n > 0: # Ensure both have active elements
        # Index 59: Correlation of absolute magnitudes of alpha and n
        final_vector[59] = calculate_correlation(alpha_mags_abs, n_mags_abs)
        # Index 60: Correlation of normalized phases of alpha and n
        final_vector[60] = calculate_correlation(alpha_phase_norm, n_phase_norm)
    # End of Cross Alpha-N Block

    # --- IV. Geometric Features (Indices 61-64, Total 4 features) ---
    # Ratios of physical dimensions to wavelength, or lattice spacing to wavelength.
    try:
        final_vector[61] = (grid_size[0] * lattice_spacing / wavelength)
        final_vector[62] = (grid_size[1] * lattice_spacing / wavelength)
        final_vector[63] = (grid_size[2] * lattice_spacing / wavelength)
        final_vector[64] = (lattice_spacing * 10.0 / wavelength) # Scaled lattice spacing to wavelength ratio
    except Exception: pass # In case of issues (e.g., wavelength=0 already handled, but for safety)
    # End of Geometric Block

    # --- V. FFT Features (Indices 65-75, Total 11 features) ---
    # FFT is performed on the original alpha_array, reshaped to grid_size.
    # Features are derived from the magnitude of the shifted FFT.
    alpha_for_fft = np.zeros(tuple(grid_size), dtype=complex)
    try:
        # Reshape original alpha array only if its size matches the total grid cells.
        # Otherwise, FFT is on a zero array, leading to zero-valued FFT features.
        if a_arr_orig.size == total_grid_cells:
            alpha_for_fft = a_arr_orig.reshape(tuple(grid_size))

        # Proceed if FFT input has non-zeros AND there were active alphas
        # (to avoid div by zero if sum_all_spectral_mags is zero but dc_val is also zero from zero input)
        if np.any(alpha_for_fft) and num_active_alpha > 0:
            ft_complex = np.fft.fftn(alpha_for_fft)
            ft_shifted = np.fft.fftshift(ft_complex) # Shift DC component to center
            ft_magnitude = np.abs(ft_shifted) # Magnitude of FFT components

            center_indices_nd = tuple()
            dc_val = 0.0
            valid_center = False

            if ft_magnitude.ndim == 0: # Scalar input (e.g., 1x1x1 grid)
                dc_val = ft_magnitude.item() if ft_magnitude.size ==1 else 0.0
                # No AC components for scalar.
            else: # Multi-dimensional FFT
                center_indices_nd = tuple(d // 2 for d in ft_magnitude.shape)
                valid_center = True
                # Validate center indices
                for i, ci_val in enumerate(center_indices_nd):
                    if not (0 <= ci_val < ft_magnitude.shape[i]):
                        valid_center = False; break
                if valid_center:
                    dc_val = ft_magnitude[center_indices_nd]
                else: # Should not happen with d // 2 if dims > 0
                    dc_val = 0.0
            
            sum_all_spectral_mags = np.sum(ft_magnitude)
            # Index 65: Ratio of DC component magnitude to sum of all spectral magnitudes
            if sum_all_spectral_mags > SAFE_DIVISION_EPS:
                final_vector[65] = dc_val / sum_all_spectral_mags

            # Prepare AC components: flatten magnitudes and remove DC
            ft_ac_magnitudes_flat = np.array([])
            if ft_magnitude.ndim > 0 and valid_center and ft_magnitude.size > 1:
                temp_flat_mags = ft_magnitude.flatten()
                flat_dc_idx = np.ravel_multi_index(center_indices_nd, ft_magnitude.shape)
                if 0 <= flat_dc_idx < temp_flat_mags.size:
                    ft_ac_magnitudes_flat = np.delete(temp_flat_mags, flat_dc_idx)
            # If ft_magnitude.ndim == 0 (scalar), or size <=1, ft_ac_magnitudes_flat remains empty.
            
            # Normalize AC magnitudes by overall max FFT magnitude (including DC)
            max_overall_ft_mag = np.max(ft_magnitude) if ft_magnitude.size > 0 else 0.0
            ft_ac_magnitudes_norm_flat = np.array([])

            if max_overall_ft_mag > SAFE_DIVISION_EPS and ft_ac_magnitudes_flat.size > 0:
                # Normalize the AC components by the max overall magnitude
                ft_ac_magnitudes_norm_flat = ft_ac_magnitudes_flat / max_overall_ft_mag


            if ft_ac_magnitudes_norm_flat.size > 0:
                # Index 66: Mean of normalized AC magnitudes
                current_mean_ac_norm = np.mean(ft_ac_magnitudes_norm_flat)
                final_vector[66] = current_mean_ac_norm
                # Index 67: Standard deviation of normalized AC magnitudes
                final_vector[67] = np.std(ft_ac_magnitudes_norm_flat) if ft_ac_magnitudes_norm_flat.size > 1 else 0.0
                
                # Index 72: Spectral flatness of normalized AC magnitudes
                # (Geometric mean / Arithmetic mean) of AC magnitudes. Closer to 1 means flatter spectrum.
                log_ac_mags_norm = np.log(ft_ac_magnitudes_norm_flat + SAFE_DIVISION_EPS) # Epsilon for log stability
                mean_log_ac_mags_norm = np.mean(log_ac_mags_norm)
                if current_mean_ac_norm > SAFE_DIVISION_EPS:
                    spectral_flatness = np.exp(mean_log_ac_mags_norm) / current_mean_ac_norm
                    final_vector[72] = spectral_flatness
                # else, remains 0.0

                # Index 73: Ratio of 2nd largest to largest normalized AC peak
                if ft_ac_magnitudes_norm_flat.size >= 2:
                    sorted_ac_mags_norm = np.sort(ft_ac_magnitudes_norm_flat)[::-1] # Sort descending
                    largest_ac_peak_norm = sorted_ac_mags_norm[0]
                    second_largest_ac_peak_norm = sorted_ac_mags_norm[1]
                    if largest_ac_peak_norm > SAFE_DIVISION_EPS:
                        final_vector[73] = second_largest_ac_peak_norm / largest_ac_peak_norm
                # else, remains 0.0

            # Features based on un-normalized AC magnitudes, relative to total AC sum
            if ft_ac_magnitudes_flat.size > 0:
                total_ac_sum = np.sum(ft_ac_magnitudes_flat)
                if total_ac_sum > SAFE_DIVISION_EPS:
                    # Index 68: Ratio of AC energy in a 3x3x3 box (excluding DC) around center to total AC energy
                    if ft_magnitude.ndim == 3 and all(d >= 3 for d in ft_magnitude.shape) and valid_center:
                        s_ = ft_magnitude.shape; ci_box = center_indices_nd
                        # Define box slices carefully to stay within bounds
                        box = ft_magnitude[max(0,ci_box[0]-1):min(s_[0],ci_box[0]+2),
                                           max(0,ci_box[1]-1):min(s_[1],ci_box[1]+2),
                                           max(0,ci_box[2]-1):min(s_[2],ci_box[2]+2)]
                        box_sum_ac_calc = np.sum(box) - dc_val # Subtract DC from sum of box
                        final_vector[68] = np.clip(max(0.0, box_sum_ac_calc) / total_ac_sum, 0.0, 1.0)
                    
                    dims = ft_magnitude.shape; ci_axis = center_indices_nd
                    # Index 69: Ratio of AC energy along X-axis to total AC energy
                    if ft_magnitude.ndim >= 1 and dims[0] > 1 and valid_center:
                        x_line_mags_calc = np.array([])
                        if len(dims) == 1: x_line_mags_calc = ft_magnitude[:]
                        elif len(dims) == 2: x_line_mags_calc = ft_magnitude[:, ci_axis[1]]
                        elif len(dims) == 3: x_line_mags_calc = ft_magnitude[:, ci_axis[1], ci_axis[2]]
                        if x_line_mags_calc.size > 0:
                           final_vector[69] = np.clip((np.sum(x_line_mags_calc) - dc_val) / total_ac_sum, 0.0, 1.0)
                    
                    # Index 70: Ratio of AC energy along Y-axis to total AC energy
                    if ft_magnitude.ndim >= 2 and dims[1] > 1 and valid_center:
                        y_line_mags_calc = np.array([])
                        if len(dims) == 2: y_line_mags_calc = ft_magnitude[ci_axis[0], :]
                        elif len(dims) == 3: y_line_mags_calc = ft_magnitude[ci_axis[0], :, ci_axis[2]]
                        if y_line_mags_calc.size > 0:
                            final_vector[70] = np.clip((np.sum(y_line_mags_calc) - dc_val) / total_ac_sum, 0.0, 1.0)
                    
                    # Index 71: Ratio of AC energy along Z-axis to total AC energy
                    if ft_magnitude.ndim == 3 and dims[2] > 1 and valid_center:
                        z_line_mags_calc = ft_magnitude[ci_axis[0], ci_axis[1], :]
                        if z_line_mags_calc.size > 0:
                           final_vector[71] = np.clip((np.sum(z_line_mags_calc) - dc_val) / total_ac_sum, 0.0, 1.0)

                    # Indices 74, 75: Radial AC Energy Distribution
                    k_indices_list = []
                    for i_dim, dim_size in enumerate(ft_shifted.shape):
                        k_indices_list.append(np.arange(dim_size) - center_indices_nd[i_dim])

                    k_dist_sq_grid = np.array([0.0]) # Default for scalar
                    if ft_shifted.ndim == 1: k_dist_sq_grid = k_indices_list[0]**2
                    elif ft_shifted.ndim == 2:
                        k_grid_x, k_grid_y = np.meshgrid(k_indices_list[0], k_indices_list[1], indexing='ij')
                        k_dist_sq_grid = k_grid_x**2 + k_grid_y**2
                    elif ft_shifted.ndim == 3:
                        k_grid_x, k_grid_y, k_grid_z = np.meshgrid(k_indices_list[0], k_indices_list[1], k_indices_list[2], indexing='ij')
                        k_dist_sq_grid = k_grid_x**2 + k_grid_y**2 + k_grid_z**2
                    
                    k_dist_flat = np.sqrt(k_dist_sq_grid).flatten()
                    
                    # Use ft_ac_magnitudes_flat (un-normalized AC mags) and corresponding k-distances
                    # Need to reconstruct k-distances for AC components only
                    ac_k_dist_radial = np.array([])
                    if ft_magnitude.ndim > 0 and valid_center and ft_magnitude.size > 1 and k_dist_flat.size == ft_magnitude.size:
                        flat_dc_idx_radial = np.ravel_multi_index(center_indices_nd, ft_magnitude.shape)
                        if 0 <= flat_dc_idx_radial < k_dist_flat.size:
                            ac_k_dist_radial = np.delete(k_dist_flat, flat_dc_idx_radial)
                    
                    if ac_k_dist_radial.size > 0 and ft_ac_magnitudes_flat.size == ac_k_dist_radial.size:
                        max_ac_k_dist = np.max(ac_k_dist_radial) if ac_k_dist_radial.size > 0 else 0.0
                        if max_ac_k_dist > SAFE_DIVISION_EPS: # Ensure there's some spread in k-space for AC
                            shell_mid_limit = max_ac_k_dist / 2.0
                            sum_inner_shell = np.sum(ft_ac_magnitudes_flat[ac_k_dist_radial <= shell_mid_limit])
                            sum_outer_shell = np.sum(ft_ac_magnitudes_flat[ac_k_dist_radial > shell_mid_limit])
                            
                            final_vector[74] = sum_inner_shell / total_ac_sum
                            final_vector[75] = sum_outer_shell / total_ac_sum
                        elif ac_k_dist_radial.size > 0 : # All AC components at k_dist=0 (should mean all AC energy is inner)
                            final_vector[74] = 1.0 
                            final_vector[75] = 0.0
    except Exception: pass # FFT features will remain 0.0 if any error occurs
    # End of FFT Block
            
    # --- Final Cleanup ---
    # Ensure all values are finite and not NaN, replacing with 0.0
    final_vector = np.nan_to_num(final_vector, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Feature Index Guide ---
    # ALPHA ARRAY FEATURES (Indices 0-30)
    #   0: Alpha Active Ratio (num_active_alpha / total_grid_cells)
    #   1: Alpha Inverse Unique Count (1.0 / count_unique_alpha)
    #   Alpha Magnitude (Normalized by Max Magnitude) Stats (Indices 2-10):
    #     2: Mean(alpha_mags_norm)
    #     3: Median(alpha_mags_norm)
    #     4: Min(alpha_mags_norm)
    #     5: Std(alpha_mags_norm)
    #     6: Skewness(alpha_mags_norm)
    #     7: Kurtosis(alpha_mags_norm)
    #     8: IQR(alpha_mags_norm)
    #     9: CV(alpha_mags_norm)
    #     10: IQR-to-Range Ratio(alpha_mags_norm)
    #   Alpha Phase (Normalized [-1,1]) Stats (Indices 11-20):
    #     11: Mean(alpha_phase_norm)
    #     12: Median(alpha_phase_norm)
    #     13: Max(alpha_phase_norm)
    #     14: Min(alpha_phase_norm)
    #     15: Std(alpha_phase_norm)
    #     16: Skewness(alpha_phase_norm)
    #     17: Kurtosis(alpha_phase_norm)
    #     18: IQR(alpha_phase_norm)
    #     19: CV(alpha_phase_norm)
    #     20: IQR-to-Range Ratio(alpha_phase_norm)
    #   Alpha Cross-features & General (Indices 21-30):
    #     21: Correlation(alpha_mags_norm, alpha_phase_norm)
    #     22: Ratio Re(alpha) > 0
    #     23: Phase at Max Absolute Magnitude (alpha_phase_norm[idx_max_amag])
    #     24: Phase at Min Absolute Magnitude (alpha_phase_norm[idx_min_amag])
    #     25: Normalized Magnitude at Max Phase (alpha_mags_norm[idx_max_aphase])
    #     26: Normalized Magnitude at Min Phase (alpha_mags_norm[idx_min_aphase])
    #     27: Alpha Phase Zone 1 (Near 0 rad)
    #     28: Alpha Phase Zone 2 (Near pi/2 rad, from below)
    #     29: Alpha Phase Zone 3 (Near pi/2 rad, from above)
    #     30: Alpha Phase Zone 4 (Near pi rad)
    # N ARRAY FEATURES (Indices 31-58)
    #   31: N Inverse Unique Count (1.0 / count_unique_n)
    #   N Magnitude (Linearly Normalized) Stats (Indices 32-41):
    #     32: Mean(n_mags_lin_norm)
    #     33: Median(n_mags_lin_norm)
    #     34: Max(n_mags_lin_norm)
    #     35: Min(n_mags_lin_norm)
    #     36: Std(n_mags_lin_norm)
    #     37: Skewness(n_mags_lin_norm)
    #     38: Kurtosis(n_mags_lin_norm)
    #     39: IQR(n_mags_lin_norm)
    #     40: CV(n_mags_lin_norm)
    #     41: IQR-to-Range Ratio(n_mags_lin_norm)
    #   N Phase (Normalized [0,1]) Stats (Indices 42-51):
    #     42: Mean(n_phase_norm)
    #     43: Median(n_phase_norm)
    #     44: Max(n_phase_norm)
    #     45: Min(n_phase_norm)
    #     46: Std(n_phase_norm)
    #     47: Skewness(n_phase_norm)
    #     48: Kurtosis(n_phase_norm)
    #     49: IQR(n_phase_norm)
    #     50: CV(n_phase_norm)
    #     51: IQR-to-Range Ratio(n_phase_norm)
    #   N Cross-features & General (Indices 52-58):
    #     52: Correlation(n_mags_lin_norm, n_phase_norm)
    #     53: Phase at Max Absolute Magnitude (n_phase_norm[idx_max_nmag])
    #     54: Phase at Min Absolute Magnitude (n_phase_norm[idx_min_nmag])
    #     55: Normalized Magnitude at Max Phase (n_mags_lin_norm[idx_max_nphase])
    #     56: Normalized Magnitude at Min Phase (n_mags_lin_norm[idx_min_nphase])
    #     57: N Phase Zone 1 (Near 0 rad)
    #     58: N Phase Zone 2 (Near pi/2 rad)
    # CROSS ALPHA-N FEATURES (Indices 59-60)
    #   59: Correlation(abs(alpha_active), abs(n_active))
    #   60: Correlation(alpha_phase_norm, n_phase_norm)
    # GEOMETRIC FEATURES (Indices 61-64)
    #   61: GridDim_X * LatticeSpacing / Wavelength
    #   62: GridDim_Y * LatticeSpacing / Wavelength
    #   63: GridDim_Z * LatticeSpacing / Wavelength
    #   64: Scaled LatticeSpacing / Wavelength (lattice_spacing * 10.0 / wavelength)
    # FFT FEATURES (Indices 65-75)
    #   65: FFT DC Ratio (dc_val / sum_all_spectral_mags)
    #   66: FFT AC Mean (Normalized by max overall FFT magnitude)
    #   67: FFT AC Std (Normalized by max overall FFT magnitude)
    #   68: FFT AC Box Ratio (3x3x3 neighborhood around DC, AC energy / total AC energy)
    #   69: FFT AC X-axis Ratio (AC energy along X / total AC energy)
    #   70: FFT AC Y-axis Ratio (AC energy along Y / total AC energy)
    #   71: FFT AC Z-axis Ratio (AC energy along Z / total AC energy)
    #   72: FFT AC Spectral Flatness (Normalized AC magnitudes)
    #   73: FFT Ratio of 2nd Largest to Largest AC Peak (Normalized AC magnitudes)
    #   74: FFT AC Radial Energy Inner Ratio (Energy in inner k-space shell / total AC energy)
    #   75: FFT AC Radial Energy Outer Ratio (Energy in outer k-space shell / total AC energy)

    return final_vector
#!/usr/bin/env python3
"""
Interactive Command-Line Interface for AI-Driven DDA Problem Solver
Allows users to manually specify shapes and get AI predictions
Designed to run in Jupyter notebook with existing functions available
"""

import numpy as np
import sys
from stable_baselines3 import SAC
import cupy as cp
from ExcitedField import E_field_solver

# Update this to your actual model path
MODEL_PATH = "dda_sac_model_gridE11.zip"

def scale_action_new(action, alpha_array):
    """Scale action using the new scaling function"""
    action = np.asarray(action)
    alpha_array = np.asarray(alpha_array)
    if action.shape != (2,):
         raise ValueError(f"scale_action_new expects a 2-element action, got shape {action.shape}")
    non_zero_alpha = alpha_array[alpha_array != 0]
    if len(non_zero_alpha) > 0:
        magnitudes = np.abs(non_zero_alpha)
        max_magnitude = float(np.max(magnitudes))
    else:
        max_magnitude = 1.0
    scale_factor = 1.5 * max_magnitude
    scaled_real = action[0] * scale_factor
    normalized_imag = (action[1] + 1) / 2
    scaled_imag = normalized_imag * scale_factor
    return np.array([scaled_real, scaled_imag], dtype=np.float64)

def predict_with_ai(model, problem):
    """Make prediction using the AI model with existing functions"""
    try:
        # Get normalized features using existing analyze_and_normalize11 function
        normalized_features = analyze_and_normalize11(problem)
        normalized_features = np.asarray(normalized_features, dtype=np.float32)
        
        # Get model's expected feature dimension
        feature_dim = model.observation_space.shape[0]
        if normalized_features.size != feature_dim:
            raise ValueError(f"Analysis function returned {normalized_features.size} features, but model expects {feature_dim}")
        
        observation = normalized_features.reshape(feature_dim,)
        
        # Make prediction
        action, _ = model.predict(observation, deterministic=True)
        action = np.array(action)
        
        if action.shape[0] != 2:
            print(f"WARNING: Model prediction shape mismatch. Expected 2, got {action.shape}")
            return np.full(2, np.nan)
        
        # Scale the action using scale_action_new function
        alpha_array_np = np.asarray(problem['alpha_array'])
        scaled_action = scale_action_new(action, alpha_array_np)
        
        return scaled_action
        
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return np.full(2, np.nan)

def load_ai_model():
    """Load the trained AI model"""
    try:
        print(f"Loading AI model from {MODEL_PATH}...")
        model = SAC.load(MODEL_PATH, device='auto')
        print(f"Model loaded successfully (device: {model.device})")
        return model
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

def get_float_input(prompt, default=None):
    """Get validated float input from user"""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            value = float(user_input)
            
            if value == 0:
                print("Error: Value cannot be 0")
                continue
                
            return value
        except ValueError:
            print("Please enter a valid number.")

def get_int_input(prompt, default=None):
    """Get validated integer input from user"""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            value = int(user_input)
            
            if value == 0:
                print("Error: Value cannot be 0")
                continue
                
            return value
        except ValueError:
            print("Please enter a valid integer.")

def get_choice_input(prompt, choices, default=None):
    """Get validated choice input from user"""
    choices_str = "/".join(choices)
    while True:
        if default is not None:
            user_input = input(f"{prompt} ({choices_str}) (default: {default}): ").strip().lower()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt} ({choices_str}): ").strip().lower()
        
        if user_input in [c.lower() for c in choices]:
            return user_input
        print(f"Please choose from: {choices_str}")

def get_position():
    """Get 3D position from user"""
    print("Enter position coordinates:")
    x = get_float_input("  X coordinate", default=0.0)
    y = get_float_input("  Y coordinate", default=0.0)
    z = get_float_input("  Z coordinate", default=0.0)
    return [x, y, z]

def get_refractive_index():
    """Get complex refractive index from user"""
    print("Enter refractive index (n = real + imag*j):")
    
    # Real part cannot be zero
    while True:
        real = get_float_input("  Real part", default=1.5)
        if real != 0:
            break
        print("Error: Real part of refractive index cannot be 0")
    
    # Imaginary part CAN be zero
    while True:
        try:
            user_input = input("  Imaginary part (default: 0.0): ").strip()
            if not user_input:
                imag = 0.0
            else:
                imag = float(user_input)
            break
        except ValueError:
            print("Please enter a valid number.")
    
    return complex(real, imag)

def create_sphere():
    """Create sphere parameters"""
    print("\n--- Sphere Parameters ---")
    center = get_position()
    radius = get_float_input("Radius")
    n = get_refractive_index()
    
    return {
        'type': 'sphere',
        'center': center,
        'radius': radius,
        'n': n
    }

def create_cylinder():
    """Create cylinder parameters"""
    print("\n--- Cylinder Parameters ---")
    center = get_position()
    radius = get_float_input("Radius")
    height = get_float_input("Height")
    axis = get_choice_input("Axis direction", ['x', 'y', 'z'], default='z')
    n = get_refractive_index()
    
    return {
        'type': 'cylinder',
        'center': center,
        'radius': radius,
        'height': height,
        'axis': axis,
        'n': n
    }

def create_rectangle():
    """Create rectangular box parameters"""
    print("\n--- Rectangle Parameters ---")
    center = get_position()
    print("Enter dimensions:")
    dx = get_float_input("  X dimension")
    dy = get_float_input("  Y dimension")
    dz = get_float_input("  Z dimension")
    n = get_refractive_index()
    
    return {
        'type': 'rectangle',
        'center': center,
        'dimensions': [dx, dy, dz],
        'n': n
    }

def create_ellipsoid():
    """Create ellipsoid parameters"""
    print("\n--- Ellipsoid Parameters ---")
    center = get_position()
    print("Enter semi-axes:")
    a = get_float_input("  Semi-axis A")
    b = get_float_input("  Semi-axis B")
    c = get_float_input("  Semi-axis C")
    n = get_refractive_index()
    
    return {
        'type': 'ellipsoid',
        'center': center,
        'semi_axes': [a, b, c],
        'n': n
    }

def create_prism():
    """Create prism parameters"""
    print("\n--- Prism Parameters ---")
    center = get_position()
    radius = get_float_input("Radius (circumscribed circle)")
    height = get_float_input("Height")
    sides = get_int_input("Number of sides", default=6)
    axis = get_choice_input("Axis direction", ['x', 'y', 'z'], default='z')
    n = get_refractive_index()
    
    return {
        'type': 'prism',
        'center': center,
        'radius': radius,
        'height': height,
        'sides': sides,
        'axis': axis,
        'n': n
    }

def create_shape():
    """Create a shape based on user input"""
    shape_creators = {
        'sphere': create_sphere,
        'cylinder': create_cylinder,
        'rectangle': create_rectangle,
        'ellipsoid': create_ellipsoid,
        'prism': create_prism
    }
    
    print("\nAvailable shapes: sphere, cylinder, rectangle, ellipsoid, prism")
    shape_type = get_choice_input("What shape do you want to create?", 
                                 list(shape_creators.keys()))
    
    return shape_creators[shape_type]()

def get_wave_parameters():
    """Get wave parameters from user"""
    print("\n--- Wave Parameters ---")
    
    # Wave direction
    print("Wave direction:")
    use_axis = get_choice_input("Use axis-aligned direction?", ['yes', 'no'], default='yes')
    
    if use_axis == 'yes':
        axis = get_choice_input("Which axis?", ['x', 'y', 'z'], default='z')
        if axis == 'x':
            k_direction = np.array([1.0, 0.0, 0.0])
        elif axis == 'y':
            k_direction = np.array([0.0, 1.0, 0.0])
        else:
            k_direction = np.array([0.0, 0.0, 1.0])
    else:
        print("Enter wave direction vector (will be normalized):")
        kx = get_float_input("  K_x")
        ky = get_float_input("  K_y")
        kz = get_float_input("  K_z")
        k_direction = np.array([kx, ky, kz])
        k_direction = k_direction / np.linalg.norm(k_direction)
    
    # Polarization
    print("\nPolarization:")
    use_axis_pol = get_choice_input("Use axis-aligned polarization?", ['yes', 'no'], default='yes')
    
    if use_axis_pol == 'yes':
        # Find axes perpendicular to k_direction
        available_axes = []
        if abs(np.dot(k_direction, [1, 0, 0])) < 0.99:
            available_axes.append(('x', np.array([1.0, 0.0, 0.0])))
        if abs(np.dot(k_direction, [0, 1, 0])) < 0.99:
            available_axes.append(('y', np.array([0.0, 1.0, 0.0])))
        if abs(np.dot(k_direction, [0, 0, 1])) < 0.99:
            available_axes.append(('z', np.array([0.0, 0.0, 1.0])))
        
        if available_axes:
            axis_names = [name for name, _ in available_axes]
            pol_axis = get_choice_input(f"Polarization axis", axis_names, default=axis_names[0])
            E_polarization = next(vec for name, vec in available_axes if name == pol_axis)
            # Make it perpendicular to k_direction
            E_polarization = E_polarization - np.dot(E_polarization, k_direction) * k_direction
            E_polarization = E_polarization / np.linalg.norm(E_polarization)
        else:
            print("No axis-aligned polarization available, using random perpendicular")
            E_polarization = np.random.randn(3)
            E_polarization = E_polarization - np.dot(E_polarization, k_direction) * k_direction
            E_polarization = E_polarization / np.linalg.norm(E_polarization)
    else:
        print("Enter polarization vector (will be made perpendicular to k and normalized):")
        ex = get_float_input("  E_x")
        ey = get_float_input("  E_y")
        ez = get_float_input("  E_z")
        E_polarization = np.array([ex, ey, ez])
        E_polarization = E_polarization - np.dot(E_polarization, k_direction) * k_direction
        E_polarization = E_polarization / np.linalg.norm(E_polarization)
    
    # Wavelength
    wavelength = get_float_input("Wavelength", default=500.0)
    
    return k_direction, E_polarization, wavelength

def calculate_default_lattice_spacing(shapes, wavelength):
    """Calculate default lattice spacing based on shapes and wavelength"""
    max_refractive_index = 1.0  # Start with minimum
    
    for shape in shapes:
        n = shape['n']
        n_magnitude = abs(n)  # Get magnitude of complex refractive index
        max_refractive_index = max(max_refractive_index, n_magnitude)
    
    default_spacing = wavelength / (10 * max_refractive_index)
    return default_spacing

def get_simulation_parameters(shapes, wavelength):
    """Get simulation parameters with smart defaults"""
    print("\n--- Simulation Parameters ---")
    
    # Calculate default lattice spacing
    default_spacing = calculate_default_lattice_spacing(shapes, wavelength)
    
    print(f"Recommended lattice spacing: {default_spacing:.6f}")
    print("(Based on: wavelength / (10 * max_refractive_index))")
    
    use_default = get_choice_input("Use recommended lattice spacing?", ['yes', 'no'], default='yes')
    
    if use_default == 'yes':
        lattice_spacing = default_spacing
    else:
        lattice_spacing = get_float_input("Custom lattice spacing")
    
    print(f"Using lattice spacing: {lattice_spacing:.6f}")
    
    # Ask about saving results
    save_results = get_choice_input("Save polarization results to file?", ['yes', 'no'], default='no')
    filename = None
    if save_results == 'yes':
        filename = input("Enter filename (without extension): ").strip()
        if not filename:
            filename = "dda_results"
        filename = filename + ".txt"
    
    return lattice_spacing, filename

def build_problem_from_shapes(shapes, k_direction, E_polarization, wavelength, lattice_spacing):
    """Build a problem dictionary from user-defined shapes using existing functions"""
    try:
        # Create shape manager and add shapes
        sm = ShapeManager()
        
        for shape in shapes:
            if shape['type'] == 'sphere':
                sm.add_shape('sphere', center=shape['center'], radius=shape['radius'], n=shape['n'])
            elif shape['type'] == 'cylinder':
                sm.add_shape('cylinder', center=shape['center'], radius=shape['radius'],
                           height=shape['height'], axis=shape['axis'], n=shape['n'])
            elif shape['type'] == 'rectangle':
                sm.add_shape('rectangle', center=shape['center'], dimensions=shape['dimensions'], n=shape['n'])
            elif shape['type'] == 'ellipsoid':
                sm.add_shape('ellipsoid', center=shape['center'], semi_axes=shape['semi_axes'], n=shape['n'])
            elif shape['type'] == 'prism':
                sm.add_shape('prism', center=shape['center'], radius=shape['radius'],
                           height=shape['height'], sides=shape['sides'], axis=shape['axis'], n=shape['n'])
        
        # Calculate wave parameters
        k = 2 * np.pi / wavelength
        k_vec = k * k_direction
        
        # Process shapes using existing functions
        sm.process_shapes(k, k_direction, E_polarization, lattice_spacing)
        alpha_array, n_array, grid_size = create_alpha_and_n_arrays(sm, lattice_spacing)
        
        # Ensure grid_size is a tuple
        if not isinstance(grid_size, tuple):
            if isinstance(grid_size, (list, np.ndarray)):
                grid_size = tuple(map(int, grid_size))
            else:
                grid_size = (int(grid_size),) * 3
        
        # Generate incident field
        axes = [np.arange(0, d) * lattice_spacing for d in grid_size]
        X, Y, Z = np.meshgrid(*axes, indexing='ij')
        phase = np.exp(1j * (k_vec[0]*X + k_vec[1]*Y + k_vec[2]*Z))
        E_inc = np.zeros((*grid_size, 3), dtype=np.complex128)
        for j in range(3):
            E_inc[..., j] = E_polarization[j] * phase
        
        return {
            'alpha_array': alpha_array,
            'n_array': n_array,
            'grid_size': grid_size,
            'E_inc': E_inc,
            'k': k,
            'lattice_spacing': lattice_spacing,
            'k_vec': k_vec,
            'k_direction': k_direction,
            'E_polarization': E_polarization,
            'wavelength': wavelength
        }
        
    except Exception as e:
        print(f"ERROR building problem: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results_to_file(filename, problem, prediction, shapes, solver_result=None):
    """Save results to a text file including solver results"""
    try:
        with open(filename, 'w') as f:
            f.write("DDA Problem Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Problem parameters
            f.write("PROBLEM PARAMETERS:\n")
            f.write(f"Grid size: {problem['grid_size']}\n")
            f.write(f"Lattice spacing: {problem['lattice_spacing']:.6f}\n")
            f.write(f"Wavelength: {problem['wavelength']:.6f}\n")
            f.write(f"Wave direction: {problem['k_direction']}\n")
            f.write(f"Polarization: {problem['E_polarization']}\n")
            f.write(f"k (wave number): {problem['k']:.6f}\n\n")
            
            # Shapes
            f.write("SHAPES:\n")
            for i, shape in enumerate(shapes, 1):
                f.write(f"Shape {i}: {shape['type']}\n")
                f.write(f"  Center: {shape['center']}\n")
                f.write(f"  Refractive index: {shape['n']}\n")
                if shape['type'] == 'sphere':
                    f.write(f"  Radius: {shape['radius']}\n")
                elif shape['type'] == 'cylinder':
                    f.write(f"  Radius: {shape['radius']}\n")
                    f.write(f"  Height: {shape['height']}\n")
                    f.write(f"  Axis: {shape['axis']}\n")
                elif shape['type'] == 'rectangle':
                    f.write(f"  Dimensions: {shape['dimensions']}\n")
                elif shape['type'] == 'ellipsoid':
                    f.write(f"  Semi-axes: {shape['semi_axes']}\n")
                elif shape['type'] == 'prism':
                    f.write(f"  Radius: {shape['radius']}\n")
                    f.write(f"  Height: {shape['height']}\n")
                    f.write(f"  Sides: {shape['sides']}\n")
                    f.write(f"  Axis: {shape['axis']}\n")
                f.write("\n")
            
            # AI Prediction
            if not np.any(np.isnan(prediction)):
                multiplier = complex(prediction[0], prediction[1])
                f.write("AI PREDICTION:\n")
                f.write(f"Optimal multiplier: {multiplier.real:.6f} + {multiplier.imag:.6f}j\n")
                f.write(f"Magnitude: {abs(multiplier):.6f}\n")
                f.write(f"Phase: {np.angle(multiplier):.6f} radians ({np.degrees(np.angle(multiplier)):.2f} degrees)\n\n")
            else:
                f.write("AI PREDICTION: Failed\n\n")
            
            # Solver Results
            if solver_result is not None:
                # run_solver returns 3 values: iterations, polarization_array, mean_norm
                iterations, polarization_array, final_norm = solver_result
                
                f.write("SOLVER RESULTS:\n")
                f.write(f"Iterations: {iterations}\n")
                f.write(f"Final norm: {final_norm:.6e}\n")
                f.write("First norm: Not available from run_solver\n")
                
                # Determine convergence status
                if iterations is not None:
                    if iterations > 0 and iterations < 10000:
                        f.write("Status: Converged\n")
                    elif iterations == 10000:
                        f.write("Status: Did not converge (max iterations reached)\n")
                    elif iterations < 0:
                        f.write("Status: Failed (NaN or error condition)\n")
                    else:
                        f.write("Status: Unknown\n")
                
                # Save polarization array info and data
                if polarization_array is not None:
                    # Convert to CPU if it's on GPU
                    if hasattr(polarization_array, 'device'):  # Check if it's a CuPy array
                        pol_array_cpu = cp.asnumpy(polarization_array)
                    else:
                        pol_array_cpu = np.array(polarization_array)
                    
                    f.write(f"Polarization array shape: {pol_array_cpu.shape}\n")
                    f.write(f"Polarization array dtype: {pol_array_cpu.dtype}\n")
                    
                    # Save the full polarization array to a separate numpy file
                    pol_filename = filename.replace('.txt', '_polarization.npy')
                    np.save(pol_filename, pol_array_cpu)
                    f.write(f"Full polarization array saved to: {pol_filename}\n")
                    print(f"Polarization array saved to: {pol_filename}")
                else:
                    f.write("Polarization array: Not available\n")
            else:
                f.write("SOLVER RESULTS: Not run\n")
            
            f.write(f"\nResults saved on: {np.datetime64('now')}\n")
        
        print(f"Results saved to: {filename}")
        
    except Exception as e:
        print(f"Error saving file: {e}")

def create_field_visualization(problem, solver_result, interaction_matrix):
    """Create Dash web app for field visualization"""
    try:
        import dash
        from dash import dcc, html, Input, Output, callback
        import plotly.graph_objects as go
        
        # Extract solver results
        iterations, polarization_array, final_norm = solver_result
        
        # Ensure polarization array is on CPU
        if hasattr(polarization_array, 'device'):  # CuPy array
            polarization_array_cpu = cp.asnumpy(polarization_array)
        else:
            polarization_array_cpu = np.array(polarization_array)
        
        print(f"DEBUG: Polarization array shape: {polarization_array_cpu.shape}")
        print(f"DEBUG: Polarization array size: {polarization_array_cpu.size}")
        
        # Calculate E-field using existing functions
        E_field = E_field_solver(polarization_array_cpu, interaction_matrix, problem['grid_size'])
        E_magnitude = cp.sqrt(cp.sum(cp.abs(E_field)**2, axis=-1))
        
        # Convert E_inc to GPU if needed
        E_inc = problem['E_inc']
        if isinstance(E_inc, np.ndarray):
            E_inc_gpu = cp.asarray(E_inc)
        else:
            E_inc_gpu = E_inc
        
        # Calculate total field
        E_total = E_field + E_inc_gpu
        E_total_magnitude = cp.sqrt(cp.sum(cp.abs(E_total)**2, axis=-1))
        
        # Transfer to CPU
        E_magnitude_cpu = cp.asnumpy(E_magnitude).astype(np.float32)
        E_total_magnitude_cpu = cp.asnumpy(E_total_magnitude).astype(np.float32)
        
        nx, ny, nz = E_magnitude_cpu.shape[:3]
        
        # Initialize Dash app
        app = dash.Dash(__name__)
        
        # App layout
        app.layout = html.Div([
            html.H1(f"Electric Field Magnitude - Converged in {iterations} iterations", 
                   style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.Label("Plane:"),
                    dcc.RadioItems(
                        id='plane-selector',
                        options=[
                            {'label': 'XY', 'value': 'XY'},
                            {'label': 'YZ', 'value': 'YZ'},
                            {'label': 'XZ', 'value': 'XZ'}
                        ],
                        value='XY',
                        inline=True
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Show Total Field:"),
                    dcc.Checklist(
                        id='show-total',
                        options=[{'label': 'Total Field', 'value': 'total'}],
                        value=[]
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Label("Slice:", id='slice-label'),
                html.Div([
                    html.Button('⏯️', id='play-pause-btn', n_clicks=0, 
                               style={'margin-right': '10px', 'fontSize': '20px'}),
                    dcc.Slider(
                        id='slice-slider',
                        min=0,
                        max=nz - 1,
                        value=nz // 2,
                        marks={i: str(i) for i in range(0, nz, max(1, nz//5))},
                        step=1,
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode='drag'  # This enables live updating while dragging
                    )
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'margin': '20px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),
            
            html.Div([
                html.Label("Animation Speed:"),
                dcc.Slider(
                    id='speed-slider',
                    min=50,
                    max=1000,
                    value=200,
                    marks={50: 'Fast', 200: 'Medium', 500: 'Slow', 1000: 'Very Slow'},
                    step=50,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'margin': '20px', 'width': '60%', 'margin-left': 'auto', 'margin-right': 'auto'}),
            
            html.Div([
                html.Label("Colormap:"),
                dcc.RadioItems(
                    id='colormap-selector',
                    options=[
                        {'label': 'Inferno', 'value': 'inferno'},
                        {'label': 'Magma', 'value': 'magma'},
                        {'label': 'Viridis', 'value': 'viridis'},
                        {'label': 'Plasma', 'value': 'plasma'}
                    ],
                    value='inferno',
                    inline=True
                )
            ], style={'margin': '20px'}),
            
            dcc.Graph(id='field-plot', style={'height': '80vh'}),
            
            # Interval component for animation
            dcc.Interval(
                id='animation-interval',
                interval=200,  # milliseconds
                n_intervals=0,
                disabled=True  # Start disabled
            ),
            
            # Store component to track play state
            dcc.Store(id='play-state', data={'playing': False, 'direction': 1})
        ])
        
        # Callback for play/pause button and animation control
        @app.callback(
            [Output('animation-interval', 'disabled'),
             Output('animation-interval', 'interval'),
             Output('play-state', 'data')],
            [Input('play-pause-btn', 'n_clicks'),
             Input('speed-slider', 'value')],
            [dash.dependencies.State('play-state', 'data')]
        )
        def control_animation(n_clicks, speed, play_state):
            if n_clicks > 0:
                # Toggle play state
                playing = not play_state['playing']
                return not playing, speed, {'playing': playing, 'direction': play_state.get('direction', 1)}
            return True, speed, play_state
        
        # Callback for automatic slice advancement
        @app.callback(
            Output('slice-slider', 'value'),
            [Input('animation-interval', 'n_intervals')],
            [dash.dependencies.State('slice-slider', 'value'),
             dash.dependencies.State('slice-slider', 'max'),
             dash.dependencies.State('play-state', 'data')]
        )
        def update_slice_auto(n_intervals, current_slice, max_slice, play_state):
            if not play_state['playing']:
                return current_slice
            
            direction = play_state.get('direction', 1)
            next_slice = current_slice + direction
            
            # Handle looping and direction reversal
            if next_slice > max_slice:
                next_slice = 0  # Loop back to beginning
            elif next_slice < 0:
                next_slice = max_slice  # Loop to end
            
            return next_slice
        
        @app.callback(
            [Output('field-plot', 'figure'),
             Output('slice-label', 'children'),
             Output('slice-slider', 'max')],
            [Input('plane-selector', 'value'),
             Input('slice-slider', 'value'),
             Input('show-total', 'value'),
             Input('colormap-selector', 'value')]
        )
        def update_plot(plane, slice_idx, show_total_list, colormap):
            # Update slider max based on plane
            if plane == 'XY':
                max_slice = nz - 1
                slice_label = f"Z Slice: {slice_idx}"
            elif plane == 'YZ':
                max_slice = nx - 1
                slice_label = f"X Slice: {slice_idx}"
            else:  # XZ
                max_slice = ny - 1
                slice_label = f"Y Slice: {slice_idx}"
            
            # Adjust slice_idx if it exceeds new max
            slice_idx = min(slice_idx, max_slice)
            
            # Select data based on total field checkbox
            data = E_total_magnitude_cpu if 'total' in show_total_list else E_magnitude_cpu
            
            # Create mesh and surface data based on plane
            if plane == 'XY':
                x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
                z = np.full_like(x, slice_idx)
                surfacecolor = data[:, :, slice_idx]
                scene_labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}
            elif plane == 'YZ':
                y, z = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
                x = np.full_like(y, slice_idx)
                surfacecolor = data[slice_idx, :, :]
                scene_labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}
            else:  # XZ
                x, z = np.meshgrid(np.arange(nx), np.arange(nz), indexing='ij')
                y = np.full_like(x, slice_idx)
                surfacecolor = data[:, slice_idx, :]
                scene_labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}
            
            # Create figure
            fig = go.Figure()
            
            # Add bounding box
            fig.add_trace(go.Scatter3d(
                x=[0, nx, nx, 0, 0, 0, nx, nx, 0, 0, nx, nx, 0, 0, nx, nx],
                y=[0, 0, ny, ny, 0, 0, 0, ny, ny, 0, 0, ny, ny, 0, 0, ny],
                z=[0, 0, 0, 0, 0, nz, nz, nz, nz, 0, 0, 0, 0, nz, nz, nz],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False,
                hoverinfo='skip',
                name='Grid'
            ))
            
            # Add surface plot
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=surfacecolor,
                colorscale=colormap,
                showscale=True,
                colorbar=dict(title='|E| Total' if 'total' in show_total_list else '|E|'),
                name='Field'
            ))
            
            # Update layout
            fig.update_layout(
                scene=dict(
                    aspectmode='data',
                    xaxis_title=scene_labels['x'],
                    zaxis_title=scene_labels['z'],
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                title=f"{plane} Plane - Slice {slice_idx}",
                margin=dict(l=0, r=0, b=0, t=50)
            )
            
            return fig, slice_label, max_slice
        
        # Run the app
        print(f"\nStarting Dash web app...")
        print(f"Field visualization will open at: http://127.0.0.1:8050")
        print(f"Grid size: {nx} x {ny} x {nz}")
        print(f"Iterations: {iterations}, Final norm: {final_norm:.6e}")
        
        app.run(debug=False, port=8050)
        
        return app
        
    except ImportError:
        print("Error: Dash is not installed. Install with: pip install dash")
        return None
    except Exception as e:
        print(f"Error creating Dash visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main interactive loop"""
    print("="*60)
    print("    Interactive AI-Driven DDA Problem Solver")
    print("="*60)
    
    # Load AI model
    model = load_ai_model()
    if model is None:
        print("Cannot continue without model. Please check MODEL_PATH.")
        return
    
    while True:
        print("\n" + "="*60)
        print("Starting new problem setup...")
        
        # Collect shapes
        shapes = []
        shape_count = 0
        
        while True:
            shape_count += 1
            print(f"\n--- Shape {shape_count} ---")
            shape = create_shape()
            shapes.append(shape)
            
            print(f"\nShape {shape_count} created:")
            print(f"  Type: {shape['type']}")
            print(f"  Center: {shape['center']}")
            print(f"  Refractive index: {shape['n']}")
            
            if len(shapes) >= 1:
                add_another = get_choice_input("\nAdd another shape?", ['yes', 'no'], default='no')
                if add_another == 'no':
                    break
        
        # Get wave parameters
        k_direction, E_polarization, wavelength = get_wave_parameters()
        
        # Get simulation parameters (now includes wavelength-based calculation)
        lattice_spacing, save_filename = get_simulation_parameters(shapes, wavelength)
        
        print(f"\n--- Problem Summary ---")
        print(f"Number of shapes: {len(shapes)}")
        print(f"Wave direction: {k_direction}")
        print(f"Polarization: {E_polarization}")
        print(f"Wavelength: {wavelength}")
        print(f"Lattice spacing: {lattice_spacing}")
        
        # Build problem
        print("\nBuilding problem...")
        problem = build_problem_from_shapes(shapes, k_direction, E_polarization, wavelength, lattice_spacing)
        
        if problem is None:
            print("Failed to build problem. Please try again.")
            continue
        
        print(f"Problem built successfully!")
        print(f"Grid size: {problem['grid_size']}")
        print(f"Alpha array shape: {problem['alpha_array'].shape}")
        
        # Make AI prediction
        print("\nMaking AI prediction...")
        prediction = predict_with_ai(model, problem)
        
        if not np.any(np.isnan(prediction)):
            multiplier = complex(prediction[0], prediction[1])
            print(f"\n*** AI PREDICTION ***")
            print(f"Optimal multiplier: {multiplier.real:.6f} + {multiplier.imag:.6f}j")
            print(f"Magnitude: {abs(multiplier):.6f}")
            print(f"Phase: {np.angle(multiplier):.6f} radians ({np.degrees(np.angle(multiplier)):.2f} degrees)")
            
            # Save results if requested
            if save_filename:
                save_results_to_file(save_filename, problem, prediction, shapes)
            
            # Ask if user wants to run solver and visualize
            run_solver_choice = get_choice_input("Run DDA solver and show field visualization?", ['yes', 'no'], default='yes')
            
            if run_solver_choice == 'yes':
                print("\nRunning DDA solver...")
                try:
                    # Calculate 3x expansion of grid dimensions
                    grid_size = problem['grid_size']
                    x_exp = grid_size[0] * 3
                    y_exp = grid_size[1] * 3
                    z_exp = grid_size[2] * 3
                    
                    # Run solver using existing function
                    solver_result = run_solver(
                        grid_size=problem['grid_size'],
                        k=problem['k'],
                        alpha_array=problem['alpha_array'],
                        lattice_spacing=problem['lattice_spacing'],
                        x_expansion=x_exp, y_expansion=y_exp, z_expansion=z_exp,  # 3x grid dimensions
                        refract_mult=multiplier,
                        E_inc=problem['E_inc'],
                        precon=True,
                        max_iter=10000,
                        double=False,
                        ratio=100
                    )
                    
                    # run_solver returns 3 values: iterations, polarization_array, mean_norm
                    iterations, polarization_array, final_norm = solver_result
                    
                    if iterations is not None and iterations > 0 and iterations < 10000:
                        print(f"Solver converged in {iterations} iterations")
                        print(f"Final residual: {final_norm:.3e}")
                        
                        # Update saved file with solver results if requested
                        if save_filename:
                            print("Updating saved file with solver results...")
                            save_results_to_file(save_filename, problem, prediction, shapes, solver_result)
                        
                        # Create interaction matrix for field calculation (no expansion needed)
                        print("Calculating interaction matrix for field visualization...")
                        inv_alpha, mask, interaction_matrix, preconditioner = prepare_interaction_matrices(
                            problem['grid_size'], problem['k'], multiplier,
                            problem['grid_size'][0], problem['grid_size'][1], problem['grid_size'][2],  # Use original grid size
                            problem['alpha_array'], problem['lattice_spacing'],
                            reduced=True, is_2d=False, cutoff=False, double_precision=False
                        )
                        
                        # Create visualization
                        print("Creating field visualization...")
                        fig = create_field_visualization(problem, solver_result, interaction_matrix)
                        
                    else:
                        print(f"Solver failed to converge properly")
                        if iterations is not None:
                            if iterations <= 0:
                                print(f"Solver failed with error code: {iterations}")
                            elif iterations >= 10000:
                                print(f"Reached maximum iterations: {iterations}")
                            print(f"Final norm: {final_norm:.3e}")
                        
                        # Still save the results even if failed
                        if save_filename:
                            print("Updating saved file with solver results...")
                            save_results_to_file(save_filename, problem, prediction, shapes, solver_result)
                        
                except Exception as e:
                    print(f"Error running solver: {e}")
                    import traceback
                    traceback.print_exc()
                    
        else:
            print("AI prediction failed!")
            if save_filename:
                save_results_to_file(save_filename, problem, prediction, shapes)
        
        # Ask if user wants to continue
        continue_choice = get_choice_input("\nSolve another problem?", ['yes', 'no'], default='no')
        if continue_choice == 'no':
            break
    
    print("\nThank you for using the AI-Driven DDA Problem Solver!")

if __name__ == "__main__":
    main()