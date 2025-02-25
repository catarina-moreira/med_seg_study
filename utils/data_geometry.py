import os

from skimage import measure
import numpy as np
import pyvista as pv

from utils.data_processing import load_ct_scan, pre_process_ct_scan

def generate_3d_reconstruction(file_path, id = "ID0", isPrediction = False, level=0.7):
    
    # Load the segmented mask file
    ct_data = pre_process_ct_scan( file_path )

    filename = os.path.basename(file_path)

    # get home directory
    HOME = os.getcwd()
    
    # Extract the 3D surface using the marching cubes algorithm
    vertices, faces, _, _ = measure.marching_cubes(ct_data, level=level)

    # Reformat the faces array to match the expected format for PyVista
    # PyVista expects a flat array where each face is prefixed by the number of points (e.g., 3 for triangles)
    faces_formatted = np.hstack([[3] + list(face) for face in faces])

    # Create a PyVista mesh for visualization
    mesh = pv.PolyData(vertices, faces_formatted)
    
    output_path = os.path.join(HOME, "outputs", "mesh", id)
    
    # create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if isPrediction:
        output_path = os.path.join(output_path, filename + "_march_cubes" + "_pred.vtk")
    else:
        output_path = os.path.join(output_path, filename + "_march_cubes" + ".vtk")

    print(f"Saving mesh to {output_path}")
    
    mesh.save(output_path)

    return mesh


def generate_3d_reconstruction_flying_edges(file_path, id="ID0", isPrediction=False, level=0.7):
    """
    Generate a 3D reconstruction using the Flying Edges algorithm.

    Args:
        file_path (str): Path to the CT scan file.
        id (str): Identifier for the output file.
        isPrediction (bool): Flag for whether the input is a prediction.
        level (float): Isosurface value for the reconstruction.

    Returns:
        pyvista.PolyData: The generated 3D mesh.
    """
    # Load and preprocess the segmented mask file
    ct_data = pre_process_ct_scan(file_path)

    filename = os.path.basename(file_path)

    # Get the home directory
    HOME = os.getcwd()

    # Create a PyVista grid from the CT data
    grid = pv.ImageData()
    grid.dimensions = ct_data.shape
    grid.spacing = (1.0, 1.0, 1.0)  # Set the voxel spacing (adjust as needed)
    grid.origin = (0.0, 0.0, 0.0)   # Set the grid origin
    grid.point_data["values"] = ct_data.flatten(order="F")  # Add the CT data to the grid

    # Extract the isosurface using the Flying Edges algorithm
    mesh = grid.contour(isosurfaces=[level], scalars="values")

    # Define the output path
    output_path = os.path.join(HOME, "outputs", "mesh", id)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if isPrediction:
        output_file = os.path.join(output_path, filename + "_flying_edges" + "_pred.obj")
    else:
        output_file = os.path.join(output_path, filename + "_flying_edges" + ".obj")

    print(f"Saving mesh to {output_file}")

    # Save the mesh to an OBJ file
    mesh.save(output_file)

    return mesh
