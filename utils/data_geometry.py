import os

from skimage import measure
import numpy as np
import pyvista as pv


from utils.data_processing import load_ct_scan

def generate_3d_reconstruction(file_path, id = "ID0", isPrediction = False):
    
    # Load the segmented mask file
    ct_data = load_ct_scan(file_path)

    filename = os.path.basename(file_path)

    # get home directory
    HOME = os.getcwd()
    
    # Extract the 3D surface using the marching cubes algorithm
    vertices, faces, _, _ = measure.marching_cubes(ct_data, level=0.7)

    # Reformat the faces array to match the expected format for PyVista
    # PyVista expects a flat array where each face is prefixed by the number of points (e.g., 3 for triangles)
    faces_formatted = np.hstack([[3] + list(face) for face in faces])

    # Create a PyVista mesh for visualization
    mesh = pv.PolyData(vertices, faces_formatted)
    
    output_path = os.path.join(HOME, "mesh", id)
    
    # create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if isPrediction:
        output_path = os.path.join(output_path, "_pred.vtk")
    else:
        output_path = os.path.join(output_path, filename + ".vtk")

    print(f"Saving mesh to {output_path}")
    
    mesh.save(output_path)

    return mesh
