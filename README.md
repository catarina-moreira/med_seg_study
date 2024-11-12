# Research on Medical Image Segmentation Libraries

This repository focuses on studying state-of-the-art medical image segmentation libraries, with a goal of integrating them into a Python pipeline for automated segmentation and 3D reconstruction from CT and MRI scans.

## Libraries Reviewed

### 1. TotalSegmentator

**TotalSegmentator** is an open-source deep learning tool specifically designed for automatic and robust segmentation of anatomical structures in medical imaging, such as CT and MRI scans. Developed by the Research and Analysis department at University Hospital Basel, it leverages the powerful nnU-Net framework to identify and delineate over 100 anatomical structures, including organs, bones, muscles, and vessels.

- **Original repository**: [TotalSegmentator GitHub](https://github.com/wasserth/TotalSegmentator)

In this study, I used TotalSegmentator in a Python pipeline to extract segmentation masks for various organs from CT scan data. The predicted masks were compared to ground truth masks from the TotalSegmentator dataset, and both sets of masks were reconstructed into 3D models using the VTK library and the Marching Cubes algorithm. Below, you can find examples of a segmentation prediction for a slice and its corresponding 3D reconstruction of the colon.

### 2. Marching Cubes

The **Marching Cubes** algorithm is a widely-used computer graphics algorithm for reconstructing 3D surfaces from volumetric data. It is particularly effective for creating polygonal mesh representations of medical scans or other 3D data and is frequently applied in fields like medical imaging, geology, and scientific visualization.

#### Key Concepts

- **Input**: A 3D grid of scalar values (e.g., intensity levels across a volume).
- **Isosurface Extraction**: Identifies surfaces within the 3D grid that match a specified scalar value, known as the **isosurface** (for example, tissue density in CT scans).
- **Cube Marching**: The 3D grid is divided into cubes, and each cube is evaluated to determine which edges intersect the isosurface, forming polygons that approximate the surface.

#### Algorithm Overview

1. **Cube Evaluation**: Each cube in the grid is evaluated based on whether each corner’s scalar value is above or below the isosurface level. This creates a unique binary pattern indicating the cube’s configuration.
   
2. **Case Table**: Each binary pattern corresponds to a specific set of triangles that represent the surface within that cube. A lookup table of 256 possible configurations tells the algorithm where to place vertices and edges.

3. **Triangle Generation**: The algorithm then creates triangles according to the lookup table, approximating the surface within each cube.

4. **Mesh Output**: These triangles are combined into a continuous mesh, forming a 3D object that represents the surface.

