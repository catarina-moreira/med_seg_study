import os
import numpy as np
import nibabel as nib
import pydicom
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_ct_scan(file_path):

    if not file_path:
        print("No .nii.gz file found in the folder.")
        return
    # Load the CT scan file
    ct_scan = nib.load(file_path)

    # Get the image data as a NumPy array
    ct_data = ct_scan.get_fdata()

    # Check if data is 3D
    if len(ct_data.shape) != 3:
        print("Expected a 3D CT scan, but got a different shape.")
        return

    # normalize the data
    ct_data = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min())

    return ct_data


# Function to load and display a specific slice from a .nii.gz file
def load_nii_slice(nii_file_path, slice_index):
    # Load the NIfTI file
    nii_image = nib.load(nii_file_path)

    # Get the data as a numpy array
    img_data = nii_image.get_fdata()

    # Check if slice_index is within the range of available slices
    if slice_index < 1 or slice_index > img_data.shape[2]:
        raise ValueError(f"Slice index {slice_index} is out of range for this image with {img_data.shape[2]} slices.")

    # Extract the specified slice (note: subtract 1 because slice_index is 1-based)
    slice_data = img_data[:, :, slice_index - 1]
    return slice_data

# Function to load and display the 350th DICOM file
def load_dicom_slice(dicom_directory, slice_index):
    # List all files in the directory
    dicom_files = [os.path.join(dicom_directory, f) for f in os.listdir(dicom_directory) if f.endswith('.dcm')]
    dicom_files.sort()  # Sort to ensure consistent ordering

    # Load the DICOM file corresponding to the slice index (350th)
    dicom_file = dicom_files[slice_index - 1]
    dicom_image = pydicom.dcmread(dicom_file)

    # Get the pixel array from the DICOM file
    pixel_array = dicom_image.pixel_array
    return pixel_array

# Function to load the 350th slice from a NIfTI file (segmentation mask)
def load_nii_segmentation(nii_file_path, slice_index):
    # Load the NIfTI file
    img = nib.load(nii_file_path)

    # Get the data as a numpy array
    img_data = img.get_fdata()

    # Return the slice corresponding to the slice index (350th)
    seg_slice = img_data[:, :, slice_index - 1]
    return seg_slice

# Function to overlap DICOM and segmentation mask
def overlay_images_backup(dicom_slice, seg_slice):
    # Normalize both images to [0, 1] for visualization purposes
    dicom_norm = (dicom_slice - np.min(dicom_slice)) / (np.max(dicom_slice) - np.min(dicom_slice))
    seg_mask = seg_slice.astype(bool)  # Convert segmentation to boolean mask

    # Create an RGB image for overlaying: DICOM as grayscale, segmentation as red
    overlay = np.zeros((dicom_norm.shape[0], dicom_norm.shape[1], 3))
    overlay[:, :, 0] = dicom_norm  # Red channel for the segmentation
    overlay[:, :, 1] = dicom_norm  # Green channel for the DICOM
    overlay[:, :, 2] = dicom_norm  # Blue channel for the DICOM
    overlay[seg_mask, 0] = 1  # Make segmentation region red


def overlay_images(dicom_slice, seg_slice, alpha=0.5, mask_cmap="Reds", ct_cmap="bone"):
    """
    Overlays a segmentation mask on top of a DICOM (CT) slice using different colormaps.

    Parameters:
    - dicom_slice: 2D numpy array representing the CT slice.
    - seg_slice: 2D numpy array representing the segmentation mask.
    - alpha: Transparency level of the segmentation mask (default is 0.5).
    - mask_cmap: Colormap for the segmentation mask (default is "Reds").
    - ct_cmap: Colormap for the CT scan (default is "bone").

    Returns:
    - overlay: 3D numpy array (RGB image) with the segmentation mask overlaid.
    """

    # Normalize CT image for better visualization
    dicom_norm = (dicom_slice - np.min(dicom_slice)) / (np.max(dicom_slice) - np.min(dicom_slice))

    # Apply the "bone" colormap to the CT image
    ct_colormap = plt.get_cmap(ct_cmap)  # Get the colormap for CT scan
    ct_colored = ct_colormap(dicom_norm)[:, :, :3]  # Convert to RGB (ignore alpha)

    # Convert segmentation mask to binary
    seg_mask = seg_slice > 0  # Ensures mask is boolean (True where segmentation exists, False otherwise)

    # Get the colormap for the mask
    mask_colormap = plt.get_cmap(mask_cmap)  # Get the colormap for the segmentation mask
    mask_colored = mask_colormap(seg_mask.astype(float))  # Convert mask to RGBA

    # Overlay the segmentation mask onto the CT scan using alpha blending
    overlay = ct_colored.copy()  # Start with the CT image
    overlay[seg_mask] = (1 - alpha) * overlay[seg_mask] + alpha * mask_colored[seg_mask, :3]

    return overlay



def pre_process_ct_scan( ct_scan_filepath ):

    # Load the NIfTI file
    nii_file = nib.load(ct_scan_filepath)
    ct_data = nii_file.get_fdata()  # Convert to numpy array

    print(f"Shape: {ct_data.shape}, Data Type: {ct_data.dtype}")

    # CT scans often have anisotropic voxel spacing (e.g., spacing between slices may differ from the in-plane resolution). 
    # Resampling ensures isotropic voxel spacing or a consistent resolution.

    # Define the original and desired spacing
    original_spacing = nii_file.header.get_zooms()  # e.g., (0.5, 0.5, 2.0)
    desired_spacing = (1.0, 1.0, 1.0)

    # Compute resampling factor
    resize_factor = [o / d for o, d in zip(original_spacing, desired_spacing)]
    new_shape = [int(s * f) for s, f in zip(ct_data.shape, resize_factor)]

    # Resample the data
    ct_data_resampled = ndimage.zoom(ct_data, resize_factor, order=1)  # Linear interpolation
    return ct_data_resampled


