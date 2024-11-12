import os
import numpy as np
import nibabel as nib
import pydicom



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
def overlay_images(dicom_slice, seg_slice):
    # Normalize both images to [0, 1] for visualization purposes
    dicom_norm = (dicom_slice - np.min(dicom_slice)) / (np.max(dicom_slice) - np.min(dicom_slice))
    seg_mask = seg_slice.astype(bool)  # Convert segmentation to boolean mask

    # Create an RGB image for overlaying: DICOM as grayscale, segmentation as red
    overlay = np.zeros((dicom_norm.shape[0], dicom_norm.shape[1], 3))
    overlay[:, :, 0] = dicom_norm  # Red channel for the segmentation
    overlay[:, :, 1] = dicom_norm  # Green channel for the DICOM
    overlay[:, :, 2] = dicom_norm  # Blue channel for the DICOM
    overlay[seg_mask, 0] = 1  # Make segmentation region red

    return overlay