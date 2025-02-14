import os
import tempfile

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import imageio.v2 as imageio
import shutil

import numpy as np
import nibabel as nib
import pydicom

import pyvista as pv

from IPython.display import display, Image, HTML

import ipywidgets as widgets

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from utils.data_processing import load_ct_scan, pre_process_ct_scan


def generate_gif(file_path, output_filepath = "ct.gif"):

    ct_data = pre_process_ct_scan( file_path )
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Normalize the CT scan data for visualization
    ct_data_normalized = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min()) * 255
    ct_data_normalized = ct_data_normalized.astype(np.uint8)

    # Define the output GIF path folder
    output_gif_folder = os.path.join("outputs", "gifs")

    # if output_gif_path doesn't exist, create it
    if not os.path.exists(output_gif_folder):
        os.makedirs(output_gif_folder)

    # Define the output GIF path file
    output_gif_path = os.path.join("outputs", "gifs", output_filepath)

    # Create a temporary directory for storing images
    temp_dir = tempfile.mkdtemp()

    # Create a GIF animation
    images = []
    for i in range(ct_data_normalized.shape[2]):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(ct_data_normalized[:, :, i], cmap="gray")
        ax.axis('off')
        plt.title(f"Slice {i + 1} of {ct_data_normalized.shape[2]}")

        # Save image to an in-memory file object
        plt.tight_layout()
        image_file = os.path.join(temp_dir, f"{filename}_slice_{i}.png")
        plt.savefig(image_file)
        plt.close()

        # Append image to the images list for GIF creation
        images.append(imageio.imread(image_file))

    # Create the GIF
    os.makedirs(output_gif_folder, exist_ok=True)    
    imageio.mimsave(output_gif_path, images, duration=0.1)

    # Remove the temporary directory and its contents after GIF creation
    shutil.rmtree(temp_dir)

    #print(f"GIF saved at: {output_gif_path}")
    return output_gif_path


def show_gif(file_path, width=500, height=500):
    return Image(filename=file_path, width=width, height=height)


# Function to load and display a DICOM file
def show_dicom_image(file_path):
    # Load the DICOM file
    dicom_image = pydicom.dcmread(file_path)

    # Get the pixel array from the DICOM file
    pixel_array = dicom_image.pixel_array

    # Plot the image using matplotlib
    plt.imshow(pixel_array, cmap=plt.cm.gray)
    plt.axis('off')  # Turn off axis labels
    plt.title(f"{os.path.basename(file_path)}")
    plt.show()


# Function to iterate through all .dcm files in a directory
def visualize_dicom_files(directory_path):
    # List all files in the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                show_dicom_image(file_path)


# Function to load and visualize all slices of a NIfTI (.nii.gz) file
def show_nii_slices(file_path, n_cols=5, cmap="bone"):
    # Load the NIfTI file
    img = nib.load(file_path)

    # Get the data as a numpy array
    img_data = img.get_fdata()

    # Number of slices in the z-axis
    num_slices = img_data.shape[2]

    # Calculate the number of rows needed
    n_rows = int(np.ceil(num_slices / n_cols))

    # Create subplots with the calculated rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    fig.subplots_adjust(hspace=0.3)

    # Plot each slice in the appropriate subplot
    for i in range(num_slices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        ax.imshow(img_data[:, :, i], cmap=cmap)
        ax.set_title(f'Slice {i+1}/{num_slices}')
        ax.axis('off')  # Hide the axes

    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])

    plt.show()


def show_nii_slices_with_mask(ct_path, mask_path, ct_map="bone", mask_map="Reds", n_cols=5, alpha=0.5):
    """
    Displays all slices of a CT scan with the segmentation mask overlay.

    Parameters:
    - ct_path: Path to the CT scan NIfTI file (.nii.gz)
    - mask_path: Path to the segmentation mask NIfTI file (.nii.gz)
    - ct_map: Colormap for the CT scan (default is "bone")
    - mask_map: Colormap for the segmentation mask (default is "Reds")
    - n_cols: Number of columns in the display grid (default is 5)
    - alpha: Transparency of the segmentation overlay (default is 0.5)
    """

    # Load the NIfTI files
    ct_img = nib.load(ct_path)
    mask_img = nib.load(mask_path)

    # Convert to numpy arrays
    ct_data = ct_img.get_fdata()
    mask_data = mask_img.get_fdata()

    # Ensure both images have the same dimensions
    assert ct_data.shape == mask_data.shape, "CT scan and segmentation mask must have the same dimensions!"

    # Get the number of slices
    num_slices = ct_data.shape[2]

    # Calculate the number of rows needed
    n_rows = int(np.ceil(num_slices / n_cols))

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    fig.subplots_adjust(hspace=0.3)

    # Create colormaps
    cmap_ct = plt.get_cmap(ct_map)  # CT scan colormap
    cmap_mask = plt.get_cmap(mask_map)  # Segmentation mask colormap

    # Normalize colormap for the mask
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Plot each slice with the segmentation mask overlay
    for i in range(num_slices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Normalize CT scan for better visualization
        ct_slice = ct_data[:, :, i]
        ct_norm = (ct_slice - np.min(ct_slice)) / (np.max(ct_slice) - np.min(ct_slice))

        # Get the mask slice and ensure it’s binary (0s and 1s)
        mask_slice = mask_data[:, :, i]
        mask_binary = mask_slice > 0  # Convert mask to boolean (True = segmentation, False = background)

        # Apply the "bone" colormap to the CT scan
        ct_colored = cmap_ct(ct_norm)[:, :, :3]  # Convert grayscale to RGB

        # Convert mask into RGBA with proper transparency
        mask_colored = cmap_mask(mask_binary.astype(float))  # Apply colormap
        mask_colored[..., 3] = mask_binary * alpha  # Apply transparency to mask only

        # Display CT scan
        ax.imshow(ct_colored)

        # Overlay segmentation mask with transparency
        ax.imshow(mask_colored)

        # Set title and remove axes
        ax.set_title(f'Slice {i+1}/{num_slices}')
        ax.axis('off')

    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])

    plt.show()



# Function to visualize DICOM, segmentation, and overlap side by side
def visualize_side_by_side(dicom_slice, seg_slice, overlay, cmap = "gray", isPrediction = False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display DICOM slice
    axes[0].imshow(dicom_slice, cmap='bone')
    axes[0].set_title('Image Slice')
    axes[0].axis('off')

    # Display segmentation slice
    axes[1].imshow(seg_slice, cmap=cmap)
    if isPrediction:
        axes[1].set_title('Predicted Segmented Slice')
    else:
        axes[1].set_title('Groundtruth Segmented Slice')
    axes[1].axis('off')

    # Display overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.show()




def visualize_mesh(mesh, file_path, id, isPrediction = False, smoothing_iter = 50, relaxation_factor = 0.1, mesh_color = '#FFCC99', opacity = 0.7, background_color = "black" ):
    
    filename = os.path.basename(file_path)
        
    mesh_smooth = mesh.smooth(n_iter=smoothing_iter, relaxation_factor=relaxation_factor)

    # Set up the PyVista plotter
    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(mesh_smooth, 
                        color=mesh_color, 
                        show_edges=False, 
                        opacity=opacity,
                        smooth_shading=True,
                        ambient=0,
                        )
    plotter.add_axes()
    plotter.set_background(background_color)

    HOME = os.getcwd()

    output_path = os.path.join(HOME, "outputs", "3d_reconstruction", id)
    
    # create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if isPrediction:
        output_path = os.path.join(output_path, "_pred.html")
    else:
        output_path = os.path.join(output_path, filename + ".html")
    plotter.export_html(output_path)  
    print(f"Saving mesh to {output_path}")


    plotter.show(jupyter_backend='trame')

    return output_path

def visualize_gifs_side_by_side(gif1, gif2, width=400):

    # HTML code to display the GIFs side-by-side
    html_code = f"""
    <div style="display: flex; justify-content: space-around;">
        <div>
            <img src="{gif1}" style="width: {width}px; height: auto;" />
        </div>
        <div>
            <img src="{gif2}" style="width: {width}px; height: auto;" />
        </div>
    </div>
    """

    # Display the HTML
    display(HTML(html_code))

def visualize_gif(gif, width=400):
    """
    Displays a single GIF in a Jupyter Notebook.

    Parameters:
        gif (str): Path or URL to the GIF.
        width (int): Width of the GIF in pixels. Default is 400.
    """

    # replace backslashes with forward slashes
    gif = gif.replace("\\", "/")

    # HTML code to display the single GIF
    html_code = f"""
    <div style="display: flex; justify-content: center;">
        <img src="{gif}" style="width: {width}px; height: auto;" loop />
    </div>
    """
    # Display the HTML
    display(HTML(html_code))


def visualize_ct_scan(ct_filepath):
    # Load the .nii file
    nii_data = nib.load(ct_filepath)
    image_data = nii_data.get_fdata()
    
    # Cache slices
    slices_cache = [image_data[:, :, i] for i in range(image_data.shape[2])]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(slices_cache[0], cmap="gray")
    ax.axis("off")
    
    # Update function
    def update(slice_index):
        img.set_data(slices_cache[slice_index])
        ax.set_title(f"Slice {slice_index}")
        fig.canvas.draw_idle()
    
    # Slider and Textbox
    slice_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=image_data.shape[2] - 1,
        step=1,
        description="Slice",
        continuous_update=False  # Faster updates
    )
    slice_textbox = widgets.IntText(
        value=0,
        description="Mask Index:"
    )
    
    # Sync slider and textbox
    def sync_widgets(change):
        if change["owner"] == slice_slider:
            slice_textbox.value = change["new"]
        elif change["owner"] == slice_textbox:
            if 0 <= change["new"] < image_data.shape[2]:
                slice_slider.value = change["new"]
    
    slice_slider.observe(sync_widgets, names="value")
    slice_textbox.observe(sync_widgets, names="value")
    
    # Interactive display
    widgets.interactive(update, slice_index=slice_slider)
    display(widgets.HBox([slice_slider, slice_textbox]))
    update(0)  # Initialize with the first slice