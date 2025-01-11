import subprocess
import sys

# List of pip packages to install
pip_packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "nibabel",
    "pydicom",
    "matplotlib",
    "jupyter_contrib_nbextensions",
    "ipympl",
    "plotly",
    "imageio",
    "scikit-image",
    "pyvista==0.44.2",
    "ipywidgets",
    "itkwidgets",
    "trame==3.2.7",
    "vtk",
    "trame-vtk",
    "trame-vuetify",
    "trame_jupyter_extension",
]

# List of conda packages to install
conda_packages = [
    "trame",
    "trame-vtk"
]

def install_pip_packages(package_list):
    for package in package_list:
        try:
            print(f"Installing {package} using pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}\n")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package} using pip. Error: {e}\n")
        except Exception as e:
            print(f"An unexpected error occurred while installing {package} using pip. Error: {e}\n")

def install_conda_packages(package_list):
    for package in package_list:
        try:
            print(f"Installing {package} using conda...")
            subprocess.check_call(["conda", "install", "-c", "conda-forge", package])
            print(f"Successfully installed {package}\n")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package} using conda. Error: {e}\n")
        except Exception as e:
            print(f"An unexpected error occurred while installing {package} using conda. Error: {e}\n")

if __name__ == "__main__":
    print("Starting package installation...\n")

    # Install pip packages
    install_pip_packages(pip_packages)

    # Install conda packages
    install_conda_packages(conda_packages)

    print("All packages have been processed.")
    

