import subprocess
import sys

# List of pip packages to install
pip_packages = [
	"torch",
	"torchvision", 
	"torchaudio",
	"git+https://github.com/wasserth/TotalSegmentator.git",
	"xmltodict",
	"acvl_utils==0.2",
	"fury",
    "acvl_utils",  
    "ace_tools" 
 
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

if __name__ == "__main__":
    print("Starting package installation...\n")

    # Install pip packages
    install_pip_packages(pip_packages)

    print("All packages have been processed.")
    

