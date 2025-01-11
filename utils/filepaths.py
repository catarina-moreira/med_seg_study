import os
import platform

# TOTAL_SEGMENTATOR_DATASET
TOTAL_SEGM_LINUX = full_path = os.path.join("/", "media", "catarina", "SHARGE DISK", "Totalsegmentator_dataset_V201")
TOTAL_SEGM_MAC = os.path.expanduser(os.path.join("/", "Volumes", "Untitled", "Totalsegmentator_dataset_v201"))
TOTAL_SEGM_WINDOWS = os.path.expanduser(os.path.join("D:", "\\", "Datasets", "Totalsegmentator_dataset_v201"))

os_name = platform.system()

if os_name == "Windows":
    TOTAL_SEGM_PATH = TOTAL_SEGM_WINDOWS
elif os_name == "Darwin":
    TOTAL_SEGM_PATH = TOTAL_SEGM_MAC
elif os_name == "Linux":
    TOTAL_SEGM_PATH = TOTAL_SEGM_LINUX
else:
    print(f"Unknown operating system: {os_name}")


# TEST CT SCAN
