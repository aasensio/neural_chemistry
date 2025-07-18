import os
import requests

# Try to import tqdm for a progress bar. If not available, provide a fallback.
from tqdm import tqdm

def download_file_if_not_exists(file_info, download_dir="data", verbose=False):
    """
    Checks if a file exists locally. If not, downloads it from the specified URL.

    Args:
        file_info (dict): A dictionary containing 'name' (local filename)
                          and 'url' (download URL).
        download_dir (str): The directory where files should be stored.
    """
    file_name = file_info['name']
    file_url = file_info['url']
    local_file_path = os.path.join(download_dir, file_name)

    # 1. Check for directory existence and create if necessary
    if not os.path.exists(download_dir):
        print(f"Creating download directory: {download_dir}")
        try:
            # os.makedirs creates all intermediate directories and handles permissions
            # mode=0o755 gives owner rwx, group rx, others rx
            os.makedirs(download_dir, mode=0o755, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {download_dir}: {e}")
            print("Please ensure you have appropriate write permissions for the specified directory.")
            return False

    # 2. Check if the file already exists
    if os.path.exists(local_file_path):
        if verbose:
            print(f"File '{file_name}' already exists at '{local_file_path}'. Skipping download.")
        return True
    else:
        print(f"File '{file_name}' not found. Attempting to download from '{file_url}'...")
        try:
            # Stream the download to handle large files efficiently
            response = requests.get(file_url, stream=True, timeout=30)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            total_size = int(response.headers.get('content-length', 0))

            with open(local_file_path, 'wb') as f:
                if tqdm and total_size > 0:
                    # Use tqdm for a progress bar
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name, ncols=80) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    # Fallback if tqdm is not available or size is unknown
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                    print(f"Downloaded '{file_name}' to '{local_file_path}'.")

            print(f"Successfully downloaded '{file_name}' to '{local_file_path}'.")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Error downloading '{file_name}': {e}")
            print("Please check your internet connection and the URL.")
            # Clean up partially downloaded file
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            return False
        except Exception as e:
            print(f"An unexpected error occurred while downloading '{file_name}': {e}")
            # Clean up partially downloaded file
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            return False

def ensure_files_exist(files_to_check, base_download_dir="data", verbose=False):
    """
    Ensures all specified files exist by downloading them if necessary.

    Args:
        files_to_check (list): A list of dictionaries, each with 'name' and 'url'.
        base_download_dir (str): The base directory to store all files.

    Returns:
        bool: True if all files are successfully ensured, False otherwise.
    """
    all_successful = True    
    for file_info in files_to_check:
        if not download_file_if_not_exists(file_info, base_download_dir, verbose):
            all_successful = False
            print(f"Failed to ensure '{file_info['name']}'. Aborting further checks for this run.")
            break # Stop if one file fails to download    
    return all_successful


# weights
# https://cloud.iac.es/index.php/s/W8Qgw8Yy95BqstR


# 2025-07-14-19_09_54.best.pth
# https://cloud.iac.es/index.php/s/2TqdbogWqcfppCB

# 2025-07-17-15_23_49.best.pth
# https://cloud.iac.es/index.php/s/R2wNkNMZMT3dTtG