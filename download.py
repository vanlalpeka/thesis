import openml
import os

# Function to download and save a dataset locally
def download_and_save_dataset(dataset_id, local_path):
    # Download the dataset from OpenML
    dataset = openml.datasets.get_dataset(dataset_id)
    
    # Get the dataset features and labels
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    # Save features and labels locally
    local_data_path = os.path.join(local_path, 'ccfraud_X.csv')
    local_labels_path = os.path.join(local_path, 'ccfraud_y.csv')
    
    X.to_csv(local_data_path, index=False)
    y.to_csv(local_labels_path, index=False)
    
    print(f"Dataset downloaded and saved locally at: {local_path}")

# Specify the OpenML dataset ID
dataset_id = 42175  # You can find the dataset ID on the OpenML website

# Specify the local path to save the dataset
local_save_path = 'dataset'

# Create the local directory if it does not exist
os.makedirs(local_save_path, exist_ok=True)

# Download and save the dataset locally
download_and_save_dataset(dataset_id, local_save_path)
