import os
import sys

# Windows fix for systemprofile PATH permission issue
path_dirs = os.environ.get("PATH", "").split(os.pathsep)
os.environ["PATH"] = os.pathsep.join([d for d in path_dirs if "systemprofile" not in d.lower()])

import json
import pandas as pd
from neuprint import Client, fetch_neurons, fetch_adjacencies, fetch_synapses, NeuronCriteria as NC

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as f:
        return json.load(f)

def initialize_client(config):
    token = config.get("token")
    if not token or token == "YOUR_NEUPRINT_AUTH_TOKEN_HERE":
        print("[WARNING] Please update 'config.json' with your actual NeuPrint Auth Token.")
        print("Obtain your token at: https://neuprint.janelia.org/ (Account Settings)")
        return None
    
    server = config.get("server", "neuprint.janelia.org")
    dataset = config.get("dataset", "hemibrain:v1.2.1")
    
    print(f"Connecting to {server} (Dataset: {dataset})...")
    client = Client(server, dataset=dataset, token=token)
    print("Successfully connected to NeuPrint!")
    return client

def fetch_and_save_kc_neurons(output_dir="data"):
    """
    Example Experiment Function:
    Downloads Kenyon Cell (KC) neurons dataset from Fruit Fly Mushroom Body
    and saves to local CSV files.
    """
    config = load_config()
    client = initialize_client(config)
    if not client:
        return

    os.makedirs(output_dir, exist_ok=True)
    print("\nFetching Kenyon Cell (KC) neurons...")
    neurons_df, roi_counts_df = fetch_neurons(NC(type="KC.*"))
    
    print(f"Retrieved {len(neurons_df)} neurons.")
    
    neurons_file = os.path.join(output_dir, "kc_neurons.csv")
    neurons_df.to_csv(neurons_file, index=False)
    print(f"Saved neuron dataset to: {neurons_file}")

    rois_file = os.path.join(output_dir, "kc_roi_counts.csv")
    roi_counts_df.to_csv(rois_file, index=False)
    print(f"Saved ROI counts to: {rois_file}")

if __name__ == "__main__":
    print("=== NeuPrint Fruit Fly Connectomics Starter ===")
    config = load_config()
    print("Current Dataset Configuration:", config)
    
    # Run fetch if token is set
    if config.get("token") != "YOUR_NEUPRINT_AUTH_TOKEN_HERE":
        fetch_and_save_kc_neurons()
    else:
        print("\nNext step: Open 'config.json' and paste your NeuPrint Auth Token!")
