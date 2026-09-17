import os
import sys

# Windows fix for systemprofile PATH permission issue
path_dirs = os.environ.get("PATH", "").split(os.pathsep)
os.environ["PATH"] = os.pathsep.join([d for d in path_dirs if "systemprofile" not in d.lower()])

import navis
import pandas as pd

def download_fruit_fly_sample_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    print("=== Downloading Fruit Fly Brain Sample Connectome Dataset ===")

    # 1. Download sample fruit fly brain neuron reconstructions
    print("Downloading fruit fly neuron reconstructions...")
    nl = navis.example_neurons()
    print(f"Successfully downloaded {len(nl)} fruit fly neurons!")

    # 2. Extract summary metrics & metadata
    summary_df = nl.summary()
    summary_path = os.path.join(data_dir, "fruitfly_neurons_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved neuron summary metrics -> {summary_path}")

    # 3. Export 3D SWC skeleton files for visualization / modeling
    swc_dir = os.path.join(data_dir, "swc_skeletons")
    os.makedirs(swc_dir, exist_ok=True)
    print("Exporting 3D SWC skeleton files...")
    for neuron in nl:
        file_name = f"{neuron.name}_{neuron.id}.swc".replace('/', '_')
        file_path = os.path.join(swc_dir, file_name)
        navis.write_swc(neuron, file_path)
        print(f"  - Saved SWC skeleton: {file_path}")

    # 4. Generate & save node table summary (nodes, coordinates, connections)
    nodes_df = pd.concat([n.nodes for n in nl], ignore_index=True)
    nodes_path = os.path.join(data_dir, "fruitfly_nodes_dataset.csv")
    nodes_df.to_csv(nodes_path, index=False)
    print(f"Saved complete node dataset ({len(nodes_df)} nodes) -> {nodes_path}")

    print("\nSUCCESS: All dataset files downloaded successfully into the 'data/' directory!")

if __name__ == "__main__":
    download_fruit_fly_sample_data()
