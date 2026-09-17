import os
import sys

# Windows fix for systemprofile PATH permission issue
path_dirs = os.environ.get("PATH", "").split(os.pathsep)
os.environ["PATH"] = os.pathsep.join([d for d in path_dirs if "systemprofile" not in d.lower()])

import navis
import matplotlib.pyplot as plt

def plot_and_save_neurons():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Loading fruit fly brain neuron dataset...")
    nl = navis.example_neurons()
    print(f"Loaded {len(nl)} neurons for 3D visualization.")

    # 1. Plot in 3D using Matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#FF5733', '#33FF57', '#3357FF', '#F033FF', '#FF33A8']
    
    for i, neuron in enumerate(nl):
        color = colors[i % len(colors)]
        # Extract x, y, z coordinates
        nodes = neuron.nodes
        ax.scatter(nodes['x'], nodes['y'], nodes['z'], s=1, color=color, label=f"{neuron.name} (ID: {neuron.id})", alpha=0.6)

    ax.set_title("3D Reconstruction of Fruit Fly Brain Projection Neurons", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    ax.legend(loc='upper right', fontsize=8)

    output_img = os.path.join(data_dir, "fruitfly_3d_neurons.png")
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()

    print(f"Saved 3D visualization image -> {output_img}")

if __name__ == "__main__":
    plot_and_save_neurons()
