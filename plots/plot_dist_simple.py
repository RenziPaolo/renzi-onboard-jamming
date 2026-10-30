import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import argparse
import os

def read_json_and_plot(json_path, output_dir):
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(json_path)

    # Extract the relevant entries
    entries = {
        'none0.0': data.get('none0.0', {}),
        'gaussian_noise1.0': data.get('gaussian_noise1.0', {}),
        'RFI1.0': data.get('RFI1.0', {})
    }

    # Prepare data for spikes
    spikes_labels = []
    spikes_avg = []
    spikes_dev = []

    # Prepare data for energy
    energy_labels = []
    energy_avg = []
    energy_dev = []

    col = 0
    
    for label, entry in entries.items():
        print(f"Processing entry: {label}")
        print(entry)
        # Check for spikes data
        if 'spikes_avg' in entry and 'spikes_dev' in entry:
            spikes_labels.append(label)
            spikes_avg.append(entry['spikes_avg'])
            spikes_dev.append(entry['spikes_dev'])
            col += 1
        print('energy_avg' in entry,'energy_dev' in entry)
        # Check for energy data
        if 'energy_avg' in entry and 'energy_dev' in entry:
            energy_labels.append(label)
            energy_avg.append(entry['energy_avg'])
            energy_dev.append(entry['energy_dev'])
            col += 1

    if not spikes_avg and not energy_avg:
        print("No valid data to plot.")
        return
    col = col//3
    print(col)
    # Create subplots
    fig, axes = plt.subplots(1, col, figsize=(14, 6))
    if col == 1:
        axes = [axes]
    
    # Plot spikes distributions
    if spikes_avg:
        x_spikes = np.linspace(
            min(spikes_avg) - 3*max(spikes_dev),
            max(spikes_avg) + 3*max(spikes_dev),
            1000
        )
        for avg, dev, label in zip(spikes_avg, spikes_dev, spikes_labels):
            y = norm.pdf(x_spikes, avg, dev)
            axes[1].plot(x_spikes, y, label=label)
        axes[1].set_title('Gaussian Distribution of Spikes Average')
        axes[1].set_xlabel('Spikes Average')
        axes[1].set_ylabel('Probability Density')
        axes[1].legend()
        axes[1].grid(True)

    # Plot energy distributions
    if energy_avg:
        x_energy = np.linspace(
            min(energy_avg) - 3*max(energy_dev),
            max(energy_avg) + 3*max(energy_dev),
            1000
        )
        for avg, dev, label in zip(energy_avg, energy_dev, energy_labels):
            y = norm.pdf(x_energy, avg, dev)
            axes[0].plot(x_energy, y, label=label)
        axes[0].set_title('Gaussian Distribution of energy Average')
        axes[0].set_xlabel('energy Average')
        axes[0].set_ylabel('Probability Density')
        axes[0].legend()
        axes[0].grid(True)

    # Adjust layout
    plt.tight_layout()
    output_path = output_dir + 'dist_plot'


    # Save the plots
    if spikes_avg:
        fig.savefig(output_path + f'_spikes.png', dpi=200, bbox_inches='tight')
    if energy_avg:
        fig.savefig(output_path + f'_energy.png', dpi=200, bbox_inches='tight')

    # Show the plot
    plt.show()


parser = argparse.ArgumentParser(description='plot distributions from JSON results')
parser.add_argument('-json_path', type=str, help='root dir of MNIST dataset')
args = parser.parse_args()

# Example usage
json_path = args.json_path   # Replace with your JSON file path
output_dir = os.path.dirname(json_path)+ '/' # Replace with your desired output directory
read_json_and_plot(json_path, output_dir)
