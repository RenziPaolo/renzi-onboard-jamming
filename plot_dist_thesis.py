import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

# Define the Gaussian function
def gaussian(x, avg, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - avg) / std) ** 2)



def read_json_and_plot(json_path, output_dir):
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(json_path)
    m = int(max(data["spikes_echo_avg"], data["spikes_RFI_avg"]))
    s = int(max(data["spikes_echo_dev"], data["spikes_RFI_dev"]))
    # Define the x range
    x = np.linspace(m - 3*s, m + 3*s, 1000)

    # Define the parameters for the three Gaussians: (mean, standard deviation)
    gaussians = [
        {"mu": data["spikes_echo_avg"], "sigma": data["spikes_echo_dev"], "label": "Echo"},
        {"mu": data["spikes_RFI_avg"], "sigma": data["spikes_RFI_dev"], "label": "RFI"},
    ]

    # Plot each Gaussian
    plt.figure(figsize=(10, 6))
    for g in gaussians:
        y = gaussian(x, g["mu"], g["sigma"])
        plt.plot(x, y, label=f"{g['label']} μ={g['mu']}, σ={g['sigma']}")

    # Add labels and legend
    plt.title("Plot of Three Gaussian Distributions")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(output_dir+"gaussians.png", format='png', dpi=200, bbox_inches='tight')

    # Show the plot (optional)
    plt.show()


parser = argparse.ArgumentParser(description='plot distributions from JSON results')
parser.add_argument('-json_path', type=str, help='root dir of MNIST dataset')
args = parser.parse_args()

# Example usage
json_path = args.json_path   # Replace with your JSON file path
output_dir = os.path.dirname(json_path)+ '/' # Replace with your desired output directory
read_json_and_plot(json_path, output_dir)
