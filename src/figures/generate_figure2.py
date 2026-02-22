import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Sample function to generate climate maps visualization
def generate_climate_map(data):
    # Define the colormap
    cmap = plt.get_cmap("viridis")
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a heatmap
    c = ax.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', norm=mcolors.Normalize(vmin=np.min(data), vmax=np.max(data)))
    
    # Add a color bar
    plt.colorbar(c, ax=ax)
    
    # Set title and labels
    ax.set_title("Climate Map Visualization")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    plt.show()

# Dummy data for testing
if __name__ == "__main__":
    # Generate some random data for demonstration
    climate_data = np.random.rand(10, 10) * 100  # Replace with actual climate data
    generate_climate_map(climate_data)