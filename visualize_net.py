import numpy as np
import matplotlib.pyplot as plt



def normalize_to_range_weight(v):
    v_min, v_max = np.min(v), np.max(v)
    return 2 * (v - v_min) / (v_max - v_min) - 1

def normalize_to_range_activations(v):
    v_min, v_max = np.min(v), np.max(v)
    return  (v - v_min) / (v_max - v_min)


def visualize_neuron(w):   
    
    # Reshape to 28x28 image
    image = w.reshape((28, 28))
    
    # Plot the image with a diverging colormap
    plt.imshow(image, cmap='seismic', interpolation='nearest')

    # Add a color bar
    plt.colorbar()

    # Remove axis labels for clarity
    plt.axis('off')

    # Show the image
    plt.show()


