import mnist_loader

# Cached MNIST dataset (global variable)
_cached_mnist = None  

def mnist_import():
    """Load MNIST dataset only if it's not already loaded."""
    global _cached_mnist

    # ✅ If dataset is already loaded, return it
    if _cached_mnist is not None:
        print("Using MNIST dataset from memory.")
        return _cached_mnist  # ✅ Returns (training_data, validation_data, test_data)

    # ✅ If dataset is not in memory, load it from file
    print("Loading MNIST dataset...")
    _cached_mnist = mnist_loader.load_data_wrapper()
    return _cached_mnist  # ✅ Now returns 3 variables correctly
