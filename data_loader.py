"""
Data loader for MNIST dataset
"""
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_mnist_from_file(data_dir='data'):
    """
    Load MNIST dataset from local files
    
    Args:
        data_dir: Directory containing MNIST files
    
    Returns:
        X, y: Features and labels
    """
    # This function would load from actual MNIST files
    # For now, we'll raise an error directing to use fetch_openml or synthetic data
    raise NotImplementedError(
        "Local file loading not implemented. "
        "Please ensure internet connection to download MNIST, "
        "or use generate_synthetic_data() for testing."
    )


def generate_synthetic_data(n_samples=10000, n_features=784, n_classes=10, random_state=42):
    """
    Generate synthetic data for testing when MNIST is not available
    
    Creates a more realistic and challenging synthetic dataset by:
    - Using smaller separation between class means
    - Adding higher within-class variance
    - Including sparse features (mimicking image data)
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of classes
        random_state: Random state for reproducibility
    
    Returns:
        X, y: Features and labels
    """
    np.random.seed(random_state)
    
    # Generate class-specific means with smaller separation
    # Reduced from 2 to 0.3 to create more overlap
    class_means = np.random.randn(n_classes, n_features) * 0.3
    
    # Generate samples
    samples_per_class = n_samples // n_classes
    X = []
    y = []
    
    for i in range(n_classes):
        # Generate samples with higher variance (multiplied by 2)
        # This creates more within-class variability
        class_samples = np.random.randn(samples_per_class, n_features) * 2 + class_means[i]
        
        # Add sparsity to mimic image data (many pixels are background)
        # Randomly zero out 70% of features
        mask = np.random.rand(samples_per_class, n_features) > 0.7
        class_samples = class_samples * mask
        
        X.append(class_samples)
        y.extend([i] * samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Normalize to [0, 1] like real MNIST
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    
    return X, y


def load_mnist(test_size=0.2, random_state=42, sample_size=None, use_synthetic=False):
    """
    Load MNIST dataset from TensorFlow/Keras or generate synthetic data
    
    Attempts to load MNIST from TensorFlow Keras datasets:
    - Training set: 60,000 samples
    - Test set: 10,000 samples
    - Image size: 28x28 pixels (784 features)
    
    Reference: https://www.tensorflow.org/datasets/catalog/mnist
    
    Args:
        test_size: Proportion of dataset to include in test split (used only when sampling)
        random_state: Random state for reproducibility
        sample_size: If provided, sample this many examples (for faster testing)
        use_synthetic: If True, generate synthetic data instead of downloading MNIST
    
    Returns:
        X_train, X_test, y_train, y_test: Training and test data
    """
    if use_synthetic:
        print("Generating synthetic data for testing...")
        X, y = generate_synthetic_data(
            n_samples=sample_size if sample_size else 10000,
            random_state=random_state
        )
    else:
        # Try to load from TensorFlow/Keras
        mnist_loaded = False
        
        try:
            print("Loading MNIST dataset from TensorFlow/Keras...")
            print("Reference: https://www.tensorflow.org/datasets/catalog/mnist")
            
            # Use keras.datasets which comes with TensorFlow
            # This loads MNIST from TensorFlow's official dataset repository
            from tensorflow import keras
            
            # Load MNIST dataset from Keras
            # train split: 60,000 samples, test split: 10,000 samples
            (X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.mnist.load_data()
            
            # Reshape images from (N, 28, 28) to (N, 784)
            X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
            X_test_full = X_test_full.reshape(X_test_full.shape[0], -1)
            
            # Normalize pixel values to [0, 1]
            X_train_full = X_train_full.astype(np.float32) / 255.0
            X_test_full = X_test_full.astype(np.float32) / 255.0
            
            # Convert labels to integers
            y_train_full = y_train_full.astype(int)
            y_test_full = y_test_full.astype(int)
            
            mnist_loaded = True
            
            print(f"Successfully loaded MNIST dataset!")
            print(f"Full training set: {X_train_full.shape[0]} samples")
            print(f"Full test set: {X_test_full.shape[0]} samples")
            
        except Exception as e:
            print(f"Failed to load MNIST from TensorFlow/Keras: {e}")
        
        # Try OpenML as a fallback
        if not mnist_loaded:
            try:
                print("Trying to load from OpenML as fallback...")
                from sklearn.datasets import fetch_openml
                mnist = fetch_openml('mnist_784', version=1, parser='auto')
                X, y = mnist.data, mnist.target.astype(int)
                
                # Convert to numpy arrays if needed
                if hasattr(X, 'values'):
                    X = X.values
                if hasattr(y, 'values'):
                    y = y.values
                
                # Normalize pixel values to [0, 1]
                X = X / 255.0
                
                # Split into standard train/test (60k/10k)
                X_train_full = X[:60000]
                y_train_full = y[:60000]
                X_test_full = X[60000:]
                y_test_full = y[60000:]
                
                mnist_loaded = True
                print(f"Successfully loaded MNIST from OpenML!")
                
            except Exception as e:
                print(f"Failed to load MNIST from OpenML: {e}")
        
        # If loading failed, use synthetic data
        if not mnist_loaded:
            print("Falling back to synthetic data...")
            X, y = generate_synthetic_data(
                n_samples=sample_size if sample_size else 10000,
                random_state=random_state
            )
        else:
            # Process the loaded MNIST data
            # Sample if requested and smaller than full dataset
            total_samples = len(X_train_full) + len(X_test_full)
            if sample_size is not None and sample_size < total_samples:
                # Combine train and test sets for sampling
                X = np.vstack([X_train_full, X_test_full])
                y = np.concatenate([y_train_full, y_test_full])
                
                # Sample the requested number
                indices = np.random.RandomState(random_state).choice(len(X), sample_size, replace=False)
                X = X[indices]
                y = y[indices]
                
                # Split into train and test sets with requested test_size ratio
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            else:
                # Use the predefined train/test split from MNIST (60k train / 10k test)
                X_train, y_train = X_train_full, y_train_full
                X_test, y_test = X_test_full, y_test_full
            
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            print(f"Feature dimension: {X_train.shape[1]}")
            
            return X_train, X_test, y_train, y_test
    
    # Split into train and test sets (for synthetic data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    """
    Standardize features to have zero mean and unit variance
    
    Args:
        X_train: Training data
        X_test: Test data
    
    Returns:
        X_train_scaled, X_test_scaled: Standardized data
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
