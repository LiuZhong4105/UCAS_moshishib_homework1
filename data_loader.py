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
    Load MNIST dataset from sklearn or generate synthetic data
    
    Args:
        test_size: Proportion of dataset to include in test split
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
        try:
            print("Loading MNIST dataset from OpenML...")
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1, parser='auto')
            X, y = mnist.data, mnist.target.astype(int)
            
            # Convert to numpy arrays if needed
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
            
            # Sample if requested
            if sample_size is not None and sample_size < len(X):
                indices = np.random.RandomState(random_state).choice(len(X), sample_size, replace=False)
                X = X[indices]
                y = y[indices]
            
            # Normalize pixel values to [0, 1]
            X = X / 255.0
        except Exception as e:
            print(f"Failed to load MNIST: {e}")
            print("Falling back to synthetic data...")
            X, y = generate_synthetic_data(
                n_samples=sample_size if sample_size else 10000,
                random_state=random_state
            )
    
    # Split into train and test sets
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
