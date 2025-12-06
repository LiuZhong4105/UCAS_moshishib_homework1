"""
Dimensionality reduction methods: PCA and LDA
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA


class PCA:
    """Principal Component Analysis for dimensionality reduction"""
    
    def __init__(self, n_components):
        """
        Args:
            n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
    
    def fit(self, X):
        """
        Fit PCA on training data
        
        Args:
            X: Training data of shape (n_samples, n_features)
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top n_components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        
        return self
    
    def transform(self, X):
        """
        Transform data to lower dimensional space
        
        Args:
            X: Data of shape (n_samples, n_features)
        
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)


class LDA:
    """Linear Discriminant Analysis for dimensionality reduction"""
    
    def __init__(self, n_components):
        """
        Args:
            n_components: Number of discriminant components to keep
        """
        self.n_components = n_components
        self.lda = SklearnLDA(n_components=n_components)
    
    def fit(self, X, y):
        """
        Fit LDA on training data
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
        """
        self.lda.fit(X, y)
        return self
    
    def transform(self, X):
        """
        Transform data to lower dimensional space
        
        Args:
            X: Data of shape (n_samples, n_features)
        
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        return self.lda.transform(X)
    
    def fit_transform(self, X, y):
        """Fit and transform in one step"""
        return self.lda.fit_transform(X, y)
