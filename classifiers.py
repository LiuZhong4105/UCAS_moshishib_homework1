"""
Classification algorithms: QDF with RDA/MQDF and KNN with KDTree
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class QDF:
    """
    Quadratic Discriminant Function with regularization
    Supports RDA (Regularized Discriminant Analysis) and MQDF (Modified Quadratic Discriminant Function)
    """
    
    def __init__(self, reg_type='rda', alpha=0.5, beta=0.5, k_mqdf=None):
        """
        Args:
            reg_type: 'rda' for Regularized DA or 'mqdf' for Modified QDF
            alpha: RDA parameter for covariance shrinkage (0 to 1)
            beta: RDA parameter for pooling (0 to 1)
            k_mqdf: Number of principal components for MQDF (if None, uses full covariance)
        """
        self.reg_type = reg_type
        self.alpha = alpha
        self.beta = beta
        self.k_mqdf = k_mqdf
        self.classes = None
        self.means = {}
        self.covariances = {}
        self.priors = {}
    
    def fit(self, X, y):
        """
        Fit QDF classifier
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
        """
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        # First pass: compute means and priors
        class_covariances = {}
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.priors[c] = len(X_c) / len(X)
            
            # Compute class covariance
            X_centered = X_c - self.means[c]
            cov_c = np.dot(X_centered.T, X_centered) / len(X_c)
            class_covariances[c] = cov_c
        
        # Second pass: apply regularization
        for c in self.classes:
            cov_c = class_covariances[c]
            
            if self.reg_type == 'rda':
                # RDA regularization
                # Pooled covariance
                cov_pooled = np.zeros((n_features, n_features))
                for cls in self.classes:
                    X_cls = X[y == cls]
                    X_cls_centered = X_cls - self.means[cls]
                    cov_pooled += np.dot(X_cls_centered.T, X_cls_centered)
                cov_pooled /= len(X)
                
                # Shrink towards pooled covariance
                cov_c = self.alpha * cov_c + (1 - self.alpha) * cov_pooled
                
                # Shrink towards identity (diagonal)
                cov_c = self.beta * cov_c + (1 - self.beta) * np.trace(cov_c) / n_features * np.eye(n_features)
                
            elif self.reg_type == 'mqdf':
                # MQDF: use only top k eigenvalues/eigenvectors
                if self.k_mqdf is not None and self.k_mqdf < n_features:
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_c)
                    idx = np.argsort(eigenvalues)[::-1]
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]
                    
                    # Keep top k components
                    eigenvalues_k = eigenvalues[:self.k_mqdf]
                    eigenvectors_k = eigenvectors[:, :self.k_mqdf]
                    
                    # Reconstruct with reduced dimensionality
                    # Add small constant to remaining eigenvalues
                    delta = np.mean(eigenvalues[self.k_mqdf:]) if self.k_mqdf < len(eigenvalues) else 1e-6
                    cov_c = eigenvectors_k @ np.diag(eigenvalues_k) @ eigenvectors_k.T + delta * np.eye(n_features)
            
            # Add small regularization for numerical stability
            self.covariances[c] = cov_c + 1e-6 * np.eye(n_features)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: Test data of shape (n_samples, n_features)
        
        Returns:
            Predicted labels of shape (n_samples,)
        """
        scores = np.zeros((len(X), len(self.classes)))
        
        for i, c in enumerate(self.classes):
            # Compute discriminant function for class c
            diff = X - self.means[c]
            
            # Compute inverse and determinant
            cov_inv = np.linalg.inv(self.covariances[c])
            cov_det = np.linalg.det(self.covariances[c])
            
            # Quadratic discriminant function
            mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
            scores[:, i] = -0.5 * (np.log(cov_det) + mahalanobis) + np.log(self.priors[c])
        
        # Return class with highest score
        return self.classes[np.argmax(scores, axis=1)]
    
    def score(self, X, y):
        """
        Compute accuracy
        
        Args:
            X: Test data
            y: True labels
        
        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class KNN:
    """KNN classifier with KDTree acceleration"""
    
    def __init__(self, n_neighbors=5, algorithm='kd_tree'):
        """
        Args:
            n_neighbors: Number of neighbors to use
            algorithm: Algorithm to use ('kd_tree', 'ball_tree', 'brute')
        """
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    
    def fit(self, X, y):
        """
        Fit KNN classifier
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
        """
        self.knn.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: Test data of shape (n_samples, n_features)
        
        Returns:
            Predicted labels of shape (n_samples,)
        """
        return self.knn.predict(X)
    
    def score(self, X, y):
        """
        Compute accuracy
        
        Args:
            X: Test data
            y: True labels
        
        Returns:
            Accuracy score
        """
        return self.knn.score(X, y)
