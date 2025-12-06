"""
Demo script to show project functionality with synthetic data
This simulates realistic performance variations
"""
import numpy as np
import time
from data_loader import generate_synthetic_data, standardize_data
from dimensionality_reduction import PCA, LDA
from classifiers import QDF, KNN
from sklearn.model_selection import train_test_split
from visualizations import create_all_plots


def generate_realistic_synthetic_mnist(n_samples=10000, random_state=42):
    """
    Generate more realistic synthetic data that mimics MNIST characteristics
    with some class overlap for more realistic accuracy scores
    """
    np.random.seed(random_state)
    
    n_features = 784
    n_classes = 10
    samples_per_class = n_samples // n_classes
    
    # Create class prototypes with more overlap
    class_centers = []
    for i in range(n_classes):
        center = np.random.randn(n_features) * 0.5
        # Add some structure to make it more digit-like
        # Simulate different "patterns" for different digits
        center[i*70:(i+1)*70] += 2.0  # Specific region for each class
        class_centers.append(center)
    
    X = []
    y = []
    
    for i in range(n_classes):
        # Generate samples with noise
        noise_level = 1.5  # Higher noise for more realistic difficulty
        class_samples = np.random.randn(samples_per_class, n_features) * noise_level + class_centers[i]
        
        # Add some outliers
        outlier_ratio = 0.05
        n_outliers = int(samples_per_class * outlier_ratio)
        outlier_indices = np.random.choice(samples_per_class, n_outliers, replace=False)
        class_samples[outlier_indices] += np.random.randn(n_outliers, n_features) * 2.0
        
        X.append(class_samples)
        y.extend([i] * samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Normalize
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    
    return X, y


def run_demo_experiments():
    """Run demo experiments with realistic synthetic data"""
    print("=" * 80)
    print("MNIST Classification Demo - Using Realistic Synthetic Data")
    print("=" * 80)
    print("\nNote: Real MNIST data would give similar patterns but different absolute values")
    
    # Generate data
    X, y = generate_realistic_synthetic_mnist(n_samples=5000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    dimensions = [5, 10, 20, 30, 50, 100]
    
    results = {
        'dimensions': dimensions,
        'pca_qdf_rda': [],
        'pca_qdf_mqdf': [],
        'pca_knn': [],
        'lda_qdf_rda': [],
        'lda_qdf_mqdf': [],
        'lda_knn': [],
        'pca_qdf_rda_time': [],
        'pca_qdf_mqdf_time': [],
        'pca_knn_time': [],
        'lda_qdf_rda_time': [],
        'lda_qdf_mqdf_time': [],
        'lda_knn_time': [],
    }
    
    for dim in dimensions:
        print(f"\n{'=' * 80}")
        print(f"Testing with {dim} dimensions")
        print(f"{'=' * 80}")
        
        lda_dim = min(dim, 9)
        
        # PCA
        print(f"\nApplying PCA with {dim} components...")
        pca = PCA(n_components=dim)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # LDA
        print(f"Applying LDA with {lda_dim} components...")
        lda = LDA(n_components=lda_dim)
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)
        
        # PCA + QDF (RDA)
        print("\nPCA + QDF (RDA)...")
        start = time.time()
        qdf_rda = QDF(reg_type='rda', alpha=0.5, beta=0.5)
        qdf_rda.fit(X_train_pca, y_train)
        acc = qdf_rda.score(X_test_pca, y_test)
        elapsed = time.time() - start
        results['pca_qdf_rda'].append(acc)
        results['pca_qdf_rda_time'].append(elapsed)
        print(f"  Accuracy: {acc:.4f}, Time: {elapsed:.2f}s")
        
        # PCA + QDF (MQDF)
        print("PCA + QDF (MQDF)...")
        start = time.time()
        qdf_mqdf = QDF(reg_type='mqdf', k_mqdf=max(3, dim // 2))
        qdf_mqdf.fit(X_train_pca, y_train)
        acc = qdf_mqdf.score(X_test_pca, y_test)
        elapsed = time.time() - start
        results['pca_qdf_mqdf'].append(acc)
        results['pca_qdf_mqdf_time'].append(elapsed)
        print(f"  Accuracy: {acc:.4f}, Time: {elapsed:.2f}s")
        
        # PCA + KNN
        print("PCA + KNN...")
        start = time.time()
        knn = KNN(n_neighbors=5)
        knn.fit(X_train_pca, y_train)
        acc = knn.score(X_test_pca, y_test)
        elapsed = time.time() - start
        results['pca_knn'].append(acc)
        results['pca_knn_time'].append(elapsed)
        print(f"  Accuracy: {acc:.4f}, Time: {elapsed:.2f}s")
        
        # LDA + QDF (RDA)
        print("LDA + QDF (RDA)...")
        start = time.time()
        qdf_rda_lda = QDF(reg_type='rda', alpha=0.5, beta=0.5)
        qdf_rda_lda.fit(X_train_lda, y_train)
        acc = qdf_rda_lda.score(X_test_lda, y_test)
        elapsed = time.time() - start
        results['lda_qdf_rda'].append(acc)
        results['lda_qdf_rda_time'].append(elapsed)
        print(f"  Accuracy: {acc:.4f}, Time: {elapsed:.2f}s")
        
        # LDA + QDF (MQDF)
        print("LDA + QDF (MQDF)...")
        start = time.time()
        qdf_mqdf_lda = QDF(reg_type='mqdf', k_mqdf=max(2, lda_dim // 2))
        qdf_mqdf_lda.fit(X_train_lda, y_train)
        acc = qdf_mqdf_lda.score(X_test_lda, y_test)
        elapsed = time.time() - start
        results['lda_qdf_mqdf'].append(acc)
        results['lda_qdf_mqdf_time'].append(elapsed)
        print(f"  Accuracy: {acc:.4f}, Time: {elapsed:.2f}s")
        
        # LDA + KNN
        print("LDA + KNN...")
        start = time.time()
        knn_lda = KNN(n_neighbors=5)
        knn_lda.fit(X_train_lda, y_train)
        acc = knn_lda.score(X_test_lda, y_test)
        elapsed = time.time() - start
        results['lda_knn'].append(acc)
        results['lda_knn_time'].append(elapsed)
        print(f"  Accuracy: {acc:.4f}, Time: {elapsed:.2f}s")
    
    # Print results table
    print("\n" + "=" * 120)
    print("EXPERIMENTAL RESULTS - ACCURACY")
    print("=" * 120)
    header = f"{'Dimensions':<12}"
    header += f"{'PCA+QDF(RDA)':<15}{'PCA+QDF(MQDF)':<16}{'PCA+KNN':<12}"
    header += f"{'LDA+QDF(RDA)':<15}{'LDA+QDF(MQDF)':<16}{'LDA+KNN':<12}"
    print(header)
    print("-" * 120)
    
    for i, dim in enumerate(results['dimensions']):
        row = f"{dim:<12}"
        row += f"{results['pca_qdf_rda'][i]:<15.4f}"
        row += f"{results['pca_qdf_mqdf'][i]:<16.4f}"
        row += f"{results['pca_knn'][i]:<12.4f}"
        row += f"{results['lda_qdf_rda'][i]:<15.4f}"
        row += f"{results['lda_qdf_mqdf'][i]:<16.4f}"
        row += f"{results['lda_knn'][i]:<12.4f}"
        print(row)
    
    print("=" * 120)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_all_plots(results)
    print("Demo completed!")
    
    return results


if __name__ == '__main__':
    run_demo_experiments()
