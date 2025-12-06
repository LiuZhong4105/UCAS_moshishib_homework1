"""
Experiment framework for comparing classifiers with different dimensionality reduction
"""
import numpy as np
import time
from data_loader import load_mnist, standardize_data
from dimensionality_reduction import PCA, LDA
from classifiers import QDF, KNN


def run_experiments(dimensions=[10, 20, 30, 50, 100, 200], sample_size=10000):
    """
    Run experiments comparing classifiers with different dimensionality reduction methods
    
    Args:
        dimensions: List of dimensions to test
        sample_size: Number of samples to use (for faster experiments)
    
    Returns:
        Dictionary containing experiment results
    """
    print("=" * 80)
    print("Starting MNIST Classification Experiments")
    print("=" * 80)
    
    # Load data
    X_train, X_test, y_train, y_test = load_mnist(sample_size=sample_size)
    
    # Standardize data
    X_train, X_test = standardize_data(X_train, X_test)
    
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
        
        # Limit LDA dimensions to at most n_classes - 1
        lda_dim = min(dim, 9)  # MNIST has 10 classes, so max 9 LDA components
        
        # PCA reduction
        print(f"\nApplying PCA with {dim} components...")
        pca = PCA(n_components=dim)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # LDA reduction
        print(f"Applying LDA with {lda_dim} components...")
        lda = LDA(n_components=lda_dim)
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)
        
        # Test PCA + QDF (RDA)
        print("\nPCA + QDF (RDA)...")
        start_time = time.time()
        qdf_rda = QDF(reg_type='rda', alpha=0.5, beta=0.5)
        qdf_rda.fit(X_train_pca, y_train)
        acc_pca_qdf_rda = qdf_rda.score(X_test_pca, y_test)
        time_pca_qdf_rda = time.time() - start_time
        results['pca_qdf_rda'].append(acc_pca_qdf_rda)
        results['pca_qdf_rda_time'].append(time_pca_qdf_rda)
        print(f"  Accuracy: {acc_pca_qdf_rda:.4f}, Time: {time_pca_qdf_rda:.2f}s")
        
        # Test PCA + QDF (MQDF)
        print("PCA + QDF (MQDF)...")
        start_time = time.time()
        k_mqdf = max(5, dim // 2)  # Use half of dimensions for MQDF
        qdf_mqdf = QDF(reg_type='mqdf', k_mqdf=k_mqdf)
        qdf_mqdf.fit(X_train_pca, y_train)
        acc_pca_qdf_mqdf = qdf_mqdf.score(X_test_pca, y_test)
        time_pca_qdf_mqdf = time.time() - start_time
        results['pca_qdf_mqdf'].append(acc_pca_qdf_mqdf)
        results['pca_qdf_mqdf_time'].append(time_pca_qdf_mqdf)
        print(f"  Accuracy: {acc_pca_qdf_mqdf:.4f}, Time: {time_pca_qdf_mqdf:.2f}s")
        
        # Test PCA + KNN
        print("PCA + KNN...")
        start_time = time.time()
        knn = KNN(n_neighbors=5, algorithm='kd_tree')
        knn.fit(X_train_pca, y_train)
        acc_pca_knn = knn.score(X_test_pca, y_test)
        time_pca_knn = time.time() - start_time
        results['pca_knn'].append(acc_pca_knn)
        results['pca_knn_time'].append(time_pca_knn)
        print(f"  Accuracy: {acc_pca_knn:.4f}, Time: {time_pca_knn:.2f}s")
        
        # Test LDA + QDF (RDA)
        print("LDA + QDF (RDA)...")
        start_time = time.time()
        qdf_rda_lda = QDF(reg_type='rda', alpha=0.5, beta=0.5)
        qdf_rda_lda.fit(X_train_lda, y_train)
        acc_lda_qdf_rda = qdf_rda_lda.score(X_test_lda, y_test)
        time_lda_qdf_rda = time.time() - start_time
        results['lda_qdf_rda'].append(acc_lda_qdf_rda)
        results['lda_qdf_rda_time'].append(time_lda_qdf_rda)
        print(f"  Accuracy: {acc_lda_qdf_rda:.4f}, Time: {time_lda_qdf_rda:.2f}s")
        
        # Test LDA + QDF (MQDF)
        print("LDA + QDF (MQDF)...")
        start_time = time.time()
        k_mqdf_lda = max(3, lda_dim // 2)
        qdf_mqdf_lda = QDF(reg_type='mqdf', k_mqdf=k_mqdf_lda)
        qdf_mqdf_lda.fit(X_train_lda, y_train)
        acc_lda_qdf_mqdf = qdf_mqdf_lda.score(X_test_lda, y_test)
        time_lda_qdf_mqdf = time.time() - start_time
        results['lda_qdf_mqdf'].append(acc_lda_qdf_mqdf)
        results['lda_qdf_mqdf_time'].append(time_lda_qdf_mqdf)
        print(f"  Accuracy: {acc_lda_qdf_mqdf:.4f}, Time: {time_lda_qdf_mqdf:.2f}s")
        
        # Test LDA + KNN
        print("LDA + KNN...")
        start_time = time.time()
        knn_lda = KNN(n_neighbors=5, algorithm='kd_tree')
        knn_lda.fit(X_train_lda, y_train)
        acc_lda_knn = knn_lda.score(X_test_lda, y_test)
        time_lda_knn = time.time() - start_time
        results['lda_knn'].append(acc_lda_knn)
        results['lda_knn_time'].append(time_lda_knn)
        print(f"  Accuracy: {acc_lda_knn:.4f}, Time: {time_lda_knn:.2f}s")
    
    print("\n" + "=" * 80)
    print("Experiments completed!")
    print("=" * 80)
    
    return results


def print_results_table(results):
    """Print results in a formatted table"""
    print("\n" + "=" * 120)
    print("EXPERIMENTAL RESULTS - ACCURACY")
    print("=" * 120)
    
    # Header
    header = f"{'Dimensions':<12}"
    header += f"{'PCA+QDF(RDA)':<15}{'PCA+QDF(MQDF)':<16}{'PCA+KNN':<12}"
    header += f"{'LDA+QDF(RDA)':<15}{'LDA+QDF(MQDF)':<16}{'LDA+KNN':<12}"
    print(header)
    print("-" * 120)
    
    # Data rows
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
    
    print("\n" + "=" * 120)
    print("EXPERIMENTAL RESULTS - TRAINING TIME (seconds)")
    print("=" * 120)
    
    # Header
    header = f"{'Dimensions':<12}"
    header += f"{'PCA+QDF(RDA)':<15}{'PCA+QDF(MQDF)':<16}{'PCA+KNN':<12}"
    header += f"{'LDA+QDF(RDA)':<15}{'LDA+QDF(MQDF)':<16}{'LDA+KNN':<12}"
    print(header)
    print("-" * 120)
    
    # Data rows
    for i, dim in enumerate(results['dimensions']):
        row = f"{dim:<12}"
        row += f"{results['pca_qdf_rda_time'][i]:<15.2f}"
        row += f"{results['pca_qdf_mqdf_time'][i]:<16.2f}"
        row += f"{results['pca_knn_time'][i]:<12.2f}"
        row += f"{results['lda_qdf_rda_time'][i]:<15.2f}"
        row += f"{results['lda_qdf_mqdf_time'][i]:<16.2f}"
        row += f"{results['lda_knn_time'][i]:<12.2f}"
        print(row)
    
    print("=" * 120)
