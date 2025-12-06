"""
Visualization functions for experimental results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_accuracy_comparison(results, save_path='accuracy_comparison.png'):
    """
    Plot accuracy comparison across different dimensions
    
    Args:
        results: Dictionary containing experiment results
        save_path: Path to save the figure
    """
    plt.figure(figsize=(14, 8))
    
    dimensions = results['dimensions']
    
    # Plot PCA results
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, results['pca_qdf_rda'], 'o-', label='QDF (RDA)', linewidth=2, markersize=8)
    plt.plot(dimensions, results['pca_qdf_mqdf'], 's-', label='QDF (MQDF)', linewidth=2, markersize=8)
    plt.plot(dimensions, results['pca_knn'], '^-', label='KNN', linewidth=2, markersize=8)
    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('PCA + Classifiers', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.5, 1.0])
    
    # Plot LDA results
    plt.subplot(1, 2, 2)
    # For LDA, dimensions are limited to 9
    lda_dims = [min(d, 9) for d in dimensions]
    plt.plot(lda_dims, results['lda_qdf_rda'], 'o-', label='QDF (RDA)', linewidth=2, markersize=8)
    plt.plot(lda_dims, results['lda_qdf_mqdf'], 's-', label='QDF (MQDF)', linewidth=2, markersize=8)
    plt.plot(lda_dims, results['lda_knn'], '^-', label='KNN', linewidth=2, markersize=8)
    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('LDA + Classifiers', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy comparison plot saved to {save_path}")
    plt.close()


def plot_time_comparison(results, save_path='time_comparison.png'):
    """
    Plot training time comparison across different dimensions
    
    Args:
        results: Dictionary containing experiment results
        save_path: Path to save the figure
    """
    plt.figure(figsize=(14, 8))
    
    dimensions = results['dimensions']
    
    # Plot PCA results
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, results['pca_qdf_rda_time'], 'o-', label='QDF (RDA)', linewidth=2, markersize=8)
    plt.plot(dimensions, results['pca_qdf_mqdf_time'], 's-', label='QDF (MQDF)', linewidth=2, markersize=8)
    plt.plot(dimensions, results['pca_knn_time'], '^-', label='KNN', linewidth=2, markersize=8)
    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('PCA + Classifiers - Training Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot LDA results
    plt.subplot(1, 2, 2)
    lda_dims = [min(d, 9) for d in dimensions]
    plt.plot(lda_dims, results['lda_qdf_rda_time'], 'o-', label='QDF (RDA)', linewidth=2, markersize=8)
    plt.plot(lda_dims, results['lda_qdf_mqdf_time'], 's-', label='QDF (MQDF)', linewidth=2, markersize=8)
    plt.plot(lda_dims, results['lda_knn_time'], '^-', label='KNN', linewidth=2, markersize=8)
    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('LDA + Classifiers - Training Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Time comparison plot saved to {save_path}")
    plt.close()


def plot_heatmap(results, save_path='accuracy_heatmap.png'):
    """
    Plot heatmap of accuracies
    
    Args:
        results: Dictionary containing experiment results
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    dimensions = results['dimensions']
    
    # PCA heatmap
    pca_data = np.array([
        results['pca_qdf_rda'],
        results['pca_qdf_mqdf'],
        results['pca_knn']
    ])
    
    sns.heatmap(pca_data, annot=True, fmt='.3f', cmap='YlGnBu', 
                xticklabels=dimensions, 
                yticklabels=['QDF (RDA)', 'QDF (MQDF)', 'KNN'],
                ax=axes[0], cbar_kws={'label': 'Accuracy'})
    axes[0].set_xlabel('Number of Dimensions', fontsize=12)
    axes[0].set_title('PCA + Classifiers', fontsize=14, fontweight='bold')
    
    # LDA heatmap
    lda_data = np.array([
        results['lda_qdf_rda'],
        results['lda_qdf_mqdf'],
        results['lda_knn']
    ])
    
    lda_dims = [min(d, 9) for d in dimensions]
    sns.heatmap(lda_data, annot=True, fmt='.3f', cmap='YlGnBu', 
                xticklabels=lda_dims, 
                yticklabels=['QDF (RDA)', 'QDF (MQDF)', 'KNN'],
                ax=axes[1], cbar_kws={'label': 'Accuracy'})
    axes[1].set_xlabel('Number of Dimensions', fontsize=12)
    axes[1].set_title('LDA + Classifiers', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {save_path}")
    plt.close()


def create_all_plots(results):
    """Create all visualization plots"""
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(results)
    plot_time_comparison(results)
    plot_heatmap(results)
    print("All visualizations completed!")
