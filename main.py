"""
Main script to run MNIST classification experiments
"""
import argparse
from experiments import run_experiments, print_results_table
from visualizations import create_all_plots


def main():
    parser = argparse.ArgumentParser(description='MNIST Classification Experiments')
    parser.add_argument('--dimensions', nargs='+', type=int, 
                        default=[10, 20, 30, 50, 100, 200],
                        help='List of dimensions to test (default: 10 20 30 50 100 200)')
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Number of samples to use (default: 10000, use None for full dataset)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Run experiments
    results = run_experiments(
        dimensions=args.dimensions,
        sample_size=args.sample_size if args.sample_size > 0 else None
    )
    
    # Print results table
    print_results_table(results)
    
    # Create visualizations
    if not args.no_viz:
        create_all_plots(results)


if __name__ == '__main__':
    main()
