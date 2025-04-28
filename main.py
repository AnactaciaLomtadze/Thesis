import os
import argparse
import logging
import time
import json
from experiment_runner import ForgettingConfig, ForgettingExperimentRunner

def setup_logger():
    """Set up logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("forgetting_recommender.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run forgetting mechanism experiments')
    
    parser.add_argument('--data_path', type=str, default='./ml-100k',
                        help='Path to MovieLens dataset')
    
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    
    parser.add_argument('--experiments', type=str, nargs='+',
                        default=['baseline_comparison', 'temporal_evaluation', 
                                'parameter_sensitivity', 'privacy_impact', 
                                'scalability_test', 'user_segmentation'],
                        help='Experiments to run')
    
    parser.add_argument('--num_users', type=int, default=50,
                        help='Number of users to evaluate')
    
    parser.add_argument('--temporal_split', action='store_true',
                        help='Use temporal split instead of random')
    
    parser.add_argument('--test_days', type=int, default=30,
                        help='Number of days for testing in temporal split')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def run_experiments(config, experiments):
    """Run the specified experiments."""
    logger = logging.getLogger(__name__)
    
    # Initialize experiment runner
    runner = ForgettingExperimentRunner(config)
    
    # Track results
    results = {}
    
    # Run each experiment
    for experiment in experiments:
        logger.info(f"Starting experiment: {experiment}")
        start_time = time.time()
        
        experiment_results = runner.run_experiment(experiment)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Completed experiment: {experiment} in {duration:.2f} seconds")
        
        # Store meta information
        results[experiment] = {
            'completed': True,
            'duration': duration
        }
    
    return results

def generate_report(config, experiment_results):
    """Generate a simple report about the experiments."""
    logger = logging.getLogger(__name__)
    
    report = {
        'configuration': {
            'data_path': config.data_path,
            'output_dir': config.output_dir,
            'num_users': config.num_users,
            'temporal_split': config.temporal_split,
            'test_days': config.test_days,
            'seed': config.seed
        },
        'experiments': experiment_results
    }
    
    # Save report as JSON
    report_path = os.path.join(config.output_dir, 'experiment_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"Report generated and saved to {report_path}")
    
    # Print summary to console
    print("\n===== Experiment Summary =====")
    for experiment, results in experiment_results.items():
        print(f"- {experiment}: {'Completed' if results['completed'] else 'Failed'} in {results['duration']:.2f} seconds")
    print("==============================\n")

def main():
    """Main function to run the application."""
    # Set up logger
    logger = setup_logger()
    logger.info("Starting Forgetting Mechanism for Recommendation Systems")
    
    # Parse arguments
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up configuration
    config = ForgettingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_users=args.num_users,
        temporal_split=args.temporal_split,
        test_days=args.test_days,
        seed=args.seed
    )
    
    logger.info(f"Configuration: {vars(config)}")
    
    # Run experiments
    experiment_results = run_experiments(config, args.experiments)
    
    # Generate report
    generate_report(config, experiment_results)
    
    logger.info("All experiments completed successfully")

if __name__ == "__main__":
    main()