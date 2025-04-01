#!/usr/bin/env python
"""
Entity Resolution Runner Script

This script runs the entity resolution framework on a dataset and compares approaches.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from base import EntityResolutionBase
from resolvers.rule_based import RuleBasedResolver
from resolvers.graph_based import GraphBasedResolver
from resolvers.ml_based import MLEntityResolver
from resolvers.semantic_matching import SemanticMatchingResolver
from resolvers.deep_learning import DeepLearningResolver
from resolvers.deterministic_feature import DeterministicFeatureResolver
from comparison import EntityResolutionComparison
from utils import setup_logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Entity Resolution Framework")
    
    parser.add_argument("--input", type=str, required=True,
                        help="Input parquet file path")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--sample-size", type=int,
                        help="Sample size for testing with smaller dataset")
    parser.add_argument("--approaches", type=str, default="ALL",
                        help="Comma-separated list of approaches to run (ML,RULE,GRAPH,SEMANTIC,DEEP,DETERMINISTIC or ALL)")
    parser.add_argument("--max-comparisons", type=int, default=None,
                        help="Maximum number of pairwise comparisons to perform (for large datasets)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs to use (-1 for all available cores)")
    parser.add_argument("--block-size-limit", type=int, default=1000,
                        help="Maximum number of records in a block (prevent quadratic explosion)")
    
    return parser.parse_args()

def main():
    """Main function to run the entity resolution framework."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.output_dir)
    
    # Initialize comparison framework
    comparison = EntityResolutionComparison(
        input_file=args.input,
        output_dir=args.output_dir,
        sample_size=args.sample_size
    )
    
    # Determine which approaches to run
    if args.approaches.upper() == "ALL":
        approaches_to_run = ["ML", "RULE", "GRAPH", "SEMANTIC", "DEEP", "DETERMINISTIC"]
    else:
        approaches_to_run = [a.strip().upper() for a in args.approaches.split(",")]
    
    # Register selected approaches
    for approach in approaches_to_run:
        if approach == "RULE":
            comparison.register_approach(RuleBasedResolver(match_threshold=0.7))
        elif approach == "ML":
            comparison.register_approach(MLEntityResolver(match_threshold=0.5, training_pairs=10000))
        elif approach == "GRAPH":
            comparison.register_approach(GraphBasedResolver(edge_threshold=0.6, resolution=1.2))
        elif approach == "SEMANTIC":
            comparison.register_approach(SemanticMatchingResolver(similarity_threshold=0.7, use_tfidf=True))
        elif approach == "DEEP":
            comparison.register_approach(DeepLearningResolver(match_threshold=0.5, model_type='feedforward'))
        elif approach == "DETERMINISTIC":
            comparison.register_approach(DeterministicFeatureResolver(
                match_threshold=0.75, 
                max_comparisons=args.max_comparisons,
                n_jobs=args.n_jobs,
                use_advanced_blocking=True,
                use_progressive_resolution=True,
                block_size_limit=args.block_size_limit,
                max_deduplication_rate=0.40
            ))
        else:
            logger.warning(f"Unknown approach: {approach}")
    
    # Load data
    comparison.load_data()
    
    # Run all registered approaches
    results = comparison.run_all_approaches()
    
    # Compare the approaches
    comparison.compare_approaches()
    
    # Generate visualizations
    comparison.generate_comparison_visualizations()
    
    # Print summary
    print("\nEntity Resolution Summary:")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Approaches run: {', '.join(approaches_to_run)}")
    for approach_name, (_, canonical_df) in results.items():
        approach = next((a for a in comparison.approaches if a.name == approach_name), None)
        if approach and hasattr(approach, 'performance_metrics') and approach.performance_metrics:
            print(f"\n{approach.name}:")
            print(f"  - Unique entities: {approach.performance_metrics['unique_entities']:,}")
            print(f"  - Deduplication rate: {approach.performance_metrics['deduplication_rate']:.2f}%")
            print(f"  - Processing time: {approach.performance_metrics['processing_time']:.2f} seconds")
    
    print(f"\nDetailed results and logs saved to {args.output_dir}")

if __name__ == "__main__":
    main()