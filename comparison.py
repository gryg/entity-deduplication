"""
Comparison framework for entity resolution approaches.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

from utils import calculate_rand_index

logger = logging.getLogger("entity_resolution")

class EntityResolutionComparison:
    """Framework for comparing different entity resolution approaches."""
    
    def __init__(self, input_file: str, output_dir: str, sample_size: Optional[int] = None):
        """
        Initialize the comparison framework.
        
        Args:
            input_file: Path to input parquet file
            output_dir: Directory to save output files
            sample_size: Optional sample size for testing with smaller dataset
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.approaches = []
        self.results = {}
        self.df = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def register_approach(self, approach) -> None:
        """
        Register an entity resolution approach.
        
        Args:
            approach: EntityResolutionBase instance
        """
        self.approaches.append(approach)
        logger.info(f"Registered approach: {approach.name}")
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the data from the input file.
        
        Returns:
            Loaded dataframe
        """
        logger.info(f"Loading data from {self.input_file}...")
        try:
            df = pd.read_parquet(self.input_file)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            # Use a sample if specified
            if self.sample_size and self.sample_size < len(df):
                logger.info(f"Using sample of {self.sample_size} records")
                df = df.sample(self.sample_size, random_state=42).reset_index(drop=True)
                
            self.df = df
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def run_all_approaches(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Run all registered approaches and store results.
        
        Returns:
            Dictionary mapping approach names to result tuples (result_df, canonical_df)
        """
        if self.df is None:
            self.load_data()
            
        results = {}
        
        for approach in self.approaches:
            logger.info(f"Running approach: {approach.name}")
            try:
                result_df, canonical_df = approach.run_entity_resolution(self.df)
                
                # Save the results
                approach.save_results(self.output_dir, result_df, canonical_df)
                
                # Store results for comparison
                results[approach.name] = (result_df, canonical_df)
                
                logger.info(f"Completed approach: {approach.name}")
                logger.info(f"Found {len(canonical_df)} unique entities (deduplication rate: {approach.performance_metrics['deduplication_rate']:.2f}%)")
            except Exception as e:
                logger.error(f"Error running approach {approach.name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        self.results = results
        return results
    
    def compare_approaches(self) -> None:
        """Compare the results of all approaches and generate comparison report."""
        if not self.results:
            logger.warning("No results to compare. Run approaches first.")
            return
            
        logger.info("Comparing approaches...")
        
        # Create comparison metrics
        comparison = {
            'approach_names': [],
            'unique_entities': [],
            'deduplication_rates': [],
            'largest_clusters': [],
            'processing_times': [],
            'singleton_ratios': []
        }
        
        for approach_name, (_, canonical_df) in self.results.items():
            approach = next((a for a in self.approaches if a.name == approach_name), None)
            if approach and approach.performance_metrics:
                comparison['approach_names'].append(approach_name)
                comparison['unique_entities'].append(approach.performance_metrics['unique_entities'])
                comparison['deduplication_rates'].append(approach.performance_metrics['deduplication_rate'])
                comparison['largest_clusters'].append(approach.performance_metrics['largest_cluster_size'])
                comparison['processing_times'].append(approach.performance_metrics['processing_time'])
                
                # Calculate singleton ratio
                singleton_ratio = (approach.performance_metrics['singleton_clusters'] / 
                                   approach.performance_metrics['unique_entities'] * 100)
                comparison['singleton_ratios'].append(singleton_ratio)
        
        # Create a comparison DataFrame
        comparison_df = pd.DataFrame({
            'Approach': comparison['approach_names'],
            'Unique Entities': comparison['unique_entities'],
            'Deduplication Rate (%)': comparison['deduplication_rates'],
            'Largest Cluster Size': comparison['largest_clusters'],
            'Processing Time (s)': comparison['processing_times'],
            'Singleton Ratio (%)': comparison['singleton_ratios']
        })
        
        # Save comparison to CSV
        comparison_file = os.path.join(self.output_dir, "approach_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Comparison saved to {comparison_file}")
        
        # Print comparison table
        print("\nComparison of Entity Resolution Approaches:")
        print(comparison_df.to_string(index=False))
        
        # Also compare cluster distributions
        self._compare_cluster_distributions()
        
        # Compare pairwise agreement between approaches
        self._compare_pairwise_agreement()
    
    def _compare_cluster_distributions(self) -> None:
        """Compare the distribution of cluster sizes across approaches."""
        if not self.results:
            return
            
        # Create distribution comparison
        distributions = {}
        
        for approach_name, (result_df, _) in self.results.items():
            # Calculate cluster size distribution
            cluster_sizes = result_df['cluster_id'].value_counts()
            distribution = {
                '1 record': int(sum(cluster_sizes == 1)),
                '2-5 records': int(sum((cluster_sizes > 1) & (cluster_sizes <= 5))),
                '6-10 records': int(sum((cluster_sizes > 5) & (cluster_sizes <= 10))),
                '11-20 records': int(sum((cluster_sizes > 10) & (cluster_sizes <= 20))),
                '21-50 records': int(sum((cluster_sizes > 20) & (cluster_sizes <= 50))),
                '51-100 records': int(sum((cluster_sizes > 50) & (cluster_sizes <= 100))),
                '101+ records': int(sum(cluster_sizes > 100))
            }
            distributions[approach_name] = distribution
        
        # Create DataFrame
        dist_df = pd.DataFrame(distributions).T
        
        # Save to CSV
        dist_file = os.path.join(self.output_dir, "cluster_size_distribution_comparison.csv")
        dist_df.to_csv(dist_file)
        logger.info(f"Cluster size distribution comparison saved to {dist_file}")
    
    def _compare_pairwise_agreement(self) -> None:
        """Compare the agreement between different approaches on record pairs."""
        if len(self.results) < 2:
            logger.info("At least two approaches required for pairwise agreement comparison")
            return
            
        logger.info("Calculating pairwise agreement between approaches...")
        
        # Get all approach names
        approach_names = list(self.results.keys())
        
        # Initialize agreement matrix
        agreement_matrix = pd.DataFrame(
            index=approach_names,
            columns=approach_names,
            data=0.0
        )
        
        # Calculate Rand Index between each pair of approaches
        for i, approach1 in enumerate(approach_names):
            result_df1 = self.results[approach1][0]
            
            for j, approach2 in enumerate(approach_names[i:], i):
                if i == j:
                    agreement_matrix.loc[approach1, approach2] = 1.0
                    continue
                    
                result_df2 = self.results[approach2][0]
                
                # Calculate agreement using a sample of record pairs
                agreement_score = calculate_rand_index(result_df1, result_df2)
                
                agreement_matrix.loc[approach1, approach2] = agreement_score
                agreement_matrix.loc[approach2, approach1] = agreement_score
        
        # Save agreement matrix
        agreement_file = os.path.join(self.output_dir, "approach_agreement_matrix.csv")
        agreement_matrix.to_csv(agreement_file)
        logger.info(f"Approach agreement matrix saved to {agreement_file}")
    
    def generate_comparison_visualizations(self) -> None:
        """Generate visualizations comparing the different approaches."""
        if not self.results:
            return
            
        logger.info("Generating comparison visualizations...")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Bar chart of deduplication rates
        plt.figure(figsize=(10, 6))
        approach_names = []
        dedup_rates = []
        
        for approach in self.approaches:
            if approach.performance_metrics:
                approach_names.append(approach.name)
                dedup_rates.append(approach.performance_metrics['deduplication_rate'])
        
        plt.bar(approach_names, dedup_rates, color='skyblue')
        plt.title('Deduplication Rate by Approach', fontsize=14)
        plt.xlabel('Approach', fontsize=12)
        plt.ylabel('Deduplication Rate (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "deduplication_rate_comparison.png"), dpi=300)
        plt.close()
        
        # 2. Execution time comparison
        plt.figure(figsize=(10, 6))
        exec_times = [approach.performance_metrics['processing_time'] for approach in self.approaches if approach.performance_metrics]
        
        plt.bar(approach_names, exec_times, color='lightgreen')
        plt.title('Processing Time by Approach', fontsize=14)
        plt.xlabel('Approach', fontsize=12)
        plt.ylabel('Processing Time (seconds)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "processing_time_comparison.png"), dpi=300)
        plt.close()
        
        # 3. Cluster size distribution comparison
        plt.figure(figsize=(12, 8))
        
        distributions = {}
        for approach_name, (result_df, _) in self.results.items():
            # Calculate cluster size distribution
            cluster_sizes = result_df['cluster_id'].value_counts()
            distribution = {
                '1 record': int(sum(cluster_sizes == 1)),
                '2-5 records': int(sum((cluster_sizes > 1) & (cluster_sizes <= 5))),
                '6-10 records': int(sum((cluster_sizes > 5) & (cluster_sizes <= 10))),
                '11-20 records': int(sum((cluster_sizes > 10) & (cluster_sizes <= 20))),
                '21-50 records': int(sum((cluster_sizes > 20) & (cluster_sizes <= 50))),
                '51-100 records': int(sum((cluster_sizes > 50) & (cluster_sizes <= 100))),
                '101+ records': int(sum(cluster_sizes > 100))
            }
            distributions[approach_name] = distribution
        
        # Create DataFrame for plotting
        dist_df = pd.DataFrame(distributions).T
        
        # Plot stacked bar chart
        dist_df.plot(kind='bar', stacked=False, figsize=(12, 8), colormap='viridis')
        plt.title('Cluster Size Distribution by Approach', fontsize=14)
        plt.xlabel('Approach', fontsize=12)
        plt.ylabel('Number of Clusters', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Cluster Size', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "cluster_size_distribution_comparison.png"), dpi=300)
        plt.close()
        
        # 4. Heatmap of approach agreement
        if len(self.results) >= 2:
            agreement_file = os.path.join(self.output_dir, "approach_agreement_matrix.csv")
            if os.path.exists(agreement_file):
                agreement_matrix = pd.read_csv(agreement_file, index_col=0)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(agreement_matrix, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
                plt.title('Approach Agreement (Rand Index)', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, "approach_agreement_heatmap.png"), dpi=300)
                plt.close()
        
        logger.info(f"Comparison visualizations saved to {viz_dir}")