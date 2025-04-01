"""
Utility functions for entity resolution.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Set, Any, Optional

logger = logging.getLogger("entity_resolution")

def setup_logging(output_dir: str, name: str = "entity_resolution") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        output_dir: Directory to save log file
        name: Logger name
        
    Returns:
        Configured logger
    """
    log_file = os.path.join(output_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(name)


def create_clusters_from_matches(df: pd.DataFrame, matches: List[Tuple[int, int, float]]) -> Dict[int, int]:
    """
    Create clusters using graph-based connected components.
    
    Args:
        df: Dataframe containing the records
        matches: List of tuples (idx1, idx2, score) representing matched record pairs
        
    Returns:
        Dictionary mapping record index to cluster ID
    """
    logger.info("Creating clusters from matches...")
    
    # Create a graph
    G = nx.Graph()
    
    # Add all records as nodes
    for idx in df.index:
        G.add_node(idx)
    
    # Add edges for matches with weights based on match score
    for idx1, idx2, score in matches:
        G.add_edge(idx1, idx2, weight=score)
    
    # Find connected components
    logger.info(f"Finding connected components...")
    clusters = list(nx.connected_components(G))
    logger.info(f"Found {len(clusters)} initial clusters")
    
    # Identify suspicious clusters (unusually large)
    suspicious_clusters = []
    for i, component in enumerate(clusters):
        if len(component) > 50:  # Large clusters are suspicious
            suspicious_clusters.append((i, component))
    
    # For suspicious clusters, try to split them using community detection
    if suspicious_clusters:
        logger.info(f"Found {len(suspicious_clusters)} suspicious large clusters")
        for i, component in suspicious_clusters:
            # Create subgraph for this component
            subgraph = G.subgraph(component)
            
            # Try to detect communities
            try:
                from networkx.algorithms import community
                communities = community.louvain_communities(subgraph)
                
                # If successful in finding multiple communities, replace the original component
                if len(communities) > 1:
                    logger.info(f"Split cluster {i} into {len(communities)} sub-clusters")
                    # Remove the original component
                    clusters.remove(component)
                    # Add the new communities
                    clusters.extend(communities)
            except Exception as e:
                logger.info(f"Could not split cluster {i}: {e}")
    
    # Create a mapping from record index to cluster ID
    cluster_mapping = {}
    for cluster_id, component in enumerate(clusters):
        for idx in component:
            cluster_mapping[idx] = cluster_id
    
    # For any records not in a cluster, assign them to their own singleton cluster
    next_cluster_id = len(clusters)
    for idx in df.index:
        if idx not in cluster_mapping:
            cluster_mapping[idx] = next_cluster_id
            next_cluster_id += 1
            
    logger.info(f"Final cluster count: {next_cluster_id}")
    logger.info(f"Largest cluster size: {max(len(c) for c in clusters) if clusters else 0}")
    
    return cluster_mapping


def create_canonical_records(df: pd.DataFrame, cluster_mapping: Dict[int, int]) -> pd.DataFrame:
    """
    Create canonical records for each cluster.
    
    Args:
        df: Original dataframe
        cluster_mapping: Dictionary mapping record index to cluster ID
        
    Returns:
        Dataframe with canonical records (one per cluster)
    """
    logger.info("Creating canonical records...")
    
    # Add cluster IDs to original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster_id'] = df_with_clusters.index.map(lambda x: cluster_mapping.get(x, x))
    
    # Group by cluster
    grouped = df_with_clusters.groupby('cluster_id')
    
    # Create canonical records
    canonical_records = []
    
    for cluster_id, group in grouped:
        # Choose most complete record as canonical based on non-null values
        completeness_scores = group.apply(
            lambda row: sum(1 for col in row.index if pd.notnull(row[col]) and row[col] != ''), 
            axis=1
        )
        most_complete_idx = completeness_scores.idxmax()
        canonical = group.loc[most_complete_idx].copy()
        
        # Add metadata about the cluster
        canonical['cluster_size'] = len(group)
        canonical['record_ids'] = ','.join(map(str, group.index.tolist()))
        
        # Enhanced logic: Try to find better values for key fields from the cluster
        
        # 1. For companies with missing names, find the most common non-empty name
        if not canonical['company_name'] or pd.isna(canonical['company_name']):
            from collections import Counter
            valid_names = group['company_name'].dropna().tolist()
            if valid_names:
                canonical['company_name'] = Counter(valid_names).most_common(1)[0][0]
        
        # 2. Use the most common domain if available
        if not canonical['website_domain'] or pd.isna(canonical['website_domain']):
            from collections import Counter
            valid_domains = group['website_domain'].dropna().tolist()
            if valid_domains:
                canonical['website_domain'] = Counter(valid_domains).most_common(1)[0][0]
        
        # 3. Use the most detailed address if available
        if not canonical['main_address_raw_text'] or pd.isna(canonical['main_address_raw_text']):
            # Look for the address with the most components
            valid_addresses = group['main_address_raw_text'].dropna()
            if not valid_addresses.empty:
                try:
                    address_complexity = valid_addresses.apply(lambda x: len(str(x).split(',')))
                    if not address_complexity.empty:
                        max_idx = address_complexity.idxmax()
                        canonical['main_address_raw_text'] = valid_addresses.loc[max_idx]
                except Exception as e:
                    # Safely handle indexing errors
                    logger.warning(f"Error selecting address: {e}")
                    # Fallback: just take the first non-empty address
                    canonical['main_address_raw_text'] = valid_addresses.iloc[0] if len(valid_addresses) > 0 else ""
        
        canonical_records.append(canonical)
    
    return pd.DataFrame(canonical_records)


def calculate_performance_metrics(df: pd.DataFrame, canonical_df: pd.DataFrame, 
                                  start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """
    Calculate performance metrics for the entity resolution approach.
    
    Args:
        df: Dataframe with clustered records
        canonical_df: Dataframe with canonical records
        start_time: Start time of the resolution
        end_time: End time of the resolution
        
    Returns:
        Dictionary of performance metrics
    """
    cluster_sizes = df['cluster_id'].value_counts()
    
    metrics = {
        'total_records': len(df),
        'unique_entities': len(canonical_df),
        'deduplication_rate': (len(df) - len(canonical_df)) / len(df) * 100,
        'average_cluster_size': cluster_sizes.mean(),
        'median_cluster_size': cluster_sizes.median(),
        'largest_cluster_size': cluster_sizes.max(),
        'singleton_clusters': sum(cluster_sizes == 1),
        'processing_time': (end_time - start_time).total_seconds(),
        'cluster_size_distribution': {
            '1 record': int(sum(cluster_sizes == 1)),
            '2-5 records': int(sum((cluster_sizes > 1) & (cluster_sizes <= 5))),
            '6-10 records': int(sum((cluster_sizes > 5) & (cluster_sizes <= 10))),
            '11-20 records': int(sum((cluster_sizes > 10) & (cluster_sizes <= 20))),
            '21-50 records': int(sum((cluster_sizes > 20) & (cluster_sizes <= 50))),
            '51-100 records': int(sum((cluster_sizes > 50) & (cluster_sizes <= 100))),
            '101+ records': int(sum(cluster_sizes > 100))
        }
    }
    
    return metrics


def save_results(approach_name: str, output_dir: str, result_df: pd.DataFrame, 
                canonical_df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    """
    Save the results of the entity resolution process.
    
    Args:
        approach_name: Name of the resolution approach
        output_dir: Directory to save results
        result_df: Dataframe with clustered records
        canonical_df: Dataframe with canonical records
        metrics: Performance metrics dictionary
    """
    approach_dir = os.path.join(output_dir, approach_name.replace(" ", "_").lower())
    os.makedirs(approach_dir, exist_ok=True)
    
    # Save the result dataframes
    result_df.to_csv(os.path.join(approach_dir, "resolved_data.csv"), index=False)
    canonical_df.to_csv(os.path.join(approach_dir, "canonical_entities.csv"), index=False)
    
    # Convert NumPy types to Python native types for JSON serialization
    json_safe_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            # Handle nested dictionaries
            json_safe_metrics[key] = {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v 
                                    for k, v in value.items()}
        else:
            # Handle direct values
            if isinstance(value, np.integer):
                json_safe_metrics[key] = int(value)
            elif isinstance(value, np.floating):
                json_safe_metrics[key] = float(value)
            else:
                json_safe_metrics[key] = value
    
    # Save performance metrics
    with open(os.path.join(approach_dir, "performance_metrics.json"), 'w') as f:
        json.dump(json_safe_metrics, f, indent=2)
        
    logger.info(f"Results saved to {approach_dir}")


def calculate_rand_index(df1: pd.DataFrame, df2: pd.DataFrame, sample_size: int = 10000) -> float:
    """
    Calculate Rand Index between two clusterings.
    
    Args:
        df1: First clustering
        df2: Second clustering
        sample_size: Number of pairs to sample
        
    Returns:
        Rand Index score (0-1)
    """
    # Ensure both DataFrames have the same indices
    common_indices = set(df1.index).intersection(set(df2.index))
    
    # Sample record pairs to evaluate
    from itertools import combinations
    
    # Sample records
    if len(common_indices) > 1000:
        sampled_indices = np.random.choice(list(common_indices), size=1000, replace=False)
    else:
        sampled_indices = list(common_indices)
    
    # Generate pairs
    pairs = list(combinations(sampled_indices, 2))
    
    # Sample pairs if too many
    if len(pairs) > sample_size:
        # Convert to a format that numpy.random.choice can handle
        pairs_array = np.array(pairs)
        # Randomly sample row indices since we can't directly sample pairs
        random_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
        # Use the sampled indices to get a subset of pairs
        pairs = [tuple(pairs_array[i]) for i in random_indices]
    
    # Count agreements and disagreements
    a = 0  # Both clusterings agree that the pair belongs to the same cluster
    b = 0  # Both clusterings agree that the pair belongs to different clusters
    c = 0  # Clustering 1 places the pair in the same cluster, but clustering 2 doesn't
    d = 0  # Clustering 2 places the pair in the same cluster, but clustering 1 doesn't
    
    for idx1, idx2 in pairs:
        same_cluster1 = df1.loc[idx1, 'cluster_id'] == df1.loc[idx2, 'cluster_id']
        same_cluster2 = df2.loc[idx1, 'cluster_id'] == df2.loc[idx2, 'cluster_id']
        
        if same_cluster1 and same_cluster2:
            a += 1
        elif not same_cluster1 and not same_cluster2:
            b += 1
        elif same_cluster1 and not same_cluster2:
            c += 1
        else:  # not same_cluster1 and same_cluster2
            d += 1
    
    # Calculate Rand Index
    rand_index = (a + b) / (a + b + c + d) if (a + b + c + d) > 0 else 0
    
    return rand_index
