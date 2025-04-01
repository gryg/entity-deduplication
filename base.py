"""
Base class for entity resolution approaches.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from preprocessing import preprocess_dataframe, extract_record_data
from utils import (
    create_clusters_from_matches,
    create_canonical_records,
    calculate_performance_metrics,
    save_results
)

logger = logging.getLogger("entity_resolution")

class EntityResolutionBase:
    """Base class for entity resolution approaches."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize with approach name and description.
        
        Args:
            name: Name of the approach
            description: Description of the approach
        """
        self.name = name
        self.description = description
        self.results = {}
        self.performance_metrics = {}
        self.start_time = None
        self.end_time = None
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for entity resolution.
        
        Args:
            df: Input dataframe
            
        Returns:
            Processed dataframe with normalized fields
        """
        logger.info(f"[{self.name}] Preprocessing data...")
        return preprocess_dataframe(df)
    
    def _extract_record_data(self, record: pd.Series) -> Dict[str, Any]:
        """
        Extract relevant fields from a record for comparison.
        
        Args:
            record: Pandas Series containing a data record
            
        Returns:
            Dictionary of normalized fields for comparison
        """
        return extract_record_data(record)
    
    def _create_canonical_records(self, df: pd.DataFrame, cluster_mapping: Dict[int, int]) -> pd.DataFrame:
        """
        Create canonical records for each cluster.
        
        Args:
            df: Original dataframe
            cluster_mapping: Dictionary mapping record index to cluster ID
            
        Returns:
            Dataframe with canonical records (one per cluster)
        """
        logger.info(f"[{self.name}] Creating canonical records...")
        return create_canonical_records(df, cluster_mapping)
    
    def calculate_performance_metrics(self, df: pd.DataFrame, canonical_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics for the approach.
        
        Args:
            df: Dataframe with clustered records
            canonical_df: Dataframe with canonical records
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = calculate_performance_metrics(df, canonical_df, self.start_time, self.end_time)
        self.performance_metrics = metrics
        return metrics
    
    def _create_clusters(self, df: pd.DataFrame, matches: List[Tuple[int, int, float]]) -> Dict[int, int]:
        """
        Create clusters using graph-based connected components.
        
        Args:
            df: Dataframe containing the records
            matches: List of tuples (idx1, idx2, score) representing matched record pairs
            
        Returns:
            Dictionary mapping record index to cluster ID
        """
        logger.info(f"[{self.name}] Creating clusters...")
        return create_clusters_from_matches(df, matches)
    
    def _run_resolution(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Implementation-specific entity resolution logic.
        To be implemented by subclasses.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (result_df, canonical_df)
        """
        raise NotImplementedError("Each approach must implement _run_resolution method")
        
    def run_entity_resolution(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the entity resolution process.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (result_df, canonical_df)
        """
        logger.info(f"Running entity resolution with {self.name}...")
        self.start_time = datetime.now()
        
        try:
            result_df, canonical_df = self._run_resolution(df)
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        self.end_time = datetime.now()
        logger.info(f"Entity resolution completed in {(self.end_time - self.start_time).total_seconds():.2f} seconds")
        
        if result_df is not None and canonical_df is not None:
            # Calculate performance metrics
            self.calculate_performance_metrics(result_df, canonical_df)
            
        return result_df, canonical_df
        
    def save_results(self, output_dir: str, result_df: pd.DataFrame, canonical_df: pd.DataFrame) -> None:
        """
        Save the results of the entity resolution process.
        
        Args:
            output_dir: Directory to save results
            result_df: Dataframe with clustered records
            canonical_df: Dataframe with canonical records
        """
        save_results(self.name, output_dir, result_df, canonical_df, self.performance_metrics)
