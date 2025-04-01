"""
Graph-based entity resolution approach.
"""

import logging
import pandas as pd
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
from rapidfuzz import fuzz

from base import EntityResolutionBase

logger = logging.getLogger("entity_resolution")

class GraphBasedResolver(EntityResolutionBase):
    """Enhanced graph-based entity resolution approach using community detection."""
    
    def __init__(self, edge_threshold=0.6, resolution=1.2):
        """
        Initialize the graph-based resolver with configurable parameters.
        
        Args:
            edge_threshold: Threshold for adding edges to the graph (default: 0.6)
            resolution: Resolution parameter for community detection (default: 1.2)
        """
        super().__init__(
            name="Graph-based Entity Resolution",
            description="Uses graph clustering and community detection to identify duplicates"
        )
        self.edge_threshold = edge_threshold
        self.resolution = resolution
    
    def _run_resolution(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run graph-based entity resolution process.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (result_df, canonical_df)
        """
        # Preprocess the data
        processed_df = self.preprocess_data(df)
        
        # Generate initial similarity graph
        G = self._build_similarity_graph(processed_df)
        
        # Apply community detection to find clusters
        clusters = self._detect_communities(G)
        
        # Create mapping from record index to cluster ID
        cluster_mapping = {}
        for cluster_id, nodes in enumerate(clusters):
            for node in nodes:
                cluster_mapping[node] = cluster_id
        
        # For any records not in a cluster, assign them to their own singleton cluster
        next_cluster_id = len(clusters)
        for idx in df.index:
            if idx not in cluster_mapping:
                cluster_mapping[idx] = next_cluster_id
                next_cluster_id += 1
        
        # Add clusters to original DataFrame
        result_df = df.copy()
        result_df['cluster_id'] = result_df.index.map(lambda x: cluster_mapping.get(x, x))
        
        # Create canonical records
        canonical_df = self._create_canonical_records(df, cluster_mapping)
        
        return result_df, canonical_df
    
    def _build_similarity_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build a similarity graph connecting potentially matching entities.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            NetworkX graph with nodes as record indices and edges connecting similar records
        """
        logger.info(f"[{self.name}] Building similarity graph...")
        
        # Create a graph
        G = nx.Graph()
        
        # Add all records as nodes
        for idx in df.index:
            G.add_node(idx)
        
        # Create blocking keys to reduce comparison space
        blocks = defaultdict(list)
        
        # Domain blocking
        df_with_domain = df[df['normalized_domain'] != '']
        domain_groups = df_with_domain.groupby('normalized_domain').indices
        for domain, indices in domain_groups.items():
            if domain and len(indices) > 1:
                blocks[f"domain_{domain}"] = list(indices)
        
        # Phone blocking
        df_with_phone = df[df['normalized_phone'] != '']
        phone_groups = df_with_phone.groupby('normalized_phone').indices
        for phone, indices in phone_groups.items():
            if phone and len(phone) >= 10 and len(indices) > 1:
                blocks[f"phone_{phone}"] = list(indices)
        
        # Name + country blocking
        for idx, row in df.iterrows():
            name_prefix = row['normalized_name'][:4] if isinstance(row['normalized_name'], str) and row['normalized_name'] else ''
            country = row['main_country_code'] if isinstance(row['main_country_code'], str) and row['main_country_code'] else ''
            if name_prefix and country:
                blocks[f"name_country_{name_prefix}_{country}"].append(idx)
        
        # Email domain blocking (excluding common providers)
        email_domain_groups = df[df['email_domain'] != ''].groupby('email_domain').indices
        for domain, indices in email_domain_groups.items():
            if domain and domain not in ['gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com']:
                blocks[f"email_{domain}"] = list(indices)
        
        # Process each block to add edges
        edges_added = 0
        for block_key, indices in blocks.items():
            # Skip overly large blocks to prevent quadratic explosion
            if len(indices) > 100:
                continue
                
            # Add edges for each pair in the block
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    rec1 = self._extract_record_data(df.iloc[idx1])
                    rec2 = self._extract_record_data(df.iloc[idx2])
                    
                    # Compute similarity score
                    name_sim = fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0
                    
                    # Add edge if similarity is above threshold with weight based on block type
                    if 'domain' in block_key:
                        # Domain match - require moderate name similarity
                        if name_sim > self.edge_threshold:
                            G.add_edge(idx1, idx2, weight=0.8)
                            edges_added += 1
                    elif 'phone' in block_key:
                        # Phone match - strong signal
                        G.add_edge(idx1, idx2, weight=0.9)
                        edges_added += 1
                    elif 'email' in block_key:
                        # Business email domain is a strong signal
                        if name_sim > self.edge_threshold:
                            G.add_edge(idx1, idx2, weight=0.85)
                            edges_added += 1
                    elif name_sim > 0.85:
                        # High name similarity in same country
                        G.add_edge(idx1, idx2, weight=name_sim)
                        edges_added += 1
        
        logger.info(f"[{self.name}] Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _detect_communities(self, G: nx.Graph) -> List[Set[int]]:
        """
        Detect communities in the similarity graph.
        
        Args:
            G: NetworkX graph with nodes as record indices
            
        Returns:
            List of sets, each containing node indices belonging to the same community
        """
        logger.info(f"[{self.name}] Detecting communities...")
        
        # First, find connected components
        components = list(nx.connected_components(G))
        logger.info(f"[{self.name}] Found {len(components)} connected components")
        
        # Then, for large components, apply community detection
        refined_clusters = []
        large_component_count = 0
        
        for component in components:
            if len(component) > 50:
                large_component_count += 1
                subgraph = G.subgraph(component)
                
                try:
                    # Try to detect communities with Louvain method
                    from networkx.algorithms import community
                    communities = community.louvain_communities(subgraph, resolution=self.resolution)
                    
                    logger.info(f"[{self.name}] Split component with {len(component)} nodes into {len(communities)} communities")
                    refined_clusters.extend(communities)
                except Exception as e:
                    logger.info(f"[{self.name}] Could not split component: {e}")
                    refined_clusters.append(component)
            else:
                refined_clusters.append(component)
        
        logger.info(f"[{self.name}] Refined {large_component_count} large components")
        logger.info(f"[{self.name}] Final cluster count: {len(refined_clusters)}")
        return refined_clusters
