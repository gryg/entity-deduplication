"""
Entity Resolution Framework

A modular framework for identifying duplicate company records in datasets
with multiple entity resolution approaches.
"""

from base import EntityResolutionBase
from comparison import EntityResolutionComparison
from resolvers import (
    RuleBasedResolver,
    GraphBasedResolver,
    MLEntityResolver,
    SemanticMatchingResolver,
    DeepLearningResolver,
    DeterministicFeatureResolver
)

__all__ = [
    'EntityResolutionBase',
    'EntityResolutionComparison',
    'RuleBasedResolver',
    'GraphBasedResolver',
    'MLEntityResolver',
    'SemanticMatchingResolver',
    'DeepLearningResolver',
    'DeterministicFeatureResolver'
]