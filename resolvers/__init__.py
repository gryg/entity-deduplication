"""
Entity resolution resolvers.
"""

from .rule_based import RuleBasedResolver
from .graph_based import GraphBasedResolver
from .ml_based import MLEntityResolver
from .semantic_matching import SemanticMatchingResolver
from .deep_learning import DeepLearningResolver

__all__ = [
    'RuleBasedResolver',
    'GraphBasedResolver',
    'MLEntityResolver',
    'SemanticMatchingResolver',
    'DeepLearningResolver'
]