"""SVM Classification models.

Models that implement the Support Vector Machine algorithm for classification tasks.
Finds the optimal hyperplane that separates the classes in the feature space.
Handles linear and non-linear classification tasks by using different kernel functions.
"""

from .numpy_impl import SvmClassification
from .scratch import SvmClassification as SvmClassificationScratch

__all__ = [
    "SvmClassification",
    "SvmClassificationScratch",
]
