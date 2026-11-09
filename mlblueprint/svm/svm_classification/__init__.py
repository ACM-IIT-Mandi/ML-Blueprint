"""SVM Classification models.

Models that implement the Support Vector Machine (SVM) algorithm for classification tasks.
The SVM algorithm finds the optimal hyperplane that separates the classes in the feature space. 
It can handle both linear and non-linear classification tasks by using different kernel functions. 
"""

from .numpy_impl import SvmClassification
from .scratch import SvmClassification as SvmClassificationScratch

__all__ = [
    "SvmClassification",
    "SvmClassificationScratch",
]
