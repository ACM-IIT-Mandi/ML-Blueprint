"""Support vector machines.

Classifiers that try to draw the widest possible boundary between classes, and the
kernel trick that lets them handle curved boundaries too.

Algorithms
----------
- `svmClassification`: SVM classification using NumPy.
"""

from .svm_classification import SvmClassification, SvmClassificationScratch 

__all__: list[str] = [
    "SvmClassification",
    "SvmClassificationScratch",
]
