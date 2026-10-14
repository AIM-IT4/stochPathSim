"""Model-validation lab: convergence and discretisation studies, model comparison and HTML reports."""

from .compare import ModelComparison, compare_models
from .convergence import convergence_plot, convergence_study, discretisation_plot, discretisation_study
from .report import Check, ValidationReport

__all__ = [
    "convergence_study",
    "convergence_plot",
    "discretisation_study",
    "discretisation_plot",
    "compare_models",
    "ModelComparison",
    "ValidationReport",
    "Check",
]
