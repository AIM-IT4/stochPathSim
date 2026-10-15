"""Interactive Plotly charts. Every function returns a ``plotly.graph_objects.Figure``."""

from .attribution import portfolio_risk_chart
from .paths import correlation_heatmap, density_evolution, fan, path_explorer, paths, terminal_distribution
from .rates import curve_fit_check, short_rate_fan
from .risk import backtest_dashboard, exposure_dashboard, var_plot
from .theme import PALETTE, set_theme, template

__all__ = [
    "paths",
    "fan",
    "path_explorer",
    "terminal_distribution",
    "density_evolution",
    "correlation_heatmap",
    "exposure_dashboard",
    "backtest_dashboard",
    "var_plot",
    "portfolio_risk_chart",
    "curve_fit_check",
    "short_rate_fan",
    "set_theme",
    "template",
    "PALETTE",
]
