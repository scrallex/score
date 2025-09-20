"""
Top‑level package for the Sep Text Manifold library.

This package provides modules for ingesting text data, computing
quantum‑inspired metrics over sliding windows of bytes, building a
manifold of signals, extracting and scoring strings, detecting themes
and exposing the results via a simple API and command line interface.

The heavy numerical work is delegated to the SEP Engine's QFH/QBSA
implementation.  See `docs/integration_with_sep.md` for details on
how to integrate with the existing C++ codebase.
"""

from .ingest import ingest_directory
from .encode import encode_window
from .manifold import build_manifold
from .pipeline import AnalysisResult, analyse_directory
from .strings import extract_strings, aggregate_string_metrics
from .scoring import patternability_score, connector_score
from .themes import build_theme_graph, detect_themes
from .dilution import (
    path_dilution,
    signal_dilution,
    semantic_dilution,
    compute_dilution_metrics,
)
from .comparison import stm_val_alignment, detection_lead_time, AlignmentResult, LeadTimeResult
from .feedback import suggest_twin_action, TwinSuggestion

__all__ = [
    "ingest_directory",
    "encode_window",
    "build_manifold",
    "extract_strings",
    "aggregate_string_metrics",
    "analyse_directory",
    "AnalysisResult",
    "patternability_score",
    "connector_score",
    "build_theme_graph",
    "detect_themes",
    "path_dilution",
    "signal_dilution",
    "semantic_dilution",
    "compute_dilution_metrics",
    "stm_val_alignment",
    "detection_lead_time",
    "AlignmentResult",
    "LeadTimeResult",
    "suggest_twin_action",
    "TwinSuggestion",
]
