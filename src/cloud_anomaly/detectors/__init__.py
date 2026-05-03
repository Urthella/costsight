"""Anomaly detectors. Each module exposes ``detect(df) -> pd.DataFrame``.

Output schema: columns ``date``, ``service``, ``cost``, ``score``, ``is_anomaly``.
"""
from .zscore import detect as zscore_detect
from .stl import detect as stl_detect
from .iforest import detect as iforest_detect

DETECTORS = {
    "zscore": zscore_detect,
    "stl": stl_detect,
    "iforest": iforest_detect,
}

__all__ = ["zscore_detect", "stl_detect", "iforest_detect", "DETECTORS"]
