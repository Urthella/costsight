"""A-priori (theoretical) detector ratings used in the proposal deck.

These are the qualitative scores we predicted before measuring anything,
based on textbook reasoning about each algorithm. They live next to the
empirical numbers so the dashboard can show "what we expected" vs.
"what we measured" side-by-side — the central narrative of Project 13.

All values are normalized to [0, 1]. Higher is better for every axis,
including Speed and Interpretability.
"""
from __future__ import annotations

RADAR_AXES = (
    "Point Spike",
    "Level Shift",
    "Gradual Drift",
    "Speed",
    "Interpretability",
)

THEORETICAL_SCORES: dict[str, dict[str, float]] = {
    "zscore": {
        "Point Spike": 0.95,
        "Level Shift": 0.20,
        "Gradual Drift": 0.10,
        "Speed": 0.95,
        "Interpretability": 0.95,
    },
    "stl": {
        "Point Spike": 0.80,
        "Level Shift": 0.80,
        "Gradual Drift": 0.85,
        "Speed": 0.55,
        "Interpretability": 0.75,
    },
    "iforest": {
        "Point Spike": 0.65,
        "Level Shift": 0.60,
        "Gradual Drift": 0.55,
        "Speed": 0.65,
        "Interpretability": 0.30,
    },
    "ensemble": {
        "Point Spike": 0.85,
        "Level Shift": 0.75,
        "Gradual Drift": 0.70,
        "Speed": 0.40,
        "Interpretability": 0.65,
    },
}

INTERPRETABILITY_QUALITATIVE: dict[str, float] = {
    "zscore": 0.95,
    "stl": 0.75,
    "iforest": 0.30,
    "ensemble": 0.65,
}
