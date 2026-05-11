"""
utils.py
--------
Fonctions utilitaires partagées : temps, lissage, filtrage de bruit,
normalisation des coordonnées.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, List, Tuple

import numpy as np


def now() -> float:
    """Timestamp haute résolution (secondes)."""
    return time.perf_counter()


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Distance euclidienne entre deux points 2D."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


class MovingAverage2D:
    """
    Lissage d'une trajectoire 2D via moyenne glissante.
    Réduit le tremblement du doigt et la jitter des landmarks.
    """

    def __init__(self, window: int = 5) -> None:
        self.window = max(1, int(window))
        self._buf: Deque[Tuple[float, float]] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._buf.clear()

    def update(self, point: Tuple[float, float]) -> Tuple[float, float]:
        self._buf.append(point)
        xs = np.mean([p[0] for p in self._buf])
        ys = np.mean([p[1] for p in self._buf])
        return float(xs), float(ys)


class FPSCounter:
    """Compteur de FPS basé sur une fenêtre glissante des delta-temps."""

    def __init__(self, window: int = 30) -> None:
        self._times: Deque[float] = deque(maxlen=window)
        self._last: float | None = None

    def tick(self) -> float:
        t = now()
        if self._last is not None:
            self._times.append(t - self._last)
        self._last = t
        if not self._times:
            return 0.0
        mean_dt = float(np.mean(self._times))
        return 1.0 / mean_dt if mean_dt > 0 else 0.0


def filter_small_movement(
    points: List[Tuple[int, int]],
    new_point: Tuple[int, int],
    min_dist: float = 2.0,
) -> bool:
    """
    Renvoie True si le nouveau point doit être ajouté (mouvement significatif).
    Filtre les micro-mouvements (bruit).
    """
    if not points:
        return True
    return euclidean(points[-1], new_point) >= min_dist
