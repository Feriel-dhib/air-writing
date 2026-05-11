"""
trajectory.py
-------------
Buffer de trajectoire 2D + logique temporelle.

Le TrajectoryBuffer :
 - Accumule les points (x, y) pendant que l'utilisateur écrit.
 - Mémorise le dernier timestamp d'activité.
 - Permet de détecter une pause (fin de caractère) au-delà d'un seuil.
 - Peut être remis à zéro après chaque prédiction.
"""

from __future__ import annotations

from typing import List, Tuple

from .utils import MovingAverage2D, filter_small_movement, now


class TrajectoryBuffer:
    def __init__(
        self,
        pause_threshold: float = 1.5,
        min_movement_px: float = 2.0,
        smoothing_window: int = 5,
    ) -> None:
        """
        Args:
            pause_threshold: secondes d'inactivité pour déclencher la fin d'un caractère.
            min_movement_px: déplacement minimal (pixels) pour enregistrer un nouveau point.
            smoothing_window: taille de la fenêtre de moyenne glissante.
        """
        self.pause_threshold = pause_threshold
        self.min_movement_px = min_movement_px
        self._points: List[Tuple[int, int]] = []
        self._last_update: float = now()
        self._smoother = MovingAverage2D(window=smoothing_window)

    # ------------------------------------------------------------------ API
    def add_point(self, point: Tuple[int, int]) -> None:
        """Ajoute un point à la trajectoire après lissage et filtrage du bruit."""
        smoothed = self._smoother.update(point)
        smoothed_int = (int(smoothed[0]), int(smoothed[1]))
        if filter_small_movement(self._points, smoothed_int, self.min_movement_px):
            self._points.append(smoothed_int)
            self._last_update = now()

    def mark_activity(self) -> None:
        """Met à jour l'horodatage sans ajouter de point (main détectée mais pas d'écriture)."""
        self._last_update = now()

    def reset(self) -> None:
        """Réinitialise la trajectoire et le lisseur."""
        self._points.clear()
        self._smoother.reset()
        self._last_update = now()

    def get_points(self) -> List[Tuple[int, int]]:
        return list(self._points)

    def is_empty(self) -> bool:
        return len(self._points) == 0

    def __len__(self) -> int:
        return len(self._points)

    def detect_pause(self) -> bool:
        """
        Renvoie True si :
         - il y a des points dans le buffer,
         - et le temps écoulé depuis le dernier point > pause_threshold.
        => Signal pour lancer l'inférence.
        """
        if self.is_empty():
            return False
        return (now() - self._last_update) > self.pause_threshold

    def time_since_update(self) -> float:
        return now() - self._last_update
