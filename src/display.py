"""
display.py
----------
Superposition graphique sur le flux caméra :
 - Trajectoire en cours
 - Dernier caractère prédit + confiance
 - Texte accumulé
 - FPS
 - Indicateurs d'état (écriture, pause, prêt)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


FONT = cv2.FONT_HERSHEY_SIMPLEX

# Couleurs (BGR)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (255, 0, 180)


class Overlay:
    """Regroupe toutes les fonctions d'affichage."""

    @staticmethod
    def draw_trajectory(
        frame: np.ndarray,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int] = CYAN,
        thickness: int = 4,
    ) -> None:
        if points is None or len(points) < 2:
            return
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, thickness, cv2.LINE_AA)

    @staticmethod
    def _put_text_with_bg(
        frame: np.ndarray,
        text: str,
        org: Tuple[int, int],
        scale: float = 0.7,
        color: Tuple[int, int, int] = WHITE,
        thickness: int = 2,
        bg: Tuple[int, int, int] = BLACK,
        padding: int = 6,
    ) -> None:
        (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
        x, y = org
        cv2.rectangle(
            frame,
            (x - padding, y - th - padding),
            (x + tw + padding, y + baseline + padding),
            bg,
            -1,
        )
        cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_status_bar(
        frame: np.ndarray,
        is_writing: bool,
        fps: float,
        pause_progress: float = 0.0,
    ) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
        state_text = "ECRITURE" if is_writing else "ATTENTE"
        state_color = GREEN if is_writing else YELLOW
        cv2.putText(frame, state_text, (10, 28), FONT, 0.7, state_color, 2, cv2.LINE_AA)
        cv2.putText(
            frame, f"FPS: {fps:5.1f}", (w - 130, 28), FONT, 0.7, WHITE, 2, cv2.LINE_AA
        )

        # Barre de progression de la pause (fin de caractère)
        if pause_progress > 0.0:
            bar_w = int(min(pause_progress, 1.0) * (w - 20))
            cv2.rectangle(frame, (10, 45), (10 + bar_w, 55), PURPLE, -1)
            cv2.rectangle(frame, (10, 45), (w - 10, 55), WHITE, 1)

    @staticmethod
    def draw_prediction(
        frame: np.ndarray,
        label: Optional[str],
        confidence: float,
        topk: Optional[Sequence[Tuple[str, float]]] = None,
    ) -> None:
        if label is None:
            return
        h = frame.shape[0]
        text = f"Caractere: {label}  ({confidence * 100:.1f}%)"
        Overlay._put_text_with_bg(frame, text, (10, h - 88), scale=0.75)
        if topk and len(topk) > 1:
            parts = [f"{lb} {p * 100:.0f}%" for lb, p in topk[1:4]]
            line = "Alt: " + " | ".join(parts)
            Overlay._put_text_with_bg(
                frame, line, (10, h - 58), scale=0.55, color=YELLOW, bg=(40, 40, 40)
            )

    @staticmethod
    def draw_accumulated_text(frame: np.ndarray, text: str) -> None:
        display = text if text else "(texte vide)"
        disp_line = display[:80] + ("…" if len(display) > 80 else "")
        Overlay._put_text_with_bg(
            frame,
            f"Texte: {disp_line}",
            (10, frame.shape[0] - 20),
            scale=0.9,
            color=GREEN,
            thickness=2,
        )

    @staticmethod
    def draw_help(frame: np.ndarray) -> None:
        h = frame.shape[0]
        line = "q: quitter  |  c: effacer  |  espace  |  b: retour  |  r: reset"
        Overlay._put_text_with_bg(
            frame, line, (10, h - 130), scale=0.5, color=WHITE, bg=(40, 40, 40)
        )

    @staticmethod
    def draw_fingertip(
        frame: np.ndarray, point: Optional[Tuple[int, int]], is_writing: bool
    ) -> None:
        if point is None:
            return
        color = GREEN if is_writing else RED
        cv2.circle(frame, point, 8, color, -1, cv2.LINE_AA)
        cv2.circle(frame, point, 14, color, 2, cv2.LINE_AA)

    @staticmethod
    def paste_thumbnail(frame: np.ndarray, thumb: np.ndarray, margin: int = 10) -> None:
        """Colle une vignette (ex: image 28x28 agrandie) en haut-droite."""
        if thumb is None:
            return
        h, w = frame.shape[:2]
        th, tw = thumb.shape[:2]
        x0 = w - tw - margin
        y0 = 50
        if y0 + th <= h and x0 >= 0:
            frame[y0 : y0 + th, x0 : x0 + tw] = thumb
            cv2.rectangle(frame, (x0, y0), (x0 + tw, y0 + th), WHITE, 1)
