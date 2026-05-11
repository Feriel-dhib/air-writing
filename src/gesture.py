"""
gesture.py
----------
Détecte le geste d'écriture : index levé (tendu), les autres doigts repliés.

Règle simple et robuste basée sur les landmarks MediaPipe :
 - Pour chaque doigt (sauf pouce) : tip.y < pip.y  => doigt levé
   (le repère MediaPipe : y augmente vers le bas).
 - Le pouce est traité séparément (comparaison X) car il plie latéralement.
"""

from __future__ import annotations

from typing import List, Tuple

# Indices des landmarks pertinents
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18


def _finger_up(tip_y: float, pip_y: float, margin: float = 0.02) -> bool:
    """Un doigt est 'levé' si la pointe est plus haute (y plus petit) que le PIP."""
    return tip_y < pip_y - margin


def is_writing_gesture(landmarks: List[Tuple[float, float, float]]) -> bool:
    """
    Retourne True si la configuration de la main correspond au geste d'écriture :
        index levé + majeur/annulaire/auriculaire repliés.
    Le pouce est ignoré (souvent ambigu).
    """
    if landmarks is None or len(landmarks) < 21:
        return False

    index_up = _finger_up(landmarks[INDEX_TIP][1], landmarks[INDEX_PIP][1])
    middle_down = not _finger_up(landmarks[MIDDLE_TIP][1], landmarks[MIDDLE_PIP][1])
    ring_down = not _finger_up(landmarks[RING_TIP][1], landmarks[RING_PIP][1])
    pinky_down = not _finger_up(landmarks[PINKY_TIP][1], landmarks[PINKY_PIP][1])

    return bool(index_up and middle_down and ring_down and pinky_down)


def fingers_state(landmarks: List[Tuple[float, float, float]]) -> dict:
    """Debug utilitaire : renvoie l'état de chaque doigt."""
    if landmarks is None or len(landmarks) < 21:
        return {}
    return {
        "index": _finger_up(landmarks[INDEX_TIP][1], landmarks[INDEX_PIP][1]),
        "middle": _finger_up(landmarks[MIDDLE_TIP][1], landmarks[MIDDLE_PIP][1]),
        "ring": _finger_up(landmarks[RING_TIP][1], landmarks[RING_PIP][1]),
        "pinky": _finger_up(landmarks[PINKY_TIP][1], landmarks[PINKY_PIP][1]),
    }
