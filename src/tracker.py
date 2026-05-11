"""
tracker.py
----------
Encapsule la détection de main MediaPipe et renvoie :
 - la frame (éventuellement annotée),
 - la position (x, y) du bout de l'index (landmark #8) en pixels,
 - la liste complète des 21 landmarks (coordonnées normalisées).

Utilise la **nouvelle API `mediapipe.tasks.vision.HandLandmarker`** (la
seule disponible dans mediapipe >= 0.10.20 sur Apple Silicon — l'API
legacy `mp.solutions` n'est plus exposée). Le modèle `hand_landmarker.task`
(~7 Mo) est téléchargé automatiquement au premier lancement.
"""

from __future__ import annotations

import os
import urllib.request
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# Landmark index du bout de l'index
INDEX_TIP = 8

# URL officielle du modèle MediaPipe HandLandmarker (float16, ~7 Mo)
HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_TASK_PATH = os.path.join("model", "hand_landmarker.task")

# Connexions entre landmarks (identiques à mp.solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # pouce
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # majeur
    (9, 13), (13, 14), (14, 15), (15, 16),     # annulaire
    (13, 17), (17, 18), (18, 19), (19, 20),    # auriculaire
    (0, 17),                                   # base
]


def _ensure_model(path: str = HAND_TASK_PATH, url: str = HAND_TASK_URL) -> str:
    """Télécharge le fichier .task s'il n'existe pas localement."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[HandTracker] Téléchargement du modèle MediaPipe : {url}")
    urllib.request.urlretrieve(url, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"[HandTracker] Modèle sauvegardé : {path} ({size_kb:.0f} Ko)")
    return path


def _draw_landmarks(
    frame_bgr: np.ndarray,
    landmarks_px: List[Tuple[int, int]],
) -> None:
    """Dessine les 21 landmarks et leurs connexions (remplace drawing_utils)."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, landmarks_px[a], landmarks_px[b], (0, 255, 0), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(landmarks_px):
        color = (0, 0, 255) if i == INDEX_TIP else (255, 255, 255)
        cv2.circle(frame_bgr, (x, y), 4, color, -1, cv2.LINE_AA)


class HandTracker:
    """Détection temps-réel d'une main via MediaPipe HandLandmarker."""

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ) -> None:
        path = _ensure_model(model_path or HAND_TASK_PATH)

        base_options = mp_python.BaseOptions(model_asset_path=path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_index = 0  # sert d'horodatage (ms) pour le mode VIDEO

    # ------------------------------------------------------------------
    def process(
        self,
        frame_bgr: np.ndarray,
        draw_landmarks: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]], Optional[List[Tuple[float, float, float]]]]:
        """
        Traite une frame BGR et retourne :
          (frame_annotée, point_index_px, landmarks_normalisés)
        Si aucune main n'est détectée -> (frame, None, None).
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Horodatage monotone croissant (ms) — requis en mode VIDEO
        self._frame_index += 1
        timestamp_ms = self._frame_index * 33  # ~30 FPS

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return frame_bgr, None, None

        hand_landmarks = result.hand_landmarks[0]  # une seule main

        landmarks_norm: List[Tuple[float, float, float]] = [
            (lm.x, lm.y, lm.z) for lm in hand_landmarks
        ]
        landmarks_px: List[Tuple[int, int]] = [
            (int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks
        ]

        if draw_landmarks:
            _draw_landmarks(frame_bgr, landmarks_px)

        index_px = landmarks_px[INDEX_TIP]
        return frame_bgr, index_px, landmarks_norm

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
