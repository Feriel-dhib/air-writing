"""
renderer.py
-----------
Convertit une trajectoire (liste de points 2D en pixels) en une image
28x28 en niveaux de gris au format compatible MNIST.

Étapes :
 1. Calculer la bounding box de la trajectoire.
 2. Dessiner la trajectoire (cv2.line) sur un canvas haute résolution.
 3. Centrer + adapter l'échelle pour tenir dans une boîte de 20x20
    (comme MNIST, qui centre les chiffres dans une fenêtre de 20px
    à l'intérieur d'une image 28x28).
 4. Flou gaussien pour épaissir et lisser le trait.
 5. Redimensionner en 28x28, normaliser en [0, 1].
 6. Recentrage par **centre de masse** du tracé (mieux aligné sur MNIST/EMNIST).
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


# Paramètres inspirés du prétraitement MNIST
TARGET_SIZE = 28
INNER_BOX = 20           # la trajectoire est centrée dans 20x20 puis padding -> 28x28
WORK_CANVAS = 280        # canvas intermédiaire (haute résolution pour un trait net)
STROKE_THICKNESS = 18    # épaisseur du trait sur le canvas haute résolution
BLUR_KERNEL = (9, 9)


def _recenter_centroid(img28: np.ndarray) -> np.ndarray:
    """
    Décale l'image pour que le centre de masse du tracé tombe au centre 28×28.
    Réduit la sensibilité au bruit de trajectoire / à un léger biais spatial.
    """
    t = float(img28.sum())
    if t < 1e-4:
        return img28
    yy, xx = np.mgrid[0:TARGET_SIZE, 0:TARGET_SIZE].astype(np.float32)
    cx = float((xx * img28).sum() / t)
    cy = float((yy * img28).sum() / t)
    dx = (TARGET_SIZE - 1) / 2.0 - cx
    dy = (TARGET_SIZE - 1) / 2.0 - cy
    u = np.clip(img28, 0.0, 1.0)
    u8 = (u * 255.0).astype(np.uint8)
    m = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    shifted = cv2.warpAffine(
        u8,
        m,
        (TARGET_SIZE, TARGET_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted.astype(np.float32) / 255.0


class TrajectoryRenderer:
    """Transforme une trajectoire brute en image 28x28 prête pour le CNN."""

    def render(self, points: List[Tuple[int, int]]) -> np.ndarray | None:
        """
        Renvoie une image (28, 28) float32 normalisée dans [0, 1],
        ou None si la trajectoire est trop courte.
        """
        if points is None or len(points) < 2:
            return None

        pts = np.asarray(points, dtype=np.float32)

        # 1. Bounding box
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)

        # Si c'est vraiment un point, on ignore
        if width < 5 and height < 5:
            return None

        # 2. Normalisation dans [0, INNER_BOX] tout en conservant le ratio
        scale = (INNER_BOX - 1) / max(width, height)
        norm_pts = (pts - np.array([x_min, y_min], dtype=np.float32)) * scale

        # Offset pour centrer dans INNER_BOX puis dans TARGET_SIZE
        bbox_w = width * scale
        bbox_h = height * scale
        offset_x = (INNER_BOX - bbox_w) / 2.0 + (TARGET_SIZE - INNER_BOX) / 2.0
        offset_y = (INNER_BOX - bbox_h) / 2.0 + (TARGET_SIZE - INNER_BOX) / 2.0

        # On passe d'abord par un canvas haute résolution pour un trait plus propre,
        # donc on remet à l'échelle WORK_CANVAS.
        hi_scale = WORK_CANVAS / TARGET_SIZE
        hi_pts = (norm_pts + np.array([offset_x, offset_y], dtype=np.float32)) * hi_scale
        hi_pts = hi_pts.astype(np.int32)

        canvas = np.zeros((WORK_CANVAS, WORK_CANVAS), dtype=np.uint8)
        for i in range(1, len(hi_pts)):
            cv2.line(
                canvas,
                tuple(hi_pts[i - 1]),
                tuple(hi_pts[i]),
                color=255,
                thickness=STROKE_THICKNESS,
                lineType=cv2.LINE_AA,
            )

        # 3. Flou gaussien pour adoucir les bords (se rapproche du style MNIST)
        canvas = cv2.GaussianBlur(canvas, BLUR_KERNEL, 0)

        # 4. Redimensionnement 28x28
        img28 = cv2.resize(canvas, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        # 5. Normalisation [0, 1]
        img28 = img28.astype(np.float32) / 255.0
        # 6. Centrage type MNIST (centre de masse)
        img28 = _recenter_centroid(img28)
        return img28

    @staticmethod
    def to_display(img28: np.ndarray, size: int = 140) -> np.ndarray:
        """Version agrandie pour l'affichage (debug)."""
        disp = (img28 * 255).astype(np.uint8)
        disp = cv2.resize(disp, (size, size), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
