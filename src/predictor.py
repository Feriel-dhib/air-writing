"""
predictor.py
------------
Chargement d'un modèle Keras .h5 pré-entraîné et inférence sur une image 28x28.

Supporte trois tailles de sortie (détection automatique du mapping) :
 - 10 classes : MNIST (chiffres 0-9).
 - 26 classes : EMNIST Letters (A-Z).
 - 36 classes : chiffres + A-Z concaténés.
 - 47 classes : EMNIST Balanced (chiffres + A-Z + 11 minuscules ambigües).

Entrée attendue : (1, 28, 28, 1) en float32 normalisé [0, 1].
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from tensorflow.keras.models import load_model  # type: ignore
except Exception:  # pragma: no cover
    load_model = None  # type: ignore


DIGITS = [str(i) for i in range(10)]
LETTERS = [chr(ord("A") + i) for i in range(26)]
# EMNIST Balanced (47 classes) : 0-9, A-Z, + 11 lettres minuscules ambigües.
EMNIST_BALANCED: List[str] = (
    DIGITS
    + LETTERS
    + ["a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"]
)


def _shift_img28(img28: np.ndarray, dx: float, dy: float) -> np.ndarray:
    u8 = (np.clip(img28, 0.0, 1.0) * 255.0).astype(np.uint8)
    m = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    out = cv2.warpAffine(
        u8,
        m,
        (28, 28),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return out.astype(np.float32) / 255.0


class CharacterPredictor:
    """Chargeur + inférence pour un CNN de classification de caractères."""

    def __init__(
        self,
        model_path: str = "model/model.h5",
        labels: Optional[List[str]] = None,
    ) -> None:
        if load_model is None:
            raise ImportError(
                "TensorFlow/Keras n'est pas installé. "
                "Installez les dépendances via `pip install -r requirements.txt`."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modèle introuvable : '{model_path}'. "
                "Placez votre .h5 pré-entraîné à cet emplacement "
                "ou exécutez `python download_model.py`."
            )
        self.model = load_model(model_path, compile=False)
        self.labels = labels or self._auto_labels()
        try:
            self._num_classes = int(self.model.output_shape[-1])
        except Exception:
            self._num_classes = len(self.labels)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _auto_labels(self) -> List[str]:
        """Déduit les labels à partir du nombre de sorties du modèle."""
        try:
            num_classes = int(self.model.output_shape[-1])
        except Exception:
            num_classes = 10
        if num_classes == 10:
            return DIGITS
        if num_classes == 26:
            return LETTERS
        if num_classes == 36:
            return DIGITS + LETTERS
        if num_classes == 47:
            return list(EMNIST_BALANCED)
        return [str(i) for i in range(num_classes)]

    def _raw_probabilities(self, img28: np.ndarray, tta: bool) -> np.ndarray:
        base = img28.astype(np.float32)
        variants = [base]
        if tta:
            variants.extend(
                [
                    np.clip(base * 0.9, 0.0, 1.0),
                    np.clip(base * 1.1, 0.0, 1.0),
                    _shift_img28(base, 0.5, 0.0),
                    _shift_img28(base, -0.5, 0.0),
                    _shift_img28(base, 0.0, 0.5),
                    _shift_img28(base, 0.0, -0.5),
                ]
            )
        batch = np.stack(variants, axis=0)[..., np.newaxis]
        preds = self.model.predict(batch, verbose=0)
        probs = np.mean(preds, axis=0).astype(np.float64)
        s = float(probs.sum())
        if s > 0.0:
            probs = probs / s
        return probs

    def predict(self, img28: np.ndarray, tta: bool = False) -> Tuple[str, float]:
        """Prédit le caractère depuis une image 28x28 float32 [0, 1]."""
        if img28 is None:
            return "?", 0.0
        if img28.shape != (28, 28):
            raise ValueError(f"Shape attendue (28,28), reçu {img28.shape}")
        probs = self._raw_probabilities(img28, tta)
        idx = int(np.argmax(probs))
        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        return label, float(probs[idx])

    def predict_topk(
        self,
        img28: np.ndarray,
        tta: bool = False,
        k: int = 3,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Comme ``predict`` + les *k* meilleures paires (label, probabilité)."""
        if img28 is None:
            return "?", 0.0, []
        probs = self._raw_probabilities(img28, tta)
        k = max(1, min(int(k), len(probs)))
        order = np.argsort(-probs)[:k]
        ranked: List[Tuple[str, float]] = []
        for i in order:
            lb = self.labels[int(i)] if int(i) < len(self.labels) else str(int(i))
            ranked.append((lb, float(probs[int(i)])))
        best_l, best_p = ranked[0]
        return best_l, best_p, ranked
