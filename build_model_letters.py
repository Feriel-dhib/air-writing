"""
build_model_letters.py
----------------------
Construit un modèle CNN pré-entraîné sur **EMNIST Balanced** (47 classes) :
chiffres 0-9 + lettres majuscules A-Z + 11 lettres minuscules ambiguës
(a, b, d, e, f, g, h, n, q, r, t).

Téléchargement direct depuis la source officielle NIST
(https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip)
→ pas de dépendance sur Google Drive (qui casse le package `emnist`).

Durée :
 - Téléchargement : ~2-5 min selon la connexion (562 Mo, mis en cache).
 - Entraînement : ~3-5 min sur Apple Silicon CPU/GPU.
Accuracy test attendue : ~87-90%.
"""

from __future__ import annotations

import gzip
import os
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

try:
    from tensorflow.keras.callbacks import ReduceLROnPlateau  # type: ignore
    from tensorflow.keras.layers import (  # type: ignore
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        MaxPooling2D,
    )
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.utils import to_categorical  # type: ignore
except ImportError:
    print("[!] TensorFlow / Keras n'est pas installé.")
    sys.exit(1)


# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------
TARGET_PATH = os.path.join("model", "model.h5")
NUM_CLASSES = 47
EPOCHS = 10
BATCH_SIZE = 128

EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
CACHE_DIR = Path.home() / ".cache" / "emnist_nist"
ZIP_PATH = CACHE_DIR / "gzip.zip"

FILES = {
    "train_images": "gzip/emnist-balanced-train-images-idx3-ubyte.gz",
    "train_labels": "gzip/emnist-balanced-train-labels-idx1-ubyte.gz",
    "test_images": "gzip/emnist-balanced-test-images-idx3-ubyte.gz",
    "test_labels": "gzip/emnist-balanced-test-labels-idx1-ubyte.gz",
}


# ----------------------------------------------------------------------
# Téléchargement + extraction
# ----------------------------------------------------------------------
def _download_with_progress(url: str, dest: Path) -> None:
    """Téléchargement HTTP avec barre de progression.
    Utilise un User-Agent car NIST bloque les clients Python par défaut (HTTP 403).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"     URL : {url}")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024  # 1 Mo
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total_size > 0:
                    pct = 100 * done / total_size
                    sys.stdout.write(
                        f"\r     {pct:5.1f}%  "
                        f"{done / (1024 * 1024):6.1f} / "
                        f"{total_size / (1024 * 1024):6.1f} Mo"
                    )
                    sys.stdout.flush()
    sys.stdout.write("\n")


def _parse_idx_images(gz_bytes: bytes) -> np.ndarray:
    """Parse un fichier IDX d'images (magic 0x00000803)."""
    raw = gzip.decompress(gz_bytes)
    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"Magic number images invalide : {magic}")
    data = np.frombuffer(raw[16:], dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _parse_idx_labels(gz_bytes: bytes) -> np.ndarray:
    """Parse un fichier IDX de labels (magic 0x00000801)."""
    raw = gzip.decompress(gz_bytes)
    magic, n = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"Magic number labels invalide : {magic}")
    return np.frombuffer(raw[8:], dtype=np.uint8)


def _load_emnist_balanced() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Télécharge (si nécessaire) et charge EMNIST Balanced en numpy."""
    if not ZIP_PATH.exists():
        print(f"[1/4] Téléchargement d'EMNIST (~562 Mo) → {ZIP_PATH}")
        _download_with_progress(EMNIST_URL, ZIP_PATH)
    else:
        print(f"[1/4] Fichier déjà en cache : {ZIP_PATH}")

    print("     Extraction des 4 fichiers 'balanced'…")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = {k: zf.read(v) for k, v in FILES.items()}

    x_train = _parse_idx_images(members["train_images"])
    y_train = _parse_idx_labels(members["train_labels"])
    x_test = _parse_idx_images(members["test_images"])
    y_test = _parse_idx_labels(members["test_labels"])

    # Correction d'orientation EMNIST : les images officielles sont stockées
    # transposées par rapport à MNIST (axes X/Y inversés). On remet à l'endroit.
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------
# Modèle
# ----------------------------------------------------------------------
def build_cnn() -> Sequential:
    """CNN un peu plus profond que pour MNIST (EMNIST Balanced est plus difficile)."""
    model = Sequential([
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------------------------------
# Entraînement
# ----------------------------------------------------------------------
def main() -> int:
    x_train, y_train, x_test, y_test = _load_emnist_balanced()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train_oh = to_categorical(y_train, NUM_CLASSES)
    y_test_oh = to_categorical(y_test, NUM_CLASSES)

    print(f"[2/4] Données : train={x_train.shape}, test={x_test.shape}")
    print(f"     Classes uniques : {len(np.unique(y_train))}")

    print("[3/4] Construction et entraînement du CNN…")
    model = build_cnn()
    model.summary()

    lr_cb = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
    )

    model.fit(
        x_train,
        y_train_oh,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test_oh),
        callbacks=[lr_cb],
        verbose=2,
    )

    loss, acc = model.evaluate(x_test, y_test_oh, verbose=0)
    print(f"   Accuracy test : {acc * 100:.2f}%   Loss : {loss:.4f}")

    os.makedirs(os.path.dirname(TARGET_PATH), exist_ok=True)
    model.save(TARGET_PATH)
    print(f"[4/4] Modèle sauvegardé : {TARGET_PATH}")
    print("      (47 classes ; predictor.py mappera automatiquement 0-9, A-Z, minuscules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
