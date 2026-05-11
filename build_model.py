"""
build_model.py
--------------
Crée localement un modèle CNN MNIST pré-entraîné et le sauvegarde dans
`model/model.h5`.

Pourquoi ce script ?
--------------------
Aucun hébergement public ne garantit une URL stable pour un fichier
`.h5` MNIST pré-entraîné. Plutôt que de dépendre d'un lien externe qui
peut disparaître, ce script :
 1. Télécharge le dataset MNIST (fourni nativement par Keras, ~11 Mo).
 2. Construit un petit CNN (2 blocs Conv+Pool + Dense).
 3. Entraîne rapidement (5 epochs, ~1 min sur CPU moderne, ~15 s sur GPU).
 4. Sauvegarde `model/model.h5` — directement consommable par `predictor.py`.

Ce script est une solution de "démarrage immédiat". Pour de la production,
remplacez `model/model.h5` par un modèle plus robuste (EMNIST, AHCD pour
l'arabe, etc.) — le reste du pipeline reste identique.
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    from tensorflow.keras.datasets import mnist  # type: ignore
    from tensorflow.keras.layers import (  # type: ignore
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        MaxPooling2D,
    )
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.utils import to_categorical  # type: ignore
except ImportError:
    print("[!] TensorFlow / Keras n'est pas installé. "
          "Installez les dépendances : pip install -r requirements.txt")
    sys.exit(1)


TARGET_PATH = os.path.join("model", "model.h5")
EPOCHS = 5
BATCH_SIZE = 128


def build_cnn() -> Sequential:
    """Petit CNN MNIST (~300k paramètres, > 99% d'accuracy après 5 epochs)."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> int:
    print("[1/4] Chargement du dataset MNIST…")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(f"[2/4] Données : train={x_train.shape}, test={x_test.shape}")

    print("[3/4] Construction et entraînement du CNN…")
    model = build_cnn()
    model.summary()
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        verbose=2,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"   Accuracy test : {acc * 100:.2f}%   Loss : {loss:.4f}")

    os.makedirs(os.path.dirname(TARGET_PATH), exist_ok=True)
    model.save(TARGET_PATH)
    print(f"[4/4] Modèle sauvegardé : {TARGET_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
