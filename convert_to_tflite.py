"""
convert_to_tflite.py
--------------------
Convertit un modèle Keras `.h5` en modèle TensorFlow Lite `.tflite`.

Exemples :
  python convert_to_tflite.py
  python convert_to_tflite.py --input model/model.h5 --output model/model.tflite
  python convert_to_tflite.py --quantize
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import tensorflow as tf
except ImportError:
    print("[Erreur] TensorFlow n'est pas installé. Lancez : pip install -r requirements.txt")
    sys.exit(1)


DEFAULT_INPUT = "model/model.h5"
DEFAULT_OUTPUT = "model/model.tflite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversion Keras (.h5) vers TensorFlow Lite (.tflite)")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Chemin du modèle Keras .h5")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Chemin de sortie du fichier .tflite")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Applique une quantification dynamique (taille plus petite, impact précision possible)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[Erreur] Modèle introuvable : {args.input}")
        return 1

    print(f"[1/3] Chargement du modèle : {args.input}")
    model = tf.keras.models.load_model(args.input, compile=False)

    print("[2/3] Conversion en TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"[3/3] Modèle TFLite sauvegardé : {args.output} ({size_kb:.1f} Ko)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
