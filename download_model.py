"""
download_model.py
-----------------
Télécharge un modèle MNIST CNN pré-entraîné (.h5) depuis une source publique
et le place dans `model/model.h5`.

IMPORTANT — Politique "zéro entraînement" :
Ce script NE RÉENTRAÎNE PAS de modèle. Il télécharge un fichier existant.

Sources supportées (par ordre) :
 1. URL personnalisée passée en argument : `python download_model.py --url <URL>`
 2. URL par défaut (modèle MNIST CNN public).

Si le téléchargement échoue, le script affiche des instructions pour placer
manuellement un .h5 dans `model/model.h5`. Plusieurs projets GitHub
fournissent un modèle compatible :

  - https://github.com/aryanvij02/MNIST-CNN
  - https://github.com/krishnaik06/Deep-Learning/tree/master/MNIST
  - https://github.com/Coopss/EMNIST       (pour lettres A-Z)

Tout modèle Keras dont :
  - l'entrée est de forme (None, 28, 28, 1) en float32 [0, 1]
  - la sortie est softmax (10, 26, 36 ou 47 classes)
fonctionnera avec `predictor.py` (mapping des labels automatique).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

try:
    import requests
except ImportError:
    print("[!] Le module 'requests' est requis. Installez : pip install requests")
    sys.exit(1)


DEFAULT_URLS: List[str] = [
    # Modèles CNN MNIST publics (.h5) — mis à jour régulièrement.
    "https://github.com/aryanvij02/MNIST-CNN/raw/master/mnist_cnn.h5",
    "https://huggingface.co/datasets/mnist-cnn/resolve/main/mnist_cnn.h5",
]

TARGET_PATH = os.path.join("model", "model.h5")


def download(url: str, dest: str) -> bool:
    try:
        print(f"[↓] Téléchargement depuis : {url}")
        with requests.get(url, stream=True, timeout=30, allow_redirects=True) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100 * done / total
                            sys.stdout.write(f"\r   {pct:5.1f}% ({done} / {total} octets)")
                            sys.stdout.flush()
            sys.stdout.write("\n")
        size = os.path.getsize(dest)
        if size < 1000:  # fichier suspicieusement petit -> probable page d'erreur
            os.remove(dest)
            print(f"[!] Fichier reçu trop petit ({size} octets). URL probablement invalide.")
            return False
        print(f"[✓] Modèle sauvegardé : {dest} ({size} octets)")
        return True
    except Exception as e:
        print(f"[!] Échec : {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Télécharge un CNN MNIST pré-entraîné.")
    parser.add_argument("--url", default=None, help="URL personnalisée vers un .h5")
    parser.add_argument(
        "--output", default=TARGET_PATH, help="Chemin de sortie (défaut : model/model.h5)"
    )
    args = parser.parse_args()

    urls = [args.url] if args.url else DEFAULT_URLS

    for url in urls:
        if download(url, args.output):
            return 0

    print("\n" + "=" * 70)
    print("Aucun téléchargement automatique n'a réussi.")
    print("Solution RECOMMANDÉE (fiable, ~1 min) :")
    print("   python build_model.py")
    print("   => génère localement model/model.h5 à partir du dataset MNIST")
    print("      intégré à Keras (accuracy > 99%).")
    print("\nOu placez manuellement un .h5 Keras compatible à :")
    print(f"   {os.path.abspath(args.output)}")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())
