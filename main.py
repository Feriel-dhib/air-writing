"""
main.py
-------
Point d'entrée de l'écriture aérienne — chiffres 0-9 + lettres latines.

Pipeline :
 1. Capture webcam.
 2. Tracking main (MediaPipe) → position de l'index.
 3. Détection du geste d'écriture (index levé, autres doigts repliés).
 4. Accumulation de la trajectoire.
 5. Pause > seuil → rendu 28x28 → inférence CNN → texte.

Raccourcis clavier :
 - q        : quitter
 - c        : effacer le texte accumulé
 - espace   : ajouter un espace
 - b        : retour arrière
 - r        : reset de la trajectoire courante
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import cv2

from src.display import Overlay
from src.gesture import is_writing_gesture
from src.predictor import CharacterPredictor
from src.renderer import TrajectoryRenderer
from src.tracker import HandTracker
from src.trajectory import TrajectoryBuffer
from src.utils import FPSCounter


DEFAULT_MODEL = "model/model.h5"
DEFAULT_CAM = 0
PAUSE_THRESHOLD = 1.5
MIN_CONFIDENCE = 0.55
MIN_POINTS = 25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Air Writing — chiffres + latin")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Chemin du .h5 pré-entraîné")
    p.add_argument("--camera", type=int, default=DEFAULT_CAM, help="Index webcam")
    p.add_argument("--width", type=int, default=1280, help="Largeur capture")
    p.add_argument("--height", type=int, default=720, help="Hauteur capture")
    p.add_argument(
        "--pause",
        type=float,
        default=PAUSE_THRESHOLD,
        help="Seuil de pause (s) déclenchant la fin d'un caractère",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help="Confiance minimale acceptée (0-1)",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=MIN_POINTS,
        help="Nombre minimum de points de trajectoire avant inférence",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=7,
        help="Fenêtre de lissage de la trajectoire (défaut 7)",
    )
    p.add_argument(
        "--tta",
        action="store_true",
        help="Test-time augmentation : plus stable, ~7× plus lent",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("[Air Writing] Initialisation…")

    tracker = HandTracker()
    trajectory = TrajectoryBuffer(
        pause_threshold=args.pause,
        smoothing_window=max(3, args.smooth),
    )
    renderer = TrajectoryRenderer()

    try:
        predictor: Optional[CharacterPredictor] = CharacterPredictor(model_path=args.model)
    except (FileNotFoundError, ImportError) as e:
        print(f"[Erreur] {e}")
        return 1

    n = predictor.num_classes
    print(f"[Air Writing] Modèle chargé : {args.model} ({n} classes)")
    if n == 10:
        print("[Info] Modèle chiffres uniquement (MNIST). Pour chiffres + lettres : `python build_model_letters.py`.")

    fps = FPSCounter()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("[Erreur] Impossible d'ouvrir la webcam.")
        return 1

    accumulated_text = ""
    last_label: Optional[str] = None
    last_confidence: float = 0.0
    last_thumb = None
    last_topk: list[tuple[str, float]] = []

    print("[Air Writing] Appuyez sur 'q' pour quitter.")

    consecutive_read_errors = 0
    MAX_READ_ERRORS = 30

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_read_errors += 1
                if consecutive_read_errors >= MAX_READ_ERRORS:
                    print("[Erreur] Lecture webcam échouée de manière persistante.")
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            consecutive_read_errors = 0

            frame = cv2.flip(frame, 1)

            frame, index_px, landmarks = tracker.process(frame, draw_landmarks=True)

            writing = False
            if landmarks is not None:
                writing = is_writing_gesture(landmarks)

            if writing and index_px is not None:
                trajectory.add_point(index_px)

            if trajectory.detect_pause() and len(trajectory) >= args.min_points:
                img28 = renderer.render(trajectory.get_points())
                if img28 is not None:
                    label, conf, last_topk = predictor.predict_topk(
                        img28, tta=args.tta, k=3
                    )
                    last_label, last_confidence = label, conf
                    last_thumb = renderer.to_display(img28, size=140)
                    if conf >= args.min_confidence:
                        accumulated_text += label
                        print(f"[Prediction] {label} ({conf * 100:.1f}%) -> '{accumulated_text}'")
                    else:
                        print(f"[Prediction rejetée] {label} ({conf * 100:.1f}%)")
                trajectory.reset()
            elif trajectory.detect_pause():
                trajectory.reset()

            pause_progress = 0.0
            if not trajectory.is_empty():
                pause_progress = trajectory.time_since_update() / args.pause

            Overlay.draw_trajectory(frame, trajectory.get_points())
            Overlay.draw_fingertip(frame, index_px, writing)
            Overlay.draw_status_bar(frame, writing, fps.tick(), pause_progress)
            Overlay.draw_prediction(frame, last_label, last_confidence, topk=last_topk)
            Overlay.draw_accumulated_text(frame, accumulated_text)
            Overlay.draw_help(frame)
            Overlay.paste_thumbnail(frame, last_thumb)

            cv2.imshow("Air Writing", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                accumulated_text = ""
                last_label, last_confidence, last_thumb = None, 0.0, None
                last_topk = []
            elif key == ord(" "):
                accumulated_text += " "
            elif key == ord("b"):
                accumulated_text = accumulated_text[:-1]
            elif key == ord("r"):
                trajectory.reset()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

    print(f"[Air Writing] Texte final : '{accumulated_text}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
