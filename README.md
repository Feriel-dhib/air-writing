# Air Writing — chiffres + lettres latines

Reconnaissance de caractères écrits **dans l'air** via la webcam : on lève
l'index, on trace un caractère, on marque une pause, le système affiche le
caractère reconnu et l'empile dans un texte.

Le projet est volontairement **simple et focalisé** :

- **Chiffres 0-9** (modèle MNIST).
- **Lettres latines A-Z** + 11 minuscules ambigües (modèle EMNIST Balanced,
  47 classes).

La partie arabe a été retirée : les résultats en écriture aérienne n'étaient
pas fiables (trop grand fossé entre les datasets sur papier et les gestes
filmés à la webcam).

---

## 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Le modèle MediaPipe (`model/hand_landmarker.task`) est déjà présent dans le
repo.

---

## 2. Choix du modèle

### Option A — Chiffres uniquement (rapide, déjà prêt)

Le fichier `model/model.h5` fourni est entraîné sur MNIST (10 classes,
chiffres 0-9). Aucun entraînement supplémentaire n'est nécessaire.

### Option B — Chiffres + lettres latines (recommandé)

Entraîner un modèle 47 classes sur EMNIST Balanced (écrase `model/model.h5`) :

```bash
python build_model_letters.py
```

- Télécharge EMNIST (~562 Mo, mis en cache dans `~/.cache/emnist_nist`).
- Entraînement : ~3-5 min sur Apple Silicon.
- Accuracy test attendue : ~87-90 %.

> Pour revenir aux chiffres seuls, relancer `python build_model.py`.

---

## 3. Lancer la démo

```bash
python main.py
```

Options utiles :

```bash
python main.py --tta                      # test-time augmentation (plus stable)
python main.py --min-confidence 0.6       # rejet plus strict
python main.py --pause 1.2               # pause plus courte avant de valider
python main.py --smooth 9                # lissage de trajectoire (défaut 7)
python main.py --model model/model.h5    # chemin modèle explicite
python main.py --camera 1               # index webcam (défaut 0)
python main.py --width 1920 --height 1080 # résolution capture
```

### Raccourcis clavier (dans la fenêtre OpenCV)

| Touche | Action |
|--------|--------|
| `q`    | Quitter |
| `c`    | Effacer le texte accumulé |
| `espace` | Ajouter un espace |
| `b`    | Retour arrière |
| `r`    | Réinitialiser la trajectoire en cours |

---

## 4. Conseils pour de bons résultats

1. **Bon éclairage**, main bien visible, caméra à hauteur de poitrine.
2. **Un seul doigt levé** (l'index). Les autres doigts repliés déclenchent le
   geste d'écriture.
3. **Geste grand et net** : trajectoire d'au moins ~20 cm à l'écran.
4. **Forme proche de l'imprimé** : le modèle est entraîné sur des caractères
   d'imprimerie, pas sur de la cursive.
5. **Pause claire** (~1,5 s par défaut) à la fin de chaque caractère pour
   déclencher la reconnaissance.
6. Si les prédictions sautent : activer `--tta` et/ou augmenter
   `--min-confidence` à `0.65`.

---

## 5. Conversion TFLite (optionnelle, pour mobile)

```bash
python convert_to_tflite.py --input model/model.h5 --output model/model.tflite --quantize
```

---

## 6. Structure du projet

```
air_writing/
├── main.py                  # Boucle caméra + inférence
├── build_model.py           # Entraînement MNIST (10 classes)
├── build_model_letters.py   # Entraînement EMNIST Balanced (47 classes)
├── convert_to_tflite.py     # Export .tflite
├── download_model.py        # Téléchargement de modèles pré-entraînés
├── requirements.txt
├── data/                    # Données de démo (manifests)
├── model/
│   ├── hand_landmarker.task # MediaPipe Hands
│   ├── model.h5             # CNN actif (Keras)
│   └── model.tflite         # Version TFLite (optionnelle)
└── src/
    ├── __init__.py          # Package Python
    ├── display.py           # Overlays OpenCV
    ├── gesture.py           # Détection « doigt levé »
    ├── predictor.py         # Chargement modèle + inférence
    ├── renderer.py          # Trajectoire → image 28×28
    ├── tracker.py           # MediaPipe HandLandmarker
    ├── trajectory.py        # Buffer de points + détection de pause
    └── utils.py             # FPS, helpers
```

---

## 7. Limitations connues

- L'écriture aérienne ne reproduit pas exactement les caractères d'EMNIST
  (datasets papier). Certaines lettres similaires se confondent (O/0, I/1/l,
  S/5…). L'option `--tta` et des gestes nets et grands améliorent nettement
  la robustesse.
- Le modèle EMNIST Balanced mélange des minuscules et majuscules : par
  exemple, `I` et `l` peuvent être prédits l'un pour l'autre.
# air-writing
