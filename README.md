# Onitama - Intelligence Artificielle

![Python 3](https://img.shields.io/badge/Python-3-3776AB.svg?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)
![Tests](https://img.shields.io/badge/Tests-Passants-brightgreen.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Université Paris Cité](https://img.shields.io/badge/Université-Paris%20Cité-red)](https://u-paris.fr/)

Ce projet est une implémentation du jeu de stratégie **Onitama** réalisée dans le cadre de l'UE Intelligence Artificielle à l'Université Paris Cité. Il propose un moteur de jeu complet et une IA basée sur l'algorithme **Minimax** optimisé par l'**élagage alpha-bêta**.
 
---

## Auteurs

- **TAKENNE MEKEM Simeon**
- **ZEROUALI Amine**

---

## Structure du projet

```
./
├── README.md                      # Instructions d'utilisation et présentation
├── onitama.py                     # Moteur de jeu + IA Minimax + menu CLI
├── generer_graphiques.py          # Script de benchmark et de génération de données
├── test_onitama.py                # Suite de tests unitaires
└── report/ 
    ├── rapport_onitama.pdf       # Rapport final
    └── rapport_onitama.tex  
        └── Image/                 # Graphiques générés pour le rapport
            ├── evolution_victoires.png
            ├── matrice_resultats.png
            ├── temps_calcul.png
            └── noeuds_explores.png
```
---

## Fonctionnalités

- **Moteur de jeu complet** respectant les règles officielles d'Onitama (plateau 5×5, 16 cartes, deux conditions de victoire).
- **Trois niveaux de difficulté d'IA** :
  | Niveau | Profondeur | Heuristique |
  |--------|-----------|-------------|
  | Facile | 3 | H1 - Matériel basique |
  | Moyen | 4 | H2 - Matériel dynamique |
  | Difficile | 5 | H3 - Matériel dynamique + contrôle du centre |
- **Cinq modes de jeu** accessibles depuis le menu interactif :
  1. Humain vs Humain
  2. Humain (Bleu) vs IA (Rouge)
  3. IA (Bleu) vs Humain (Rouge)
  4. IA vs IA - Mode Spectateur
  5. Tournoi automatisé IA vs IA (avec statistiques)
- **Tournoi parallélisé** via `multiprocessing` pour les analyses statistiques (mode 5 uniquement).
- **Suite de tests unitaires** couvrant l'état initial, la détection de victoire et la réversibilité des coups (indispensable au Minimax).

---

## Installation et Exécution

Le projet est implémenté en Python 3. Il utilise la bibliothèque standard, `Matplotlib` et `NumPy` pour la génération des graphiques.

### Prérequis

```bash
pip install matplotlib numpy
```

### Lancer le jeu

```bash
python onitama.py
```

Le menu interactif s'affiche et propose les 5 modes décrits ci-dessus. Pour les modes impliquant une IA, le niveau de difficulté (Facile / Moyen / Difficile) est demandé à chaque joueur. En mode Humain, les coups légaux sont listés et numérotés ; il suffit d'entrer le numéro correspondant.

### Lancer les Benchmarks

```bash
python generer_graphiques.py
```

Génère les données et graphiques d'analyse : temps de calcul moyen, nœuds explorés selon la profondeur, et résultats de tournois entre les différents niveaux d'IA.

### Lancer les Tests

```bash
python test_onitama.py
```

Ou via le module `unittest` :

```bash
python -m unittest test_onitama
```

---

## Conventions internes

Le plateau est un tableau `5×5` avec la convention `plateau[y][x]` (`y=0` en haut). Les pièces sont encodées comme suit :

| Valeur | Pièce |
|--------|-------|
| `0` | Case vide |
| `1` | Élève Bleu (J1, rangée du haut) |
| `2` | Maître Bleu |
| `-1` | Élève Rouge (J2, rangée du bas) |
| `-2` | Maître Rouge |

Les temples (arches) se trouvent en `(x=2, y=0)` pour Bleu et `(x=2, y=4)` pour Rouge.

---

## Analyse expérimentale

Le rapport **`rapport_onitama.pdf`** détaille l'efficacité de l'élagage alpha-bêta (réduction du nombre de nœuds explorés) ainsi que les résultats des tournois entre les trois niveaux d'IA.
