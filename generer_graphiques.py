# -*- coding: utf-8 -*-
"""
generer_graphiques.py
Script de benchmark pour le projet Onitama.

Produit :
  - matrice_resultats.png   : matrice de chaleur 3×3 du tournoi
  - evolution_victoires.png : courbe du taux de victoire cumulé
  - temps_calcul.png        : temps moyen par coup selon la profondeur
  - noeuds_explores.png     : noeuds explorés avec / sans élagage α-β
  - (stdout)                : tableaux numériques complets

Ce script est pensé comme un outil “offline” : il n'affiche rien à l'écran
(`matplotlib.use("Agg")`) et écrit les résultats sous forme d'images.
"""

import time
import random
import copy
import statistics
import concurrent.futures

import matplotlib
matplotlib.use("Agg")          # pas d'affichage graphique
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from onitama import Onitama, PROFILS_IA, lancer_partie

# ─────────────────────────────────────────────────────────────
# INSTRUMENTS DE MESURE
# ─────────────────────────────────────────────────────────────

class OnitamaBench(Onitama):
    """
    Sous-classe qui ajoute :
      - self.node_count          : compteur de nœuds (remis à 0 avant chaque appel)
      - minimax_sans_elagage()   : version Minimax pure (sans coupures α-β)
    """

    def __init__(self):
        super().__init__()
        self.node_count = 0

    # ── version AVEC élagage (surcharge pour compter les nœuds) ──

    def max_value(self, profondeur, alpha, beta, heuristique=3):
        # On compte le nombre d'appels récursifs dans l'arbre de recherche α‑β.
        # Cela donne une mesure de “taille d'arbre explorée” comparable entre positions.
        self.node_count += 1
        return super().max_value(profondeur, alpha, beta, heuristique)

    def min_value(self, profondeur, alpha, beta, heuristique=3):
        # Même principe pour les nœuds MIN.
        self.node_count += 1
        return super().min_value(profondeur, alpha, beta, heuristique)

    # ── version SANS élagage ──

    def max_value_no_ab(self, profondeur, heuristique=3):
        """
        Variante MAX sans α‑β : sert uniquement de référence pour quantifier le gain
        apporté par l'élagage.
        """
        self.node_count += 1
        gagnant = self.verifier_victoire()
        if gagnant is not None:
            return (10000 + profondeur) if gagnant == 1 else (-10000 - profondeur)
        if profondeur == 0:
            return self._eval(heuristique)
        v = float('-inf')
        for coup in self.obtenir_coups_possibles(1):
            pc, ae = self.jouer_coup(coup)
            v = max(v, self.min_value_no_ab(profondeur - 1, heuristique))
            self.annuler_coup(coup, pc, ae)
        return v

    def min_value_no_ab(self, profondeur, heuristique=3):
        """Variante MIN sans α‑β (voir `max_value_no_ab`)."""
        self.node_count += 1
        gagnant = self.verifier_victoire()
        if gagnant is not None:
            return (10000 + profondeur) if gagnant == 1 else (-10000 - profondeur)
        if profondeur == 0:
            return self._eval(heuristique)
        v = float('inf')
        for coup in self.obtenir_coups_possibles(-1):
            pc, ae = self.jouer_coup(coup)
            v = min(v, self.max_value_no_ab(profondeur - 1, heuristique))
            self.annuler_coup(coup, pc, ae)
        return v

    def _eval(self, h):
        if h == 1: return self.evaluation_basique()
        if h == 2: return self.evaluation_materiel_dynamique()
        return self.evaluation_avancee()

    def minimax_sans_elagage(self, profondeur, heuristique=3):
        """Retourne le meilleur coup sans utiliser α-β."""
        self.node_count = 0
        meilleur_coup = None
        coups = self.obtenir_coups_possibles(self.tour)
        random.shuffle(coups)
        if self.tour == 1:
            best = float('-inf')
            for coup in coups:
                pc, ae = self.jouer_coup(coup)
                val = self.min_value_no_ab(profondeur - 1, heuristique)
                self.annuler_coup(coup, pc, ae)
                if val > best:
                    best, meilleur_coup = val, coup
        else:
            best = float('inf')
            for coup in coups:
                pc, ae = self.jouer_coup(coup)
                val = self.max_value_no_ab(profondeur - 1, heuristique)
                self.annuler_coup(coup, pc, ae)
                if val < best:
                    best, meilleur_coup = val, coup
        return meilleur_coup

    def minimax_avec_elagage(self, profondeur, heuristique=3):
        """Alias instrumenté : remet le compteur à zéro et appelle minimax_decision."""
        self.node_count = 0
        return self.minimax_decision(profondeur, heuristique)


# ─────────────────────────────────────────────────────────────
# BENCHMARK : TEMPS DE CALCUL
# ─────────────────────────────────────────────────────────────

def benchmark_temps(n_positions=20, profondeurs=range(1, 8)):
    """
    Mesure le temps moyen (secondes) d'un appel minimax_decision
    pour chaque (profondeur, heuristique).
    Retourne un dict  resultats[profondeur][heuristique] = temps_moyen

    Note méthodo :
        - on génère des positions “réalistes” en jouant quelques coups aléatoires depuis l'initial.
        - on mesure `minimax_decision` (donc un “coup IA complet”) pour lisser les micro-variations.
    """
    print("\n" + "="*60)
    print("BENCHMARK 1 : Temps de calcul par profondeur")
    print("="*60)

    resultats = {p: {} for p in profondeurs}

    for h in [1, 2, 3]:
        nom = ["", "H1 Basique", "H2 Dynamique", "H3 Avancée"][h]
        print(f"\nHeuristique {nom} :")
        for prof in profondeurs:
            # Sauter les combinaisons trop lentes (> ~30 s estimées)
            if prof >= 8:
                resultats[prof][h] = float('nan')
                continue

            temps_list = []
            for _ in range(n_positions):
                jeu = OnitamaBench()
                # Diversification : on avance la partie de quelques coups aléatoires.
                # Objectif : éviter de benchmarker uniquement la position initiale.
                nb_coups_init = random.randint(0, 8)
                for _ in range(nb_coups_init):
                    coups = jeu.obtenir_coups_possibles(jeu.tour)
                    if not coups:
                        break
                    jeu.jouer_coup(random.choice(coups))
                    if jeu.verifier_victoire() is not None:
                        break

                t0 = time.perf_counter()
                jeu.minimax_decision(prof, h)
                t1 = time.perf_counter()
                temps_list.append(t1 - t0)

            moyenne = statistics.mean(temps_list)
            resultats[prof][h] = moyenne
            print(f"  Prof {prof} : {moyenne:.4f} s  (±{statistics.stdev(temps_list):.4f})")

    # ── Tableau texte ──
    print("\nTableau récapitulatif (temps en secondes) :")
    print(f"{'Prof':>6}  {'H1':>10}  {'H2':>10}  {'H3':>10}")
    for p in profondeurs:
        row = [resultats[p].get(h, float('nan')) for h in [1, 2, 3]]
        vals = "  ".join(f"{v:>10.4f}" if not np.isnan(v) else f"{'--':>10}" for v in row)
        print(f"{p:>6}  {vals}")

    return resultats


def tracer_temps(resultats, profondeurs=range(1, 8)):
    fig, ax = plt.subplots(figsize=(8, 5))
    couleurs = {1: "steelblue", 2: "darkorange", 3: "forestgreen"}
    labels   = {1: "H1 – Matériel basique",
                2: "H2 – Matériel dynamique",
                3: "H3 – Avancée (H2 + centre)"}
    for h in [1, 2, 3]:
        ys = [resultats[p].get(h, float('nan')) for p in profondeurs]
        ax.plot(list(profondeurs), ys, marker="o", color=couleurs[h], label=labels[h])

    ax.set_yscale("log")
    ax.set_xlabel("Profondeur de recherche", fontsize=12)
    ax.set_ylabel("Temps moyen (s) | échelle log", fontsize=12)
    ax.set_title("Temps de calcul d'un coup selon la profondeur", fontsize=13)
    ax.set_xticks(list(profondeurs))
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("temps_calcul.png", dpi=150)
    plt.close(fig)
    print("\n→ Graphique sauvegardé : temps_calcul.png")


# ─────────────────────────────────────────────────────────────
# BENCHMARK : NŒUDS EXPLORÉS (avec / sans α-β)
# ─────────────────────────────────────────────────────────────

def benchmark_noeuds(n_positions=20, profondeurs=[2, 3, 4, 5], heuristique=3):
    """
    Compare le nombre de nœuds explorés avec et sans élagage α-β.
    """
    print("\n" + "="*60)
    print("BENCHMARK 2 : Nœuds explorés (avec / sans élagage α-β)")
    print(f"  Heuristique : H{heuristique}  |  Positions testées : {n_positions}")
    print("="*60)

    resultats = {}   # profondeur -> {"avec": moy, "sans": moy}

    for prof in profondeurs:
        avec_list, sans_list = [], []
        for _ in range(n_positions):
            jeu = OnitamaBench()
            # On échantillonne plusieurs positions en jouant quelques coups aléatoires.
            nb_coups_init = random.randint(0, 6)
            for _ in range(nb_coups_init):
                coups = jeu.obtenir_coups_possibles(jeu.tour)
                if not coups or jeu.verifier_victoire() is not None:
                    break
                jeu.jouer_coup(random.choice(coups))

            # Avec élagage
            jeu.minimax_avec_elagage(prof, heuristique)
            avec_list.append(jeu.node_count)

            # Sans élagage : on repart exactement de la même position.
            # Ici on copie à la main les champs nécessaires (plateau + mains + extra + tour).
            jeu2 = OnitamaBench()
            jeu2.plateau = copy.deepcopy(jeu.plateau)
            jeu2.p1_cards = list(jeu.p1_cards)
            jeu2.p2_cards = list(jeu.p2_cards)
            jeu2.extra_card = jeu.extra_card
            jeu2.tour = jeu.tour
            jeu2.minimax_sans_elagage(prof, heuristique)
            sans_list.append(jeu2.node_count)

        moy_avec = statistics.mean(avec_list)
        moy_sans = statistics.mean(sans_list)
        ratio    = moy_sans / moy_avec if moy_avec > 0 else float('nan')
        resultats[prof] = {"avec": moy_avec, "sans": moy_sans, "ratio": ratio}
        print(f"  Prof {prof} : sans={moy_sans:>9.0f}  avec={moy_avec:>9.0f}  "
              f"ratio={ratio:.2f}×")

    print("\nTableau récapitulatif :")
    print(f"{'Prof':>6}  {'Sans élagage':>14}  {'Avec α-β':>12}  {'Ratio':>8}")
    for p in profondeurs:
        r = resultats[p]
        print(f"{p:>6}  {r['sans']:>14.0f}  {r['avec']:>12.0f}  {r['ratio']:>8.2f}×")

    return resultats


def tracer_noeuds(resultats):
    profondeurs = sorted(resultats.keys())
    sans = [resultats[p]["sans"] for p in profondeurs]
    avec = [resultats[p]["avec"] for p in profondeurs]

    x = np.arange(len(profondeurs))
    largeur = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - largeur/2, sans, largeur, label="Sans élagage α-β", color="tomato")
    ax.bar(x + largeur/2, avec, largeur, label="Avec élagage α-β",  color="steelblue")

    ax.set_yscale("log")
    ax.set_xlabel("Profondeur de recherche", fontsize=12)
    ax.set_ylabel("Nœuds explorés (échelle log)", fontsize=12)
    ax.set_title("Impact de l'élagage α-β sur le nombre de nœuds (H3)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in profondeurs])
    ax.legend()
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("noeuds_explores.png", dpi=150)
    plt.close(fig)
    print("\n→ Graphique sauvegardé : noeuds_explores.png")


# ─────────────────────────────────────────────────────────────
# TOURNOI : MATRICE 3×3
# ─────────────────────────────────────────────────────────────

def _jouer_partie_silencieuse(args):
    """Worker pour ProcessPoolExecutor."""
    profil_bleu, profil_rouge = args
    return lancer_partie("IA", "IA",
                         profil_ia_1=profil_bleu,
                         profil_ia_2=profil_rouge,
                         silencieux=True)


def generer_matrice_tournoi(n_parties=50):
    """
    Fait s'affronter toutes les paires d'IA (avec inversion de couleur incluse).
    Retourne une matrice 3×3 de taux de victoire du Joueur Bleu.

    Important : ici on mesure “Bleu gagne (%)” pour une paire (niveauBleu, niveauRouge).
    Si tu veux intégrer l'inversion de couleur, il faut lancer aussi (niveauRouge, niveauBleu)
    et agréger les deux scores. (La docstring initiale le mentionne, mais le code actuel
    calcule case par case sans faire automatiquement cette agrégation.)
    """
    print("\n" + "="*60)
    print(f"BENCHMARK 3 : Matrice de tournoi ({n_parties} parties par case)")
    print("="*60)

    niveaux = ["1", "2", "3"]
    noms    = ["Facile", "Moyen", "Difficile"]
    matrice = np.zeros((3, 3))

    for i, bleu in enumerate(niveaux):
        for j, rouge in enumerate(niveaux):
            pb = PROFILS_IA[bleu]
            pr = PROFILS_IA[rouge]
            taches = [(pb, pr)] * n_parties

            with concurrent.futures.ProcessPoolExecutor() as ex:
                resultats = list(ex.map(_jouer_partie_silencieuse, taches))

            victoires_bleu = sum(1 for r in resultats if r == 1)
            taux = victoires_bleu / n_parties * 100
            matrice[i, j] = taux
            print(f"  {noms[i]:10s} (Bleu) vs {noms[j]:10s} (Rouge) : {taux:.1f}%")

    # ── Heatmap ──
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrice, vmin=0, vmax=100, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Taux de victoire Bleu (%)")

    ax.set_xticks(range(3)); ax.set_xticklabels(noms, fontsize=11)
    ax.set_yticks(range(3)); ax.set_yticklabels(noms, fontsize=11)
    ax.set_xlabel("Rouge (Adversaire)", fontsize=12)
    ax.set_ylabel("Bleu (Joueur)", fontsize=12)
    ax.set_title(f"Matrice de tournoi | taux victoire Bleu (%)\n({n_parties} parties / case)",
                 fontsize=12)

    for i in range(3):
        for j in range(3):
            couleur = "white" if matrice[i, j] < 30 or matrice[i, j] > 70 else "black"
            ax.text(j, i, f"{matrice[i, j]:.1f}%",
                    ha="center", va="center", fontsize=13,
                    fontweight="bold", color=couleur)

    fig.tight_layout()
    fig.savefig("matrice_resultats.png", dpi=150)
    plt.close(fig)
    print("\n→ Graphique sauvegardé : matrice_resultats.png")

    return matrice


# ─────────────────────────────────────────────────────────────
# TOURNOI : COURBE D'ÉVOLUTION
# ─────────────────────────────────────────────────────────────

def generer_courbe_evolution(n_parties=500,
                             paires=None):
    """
    Trace l'évolution du taux de victoire cumulé du Joueur Bleu
    pour plusieurs paires d'IA sur un seul graphique.

    paires : liste de tuples (niveau_bleu, niveau_rouge)
             ex. [("2","1"), ("3","1"), ("3","2")]
             Par défaut : les 3 paires inter-niveaux.
    """
    if paires is None:
        paires = [("2", "1"), ("3", "1"), ("3", "2")]

    print("\n" + "="*60)
    print(f"BENCHMARK 4 : Courbe d'évolution ({n_parties} parties)")
    print("="*60)

    COULEURS = {
        ("2", "1"): ("darkorange", "Moyen vs Facile"),
        ("3", "1"): ("crimson",    "Difficile vs Facile"),
        ("3", "2"): ("steelblue",  "Difficile vs Moyen"),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(50, color="gray", linestyle="--", lw=1, label="50 %", zorder=1)

    for niveau_bleu, niveau_rouge in paires:
        pb = PROFILS_IA[niveau_bleu]
        pr = PROFILS_IA[niveau_rouge]
        taches = [(pb, pr)] * n_parties

        with concurrent.futures.ProcessPoolExecutor() as ex:
            # On soumet toutes les tâches à l'exécuteur
            futures = [ex.submit(_jouer_partie_silencieuse, tache) for tache in taches]
            resultats = []

            # as_completed nous donne les résultats au fur et à mesure qu'ils finissent
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                resultats.append(future.result())
                # Le \r et flush=True permettent d'écraser la ligne précédente pour un “progress bar” simple.
                print(f"\r    Progression : {i} / {n_parties} parties terminées...", end="", flush=True)

            print()  # Pour revenir à la ligne une fois les parties terminées

        # Taux cumulé après chaque partie
        victoires = 0
        taux_cumul = []
        for k, r in enumerate(resultats, 1):
            if r == 1:
                victoires += 1
            taux_cumul.append(victoires / k * 100)

        couleur, label_base = COULEURS.get(
            (niveau_bleu, niveau_rouge),
            ("purple", f"{pb['nom']} vs {pr['nom']}")
        )
        label = f"{label_base}  →  {taux_cumul[-1]:.1f}%"

        ax.plot(range(1, n_parties + 1), taux_cumul,
                color=couleur, lw=1.5, label=label, zorder=2)
        ax.axhline(taux_cumul[-1], color=couleur, linestyle=":",
                   lw=0.9, alpha=0.6, zorder=1)

        print(f"  {pb['nom']:10s} vs {pr['nom']:10s} : taux final {taux_cumul[-1]:.1f}%")

    ax.set_xscale("log")
    ax.set_xlabel("Nombre de parties (échelle log)", fontsize=12)
    ax.set_ylabel("Taux de victoire cumulé Bleu (%)", fontsize=12)
    ax.set_title("Évolution du taux de victoire selon les paires d'IA", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("evolution_victoires.png", dpi=150)
    plt.close(fig)
    print("→ Graphique sauvegardé : evolution_victoires.png")


# ─────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("╔══════════════════════════════════════════════════╗")
    print("║   Script de benchmark | Projet Onitama IA        ║")
    print("╚══════════════════════════════════════════════════╝")
    print("\nQue voulez-vous générer ?")
    print("  1. Temps de calcul par profondeur")
    print("  2. Nœuds explorés (avec / sans α-β)")
    print("  3. Matrice de tournoi 3×3")
    print("  4. Courbe d'évolution des victoires")
    print("  5. TOUT générer (peut prendre plusieurs heures)")
    choix = input("\nVotre choix > ").strip()

    # ── Benchmark 1 : temps ──
    if choix in ("1", "5"):
        res_temps = benchmark_temps(n_positions=20)
        tracer_temps(res_temps)

    # ── Benchmark 2 : nœuds ──
    if choix in ("2", "5"):
        res_noeuds = benchmark_noeuds(n_positions=20, profondeurs=[2, 3, 4, 5])
        tracer_noeuds(res_noeuds)

    # ── Benchmark 3 : tournoi matrice ──
    if choix in ("3", "5"):
        n = int(input("\nNombre de parties par case [50] > ") or "50")
        matrice = generer_matrice_tournoi(n_parties=n)

    # ── Benchmark 4 : courbe évolution ──
    if choix in ("4", "5"):
        n = int(input("\nNombre de parties pour la courbe [500] > ") or "500")
        generer_courbe_evolution(n_parties=n)

    print("\n✓ Terminé.")