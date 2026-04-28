# -*- coding: utf-8 -*-
"""
Implémentation (simplifiée) du jeu Onitama + IA Minimax avec élagage α‑β.

Conventions importantes utilisées partout dans le projet :
    - Plateau 5×5 : `plateau[y][x]` avec `y` = ligne (0 en haut), `x` = colonne (0 à gauche).
    - Codage des pièces :
        - `0` : case vide
        - `1` : élève Bleu (J1, en haut)
        - `2` : maître Bleu
        - `-1` : élève Rouge (J2, en bas)
        - `-2` : maître Rouge
    - Temple / arche : cases (x=2, y=0) pour Bleu et (x=2, y=4) pour Rouge.
    - Cartes : chaque carte contient une liste de déplacements (dx, dy) + une couleur ("Bleu"/"Rouge").
    Les déplacements sont exprimés dans le repère “Bleu”. Pour Rouge on applique un miroir via `sens = -1`
    (voir `obtenir_coups_possibles`).

Note : ce fichier contient aussi un menu CLI (en bas) pour jouer ou lancer un tournoi IA en parallèle.
"""

import random
import concurrent.futures


RESET = "\033[0m"
BOLD = "\033[1m"
BLUE = "\033[94m"
RED = "\033[91m"
GOLD = "\033[93m"

CARTES = {
    "Tiger": [(0, 2), (0, -1), "Bleu"],
    "Dragon": [(-2, 1), (-1, -1), (1, -1), (2, 1), "Rouge"],
    "Frog": [(-2, 0), (-1, 1), (1, -1), "Rouge"],
    "Rabbit": [(-1, -1), (1, 1), (2, 0), "Bleu"],
    "Crab": [(-2, 0), (0, 1), (2, 0), "Bleu"],
    "Elephant": [(-1, 1), (-1, 0), (1, 0), (1, 1), "Rouge"],
    "Goose": [(-1, 1), (-1, 0), (1, 0), (1, -1), "Bleu"],
    "Rooster": [(-1, -1), (-1, 0), (1, 0), (1, 1), "Rouge"],
    "Monkey": [(-1, 1), (-1, -1), (1, 1), (1, -1), "Bleu"],
    "Mantis": [(-1, 1), (0, -1), (1, 1), "Rouge"],
    "Horse": [(-1, 0), (0, 1), (0, -1), "Rouge"],
    "Ox": [(0, 1), (0, -1), (1, 0), "Bleu"],
    "Crane": [(0, 1), (-1, -1), (1, -1), "Bleu"],
    "Boar": [(0, 1), (-1, 0), (1, 0), "Rouge"],
    "Eel": [(-1, 1), (-1, -1), (1, 0), "Bleu"],
    "Cobra": [(-1, 0), (1, 1), (1, -1), "Rouge"]
}


class Onitama:
    def __init__(self):
        """
        Initialise une partie standard :
            - placement des pièces sur les deux rangées extrêmes
            - distribution aléatoire de 5 cartes (2 pour chaque joueur + 1 carte "extra")
            - détermination du joueur qui commence selon la couleur de la carte extra
        """
        self.plateau = [[1, 1, 2, 1, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [-1, -1, -2, -1, -1]]

        self.cartes = list(CARTES.keys())
        random.shuffle(self.cartes)
        self.p1_cards = [self.cartes[0], self.cartes[1]]
        self.p2_cards = [self.cartes[2], self.cartes[3]]
        self.extra_card = self.cartes[4]

        couleur_depart = CARTES[self.extra_card][-1]

        if couleur_depart == "Bleu":
            self.tour = 1
        else:
            self.tour = -1

        self._info_depart = f"Carte extra : {self.extra_card} ({couleur_depart}) -> Joueur {self.tour} commence"

    def afficher(self):
        """Affiche le plateau en ASCII + les cartes en main."""
        print(f"\n{BOLD}      0   1   2   3   4{RESET}")
        print("    ┌───┬───┬───┬───┬───┐")

        for y, ligne in enumerate(self.plateau):
            line_str = f"  {y} │"

            for x, cell in enumerate(ligne):
                symbol = " "
                color = RESET

                # Les deux temples (arche) sont (2,0) et (2,4). Quand ils sont vides,
                # on les marque en "." pour qu'ils restent visibles à l'affichage.
                is_arch = (x == 2 and (y == 0 or y == 4))

                if is_arch and cell == 0:
                    symbol = "."
                    color = GOLD

                bg_start = " "
                bg_end = " "

                if cell == 0:
                    pass

                elif cell == 1:
                    symbol = "E"
                    color = BLUE
                elif cell == 2:
                    symbol = "M"
                    color = BLUE

                elif cell == -1:
                    symbol = "E"
                    color = RED
                elif cell == -2:
                    symbol = "M"
                    color = RED

                line_str += f"{bg_start}{color}{symbol}{RESET}{bg_end}│"

            print(line_str)

            if y < 4:
                print("    ├───┼───┼───┼───┼───┤")
            else:
                print("    └───┴───┴───┴───┴───┘")

        print(f"{BLUE}Main J1 (Haut): {', '.join(self.p1_cards)}{RESET}")
        print(f"{RED}Main J2 (Bas) : {', '.join(self.p2_cards)}{RESET}")
        print(f"{GOLD}Extra         : {self.extra_card}{RESET}")

    def obtenir_coups_possibles(self, joueur):
        """
        Génère tous les coups légaux pour `joueur` (1 = Bleu, -1 = Rouge).

        Un coup est un dict :
            - `carte` : nom de carte utilisé
            - `de`    : (x, y) départ
            - `vers`  : (x, y) arrivée

        Détail clé : les vecteurs (dx, dy) des cartes sont définis dans le repère “Bleu”.
        Pour Rouge, on applique `sens = -1` afin de retourner la direction des déplacements.
        """
        coups = []

        if joueur == 1:
            deck, sens = self.p1_cards, 1
        else:
            deck, sens = self.p2_cards, -1

        for i in range(5):
            for j in range(5):
                piece = self.plateau[i][j]

                if piece != 0 and (piece * joueur) > 0:
                    for carte in deck:
                        mouvements = CARTES[carte][:-1]
                        for (x, y) in mouvements:
                            nx = j + (x * sens)
                            ny = i + (y * sens)

                            if 0 <= nx < 5 and 0 <= ny < 5:
                                cible = self.plateau[ny][nx]

                                if cible * joueur <= 0:    # case vide ou piece adverse
                                    coups.append({
                                        "carte": carte,
                                        "de": (j, i),
                                        "vers": (nx, ny)
                                    })
        return coups

    def jouer_coup(self, coup):
        """
        Applique un coup sur l'état courant.

        Effets Onitama modélisés ici :
            - déplacement (et capture éventuelle)
            - échange de cartes : la carte jouée devient la carte extra, et l'ancienne extra
            remplace la carte jouée dans la main du joueur actif
            - alternance du tour (`self.tour *= -1`)

        Retourne ce qu'il faut pour annuler proprement (utile pour Minimax) :
        `(piece_capturee, ancienne_extra)`.
        """
        x_depart, y_depart = coup["de"]
        x_arrive, y_arrive = coup["vers"]
        carte_jouee = coup["carte"]

        valeur_piece = self.plateau[y_depart][x_depart]
        piece_capturee = self.plateau[y_arrive][x_arrive]
        self.plateau[y_depart][x_depart] = 0
        self.plateau[y_arrive][x_arrive] = valeur_piece

        if self.tour == 1:
            deck = self.p1_cards
        else:
            deck = self.p2_cards

        ancienne_extra = self.extra_card

        deck.remove(carte_jouee)
        deck.append(self.extra_card)

        self.extra_card = carte_jouee

        self.tour *= -1

        return piece_capturee, ancienne_extra

    def annuler_coup(self, coup, piece_capturee, ancienne_extra):
        """
        Inverse exactement les effets de `jouer_coup`.

        Cette fonction est centrale pour Minimax : on explore des coups puis on revient à
        l'état précédent sans recopier tout le plateau à chaque fois.
        """
        x_depart, y_depart = coup["de"]
        x_arrive, y_arrive = coup["vers"]
        carte_jouee = coup["carte"]

        self.tour *= -1

        if self.tour == 1:
            deck = self.p1_cards
        else:
            deck = self.p2_cards

        valeur_piece = self.plateau[y_arrive][x_arrive]
        self.plateau[y_depart][x_depart] = valeur_piece
        self.plateau[y_arrive][x_arrive] = piece_capturee

        deck.remove(ancienne_extra)
        deck.append(carte_jouee)

        self.extra_card = ancienne_extra

    def verifier_victoire(self):
        """
        Conditions de victoire Onitama :
            - Voie du courant : amener son maître sur le temple adverse
            - Voie de la pierre : capturer le maître adverse (il n'est plus “vivant”)

        Retourne :
            - `1`    si Bleu gagne
            - `-1`   si Rouge gagne
            - `None` sinon.
        """
        master_p1_vivant = False
        master_p2_vivant = False

        for i in range(5):
            for j in range(5):
                piece = self.plateau[i][j]
                if piece == 2:  # Maitre bleu
                    master_p1_vivant = True
                    if i == 4 and j == 2:  # Voie du Courant (temple rouge)
                        return 1
                elif piece == -2:  # Maitre rouge
                    master_p2_vivant = True
                    if i == 0 and j == 2:  # Voie du Courant (temple bleu)
                        return -1

        if not master_p1_vivant:
            return -1
        if not master_p2_vivant:
            return 1
        return None

    def evaluation_basique(self):
        """
        joueur Bleu (1) est MAX.
        score positif si Bleu mène, score négatif si Rouge mène.
        """
        score = 0
        for y in range(5):
            for x in range(5):
                piece = self.plateau[y][x]
                if piece == 1:
                    score += 10
                elif piece == 2:
                    score += 100
                elif piece == -1:
                    score -= 10
                elif piece == -2:
                    score -= 100
        return score

    def evaluation_materiel_dynamique(self):
        """
        Heuristique H2 : Valeur marginale décroissante des pions.
        La valeur d'un pion augmente au fur et à mesure que les pions se raréfient.
        """

        def points_par_pions(compte):
            if compte == 0: return 0   # :D
            if compte == 1: return 8   # 8
            if compte == 2: return 14  # 8 + 6
            if compte == 3: return 18  # 8 + 6 + 4
            if compte >= 4: return 20  # 8 + 6 + 4 + 2

        pions_bleus = 0
        pions_rouges = 0
        maitre_bleu_vivant = False
        maitre_rouge_vivant = False

        for y in range(5):
            for x in range(5):
                piece = self.plateau[y][x]
                if piece == 1:
                    pions_bleus += 1
                elif piece == 2:
                    maitre_bleu_vivant = True
                elif piece == -1:
                    pions_rouges += 1
                elif piece == -2:
                    maitre_rouge_vivant = True

        if not maitre_bleu_vivant: return -10000
        if not maitre_rouge_vivant: return 10000

        score = points_par_pions(pions_bleus) - points_par_pions(pions_rouges)

        return score

    def evaluation_avancee(self):
        """
        Heuristique H3 : Combine le matériel dynamique (H2) ET le contrôle du centre.
        """
        # On part de la nouvelle heuristique (H2)
        score = self.evaluation_materiel_dynamique()

        # On ajoute les bonus de positionnement
        for y in range(5):
            for x in range(5):
                piece = self.plateau[y][x]
                if piece != 0:
                    bonus = 0
                    if 1 <= x <= 3 and 1 <= y <= 3:
                        bonus += 2
                    if x == 2 and y == 2:
                        bonus += 3

                    if piece > 0:
                        score += bonus
                    else:
                        score -= bonus

        return score

    def minimax_decision(self, profondeur, heuristique=3):
        """
        Choisit le meilleur coup pour le joueur dont c'est le tour, via Minimax + α‑β.

        - Bleu (`tour == 1`) maximise le score
        - Rouge (`tour == -1`) minimise le score
        - `random.shuffle` introduit un léger aléa utile pour éviter des parties identiques
          quand plusieurs coups ont des scores égaux.
        """
        meilleur_coup = None
        coups = self.obtenir_coups_possibles(self.tour)

        random.shuffle(coups)

        # On initialise alpha et beta pour le tout premier appel
        alpha = float('-inf')
        beta = float('inf')

        # Si l'IA joue Bleu (1) -> MAX
        if self.tour == 1:
            meilleur_score = float('-inf')
            for coup in coups:
                piece_capturee, ancienne_extra = self.jouer_coup(coup)
                val = self.min_value(profondeur - 1, alpha, beta, heuristique)
                self.annuler_coup(coup, piece_capturee, ancienne_extra)

                if val > meilleur_score:
                    meilleur_score = val
                    meilleur_coup = coup

                alpha = max(alpha, meilleur_score)
                if alpha >= beta:
                    break

        # Si l'IA joue Rouge (-1) -> MIN
        else:
            meilleur_score = float('inf')
            for coup in coups:
                piece_capturee, ancienne_extra = self.jouer_coup(coup)
                val = self.max_value(profondeur - 1, alpha, beta, heuristique)
                self.annuler_coup(coup, piece_capturee, ancienne_extra)

                if val < meilleur_score:
                    meilleur_score = val
                    meilleur_coup = coup

                beta = min(beta, meilleur_score)
                if beta <= alpha:
                    break

        return meilleur_coup

    def max_value(self, profondeur, alpha, beta, heuristique=3):
        """Noeud MAX de Minimax avec élagage α‑β (Bleu)."""
        gagnant = self.verifier_victoire()
        if gagnant is not None:
            return 10000 + profondeur if gagnant == 1 else -10000 - profondeur

        if profondeur == 0:
            if heuristique == 1:
                return self.evaluation_basique()
            elif heuristique == 2:
                return self.evaluation_materiel_dynamique()
            else:
                return self.evaluation_avancee()

        v = float('-inf')
        coups = self.obtenir_coups_possibles(1)

        for coup in coups:
            piece_capturee, ancienne_extra = self.jouer_coup(coup)
            v = max(v, self.min_value(profondeur - 1, alpha, beta, heuristique))
            self.annuler_coup(coup, piece_capturee, ancienne_extra)

            # ÉLAGAGE ALPHA-BÊTA POUR MAX
            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v

    def min_value(self, profondeur, alpha, beta, heuristique=3):
        """Noeud MIN de Minimax avec élagage α‑β (Rouge)."""
        gagnant = self.verifier_victoire()
        if gagnant is not None:
            return 10000 + profondeur if gagnant == 1 else -10000 - profondeur

        if profondeur == 0:
            if heuristique == 1:
                return self.evaluation_basique()
            elif heuristique == 2:
                return self.evaluation_materiel_dynamique()
            else:
                return self.evaluation_avancee()

        v = float('inf')
        coups = self.obtenir_coups_possibles(-1)

        for coup in coups:
            piece_capturee, ancienne_extra = self.jouer_coup(coup)
            v = min(v, self.max_value(profondeur - 1, alpha, beta, heuristique))
            self.annuler_coup(coup, piece_capturee, ancienne_extra)

            # ÉLAGAGE ALPHA-BÊTA POUR MIN
            if v <= alpha:
                return v
            beta = min(beta, v)

        return v


# CONFIGURATION DES IA 

PROFILS_IA = {
    "1": {"nom": "Facile  (Prof 3, H1 Matériel Basique)",    "profondeur": 3, "heuristique": 1},
    "2": {"nom": "Moyen   (Prof 4, H2 Matériel Dynamique)",  "profondeur": 4, "heuristique": 2},
    "3": {"nom": "Difficile (Prof 5, H3 Dynamique+Centre)",  "profondeur": 5, "heuristique": 3}
}


# CONFIGURATION POUR LE TEST D'ÉQUITÉ DES HEURISTIQUES 
"""
PROFILS_IA = {
    "1": {"nom": "H1 (Matériel Basique)",    "profondeur": 4, "heuristique": 1},
    "2": {"nom": "H2 (Matériel Dynamique)",  "profondeur": 4, "heuristique": 2},
    "3": {"nom": "H3 (Dynamique + Centre)",  "profondeur": 4, "heuristique": 3}
}
"""

def choisir_profil_ia(nom_joueur):
    print(f"\nChoisissez le niveau pour l'IA {nom_joueur} :")
    for k, v in PROFILS_IA.items():
        print(f"{k}. {v['nom']}")
    choix = input("> ")
    return PROFILS_IA.get(choix, PROFILS_IA["2"])  # Moyen par défaut


def lancer_partie(joueur_1_type, joueur_2_type, profil_ia_1=None, profil_ia_2=None, silencieux=False):
    MAX_TOURS = 1000
    jeu = Onitama()

    if not silencieux:
        print(jeu._info_depart)

    # Dictionnaires pour retrouver facilement les paramètres de chaque joueur
    types = {1: joueur_1_type, -1: joueur_2_type}
    profils = {1: profil_ia_1, -1: profil_ia_2}

    nb_tours = 0

    while True:
        if not silencieux:
            jeu.afficher()

        nb_tours += 1
        if nb_tours > MAX_TOURS:
            if not silencieux:
                print(f"\n{GOLD}*** Partie nulle (limite de {MAX_TOURS} tours atteinte) ***{RESET}")
            return 0

        gagnant = jeu.verifier_victoire()
        if gagnant is not None:
            if not silencieux:
                nom_gagnant = "BLEU (J1)" if gagnant == 1 else "ROUGE (J2)"
                print(f"\n{GOLD}*** {nom_gagnant} a gagné ! ***{RESET}")
            return gagnant

        type_actuel = types[jeu.tour]

        if not silencieux:
            nom_couleur = "BLEU" if jeu.tour == 1 else "ROUGE"
            couleur_texte = BLUE if jeu.tour == 1 else RED
            print(f"\n--- TOUR DU JOUEUR {jeu.tour} : {couleur_texte}{nom_couleur} ({type_actuel}){RESET} ---")

        coups = jeu.obtenir_coups_possibles(jeu.tour)

        if not coups:
            if not silencieux: print("Aucun mouvement possible, le tour passe.")
            jeu.tour *= -1
            continue

        if type_actuel == "IA":
            profil = profils[jeu.tour]
            if not silencieux: print(f"L'IA {profil['nom']} réfléchit...")

            coup_a_jouer = jeu.minimax_decision(profil["profondeur"], profil["heuristique"])

            if not silencieux:
                print(f"L'IA joue : {coup_a_jouer['carte']} de {coup_a_jouer['de']} vers {coup_a_jouer['vers']}")
            jeu.jouer_coup(coup_a_jouer)
            if not silencieux:
                input("Appuyez sur Entrée pour le coup suivant...")

        else:  # Humain
            for index, c in enumerate(coups):
                print(f"{index}: {c['carte']} de {c['de']} vers {c['vers']}")

            choix_coup = input("Quel coup jouer ? (Entrez le numero) > ")
            try:
                index_choisi = int(choix_coup)
                if 0 <= index_choisi < len(coups):
                    jeu.jouer_coup(coups[index_choisi])
                else:
                    print(f"Erreur: choisissez un numéro entre 0 et {len(coups) - 1}")
            except ValueError:
                print("Erreur: veuillez entrer un numéro valide")

def jouer_partie_parallele(args):
    profil_bleu, profil_rouge = args
    return lancer_partie("IA", "IA", profil_ia_1=profil_bleu, profil_ia_2=profil_rouge, silencieux=True)

# --- MENU PRINCIPAL ---
if __name__ == "__main__":
    print(f"{BOLD}=== PROJET ONITAMA - IA ==={RESET}")
    print("1. Jouer : Humain vs Humain")
    print("2. Jouer : Humain (Bleu) vs IA (Rouge)")
    print("3. Jouer : IA (Bleu) vs Humain (Rouge)")
    print("4. Jouer : IA (Bleu) vs IA (Rouge) - Mode Spectateur")
    print("5. Lancer un tournoi automatisé (IA vs IA)")

    choix_mode = input("\nVotre choix (1-5) > ")

    if choix_mode == "1":
        lancer_partie("Humain", "Humain")
    elif choix_mode == "2":
        profil_rouge = choisir_profil_ia("ROUGE")
        lancer_partie("Humain", "IA", profil_ia_2=profil_rouge)
    elif choix_mode == "3":
        profil_bleu = choisir_profil_ia("BLEU")
        lancer_partie("IA", "Humain", profil_ia_1=profil_bleu)
    elif choix_mode == "4":
        profil_bleu = choisir_profil_ia("BLEU")
        profil_rouge = choisir_profil_ia("ROUGE")
        lancer_partie("IA", "IA", profil_ia_1=profil_bleu, profil_ia_2=profil_rouge)
    elif choix_mode == "5":
        print("\n--- CONFIGURATION DU TOURNOI PARALLÈLE ---")
        profil_bleu = choisir_profil_ia("BLEU")
        profil_rouge = choisir_profil_ia("ROUGE")
        nb_parties = int(input("Combien de parties ? (Min 50 recommandé) > "))

        victoires_bleu = 0
        victoires_rouge = 0
        nuls = 0

        print("\nTournoi en cours (répartition sur plusieurs cœurs)...")

        taches = [(profil_bleu, profil_rouge) for _ in range(nb_parties)]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            resultats = list(executor.map(jouer_partie_parallele, taches))

        for gagnant in resultats:
            if gagnant == 1:
                victoires_bleu += 1
            elif gagnant == -1:
                victoires_rouge += 1
            else:
                nuls += 1

        print(f"\n\n=== RÉSULTATS DU TOURNOI ===")
        print(
            f"IA BLEU  ({profil_bleu['nom']}) : {victoires_bleu} victoires ({(victoires_bleu / nb_parties) * 100:.1f}%)")
        print(
            f"IA ROUGE ({profil_rouge['nom']}) : {victoires_rouge} victoires ({(victoires_rouge / nb_parties) * 100:.1f}%)")
        if nuls:
            print(f"Nuls : {nuls} ({(nuls / nb_parties) * 100:.1f}%)")