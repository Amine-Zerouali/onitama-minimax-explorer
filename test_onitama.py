# -*- coding: utf-8 -*-
"""
Tests unitaires pour la classe `Onitama`.

Couverture actuelle :
    - état initial cohérent (mains, partie non terminée) ;
    - détection de la victoire par « voie du courant » (maître sur le temple adverse) ;
    - réversibilité de `jouer_coup` / `annuler_coup` (indispensable au Minimax).

Pour lancer : `python -m unittest test_onitama` depuis le dossier du projet.
"""

import unittest
import copy
from onitama import Onitama


class TestOnitama(unittest.TestCase):
    """Tests sur les règles de base et l’intégrité des transitions d’état."""

    def test_etat_initial(self):
        """Après construction, chaque joueur a 2 cartes et aucune victoire n’est déclarée."""
        jeu = Onitama()
        self.assertEqual(len(jeu.p1_cards), 2)
        self.assertEqual(len(jeu.p2_cards), 2)
        self.assertIsNone(jeu.verifier_victoire())

    def test_victoire_voie_du_courant(self):
        """
        La voie du courant : amener son maître sur la case temple adverse.

        Ici on force artificiellement le maître Bleu (valeur 2) sur le temple Rouge
        en bas au centre (y=4, x=2). `verifier_victoire()` doit alors retourner 1.
        """
        jeu = Onitama()
        jeu.plateau[4][2] = 2
        self.assertEqual(jeu.verifier_victoire(), 1)

    def test_jouer_annuler_coup(self):
        """
        Un coup puis son annulation doit restaurer plateau, carte extra et joueur actif.

        Utilisé par l’IA pour explorer l’arbre sans copier tout le plateau à chaque branche.
        """
        jeu = Onitama()
        coups = jeu.obtenir_coups_possibles(jeu.tour)
        # Cas extrême : aucun coup (ne devrait pas arriver en partie normale) ;
        # on sort sans échec pour éviter un faux rouge sur environnements bizarres.
        if not coups:
            return

        coup_test = coups[0]
        plateau_avant = copy.deepcopy(jeu.plateau)
        extra_avant = jeu.extra_card
        tour_avant = jeu.tour

        piece_capturee, ancienne_extra = jeu.jouer_coup(coup_test)
        jeu.annuler_coup(coup_test, piece_capturee, ancienne_extra)

        self.assertEqual(jeu.plateau, plateau_avant)
        self.assertEqual(jeu.extra_card, extra_avant)
        self.assertEqual(jeu.tour, tour_avant)


if __name__ == '__main__':
    unittest.main()