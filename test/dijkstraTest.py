import unittest
import sys
import os
import math
import numpy as np

# Configuration du chemin (déjà dans ton code)
path_to_source = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Source_python'))
if path_to_source not in sys.path:
    sys.path.insert(0, path_to_source)

from graphe_non_oriente import GrapheValueNonOriente
from algo_Dijkstra import AlgoDijkstra


class PCCSommet(unittest.TestCase):

    def setUp(self):
        # Matrice du TD : 0->1(8), 0->2(6), 0->3(2), 2->1(3), 3->1(5), 3->2(1)
        # Note : Bien que GrapheValueNonOriente suggère un graphe non orienté,
        # Dijkstra fonctionne parfaitement sur cette matrice asymétrique (orientée).
        self.matrice = [
            [math.inf, 8, 6, 2],
            [math.inf, math.inf, math.inf, math.inf],
            [math.inf, 3, math.inf, math.inf],
            [math.inf, 5, 1, math.inf]
        ]
        self.graphe = GrapheValueNonOriente(np.array(self.matrice))
        self.algo = AlgoDijkstra(self.graphe)

    def test_pccsommet_TD1_case0(self):
        dist, peres = self.algo.calculPCCSommet(0)
        expected_dist = [0, 6, 3, 2]
        expected_peres = [None, 2, 3, 0]
        self.assertEqual(list(dist), expected_dist)
        self.assertEqual(list(peres), expected_peres)

    def test_pccsommet_TD1_case1(self):
        """Test pour le sommet 1 (sommet puits, aucune sortie)"""
        dist, peres = self.algo.calculPCCSommet(1)
        expected_dist = [math.inf, 0, math.inf, math.inf]
        expected_peres = [None, None, None, None]
        self.assertEqual(list(dist), expected_dist)
        self.assertEqual(list(peres), list(expected_peres))

    def test_pccsommet_TD1_case2(self):
        """Test pour le sommet 2"""
        dist, peres = self.algo.calculPCCSommet(2)
        expected_dist = [math.inf, 3, 0, math.inf]
        expected_peres = [None, 2, None, None]
        self.assertEqual(list(dist), expected_dist)
        self.assertEqual(list(peres), expected_peres)

    def test_pccsommet_TD1_case3(self):
        dist, peres = self.algo.calculPCCSommet(3)
        expected_dist = [math.inf, 4, 1, 0]
        expected_peres = [None, 2, 3, None]
        self.assertEqual(list(dist), expected_dist)
        self.assertEqual(list(peres), expected_peres)


class PCCTousSommets(unittest.TestCase):
    """Vérifie le calcul global sur tous les sommets"""

    def setUp(self):
        # Petit graphe simple cyclique 0-1(10), 1-2(20), 0-2(5)
        self.matrice = [
            [math.inf, 10, 5],
            [10, math.inf, 20],
            [5, 20, math.inf]
        ]
        self.graphe = GrapheValueNonOriente(np.array(self.matrice))
        self.algo = AlgoDijkstra(self.graphe)

    def test_calcul_global(self):
        dist_mat, preds_mat = self.algo.calculPCCTousSommets()

        # Distance de 1 à 2 en passant par 0 (10 + 5 = 15)
        # car 1->2 direct coûte 20.
        self.assertEqual(dist_mat[1][2], 15)
        self.assertEqual(preds_mat[1][2], 0)

        # La diagonale doit être à 0
        for i in range(self.graphe.nb_sommets()):
            self.assertEqual(dist_mat[i][i], 0)


if __name__ == '__main__':
    unittest.main()