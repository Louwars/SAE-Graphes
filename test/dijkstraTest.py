import unittest
import sys
import os
import math
import numpy as np

path_to_source = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Source_python'))
if path_to_source not in sys.path:
    sys.path.insert(0, path_to_source)

from graphe_non_oriente import GrapheValueNonOriente
from algo_Dijkstra import AlgoDijkstra

class TestDijkstra(unittest.TestCase):

    def setUp(self):
        self.matrice = [
                [math.inf, 8, 6, 2],
                [math.inf, math.inf, math.inf, math.inf],
                [math.inf, 3, math.inf, math.inf],
                [math.inf, 5, 1, math.inf]
        ]
        self.graphe = GrapheValueNonOriente(np.array(self.matrice))
        self.algo = AlgoDijkstra(self.graphe)

    def test_dijkstra_TD1_case0(self):
        dist, peres = self.algo.calculPCCSommet(0)

        expected_dist = [0, 6, 3, 2]
        expected_peres = [None, 2, 3, 0]

        self.assertEqual(list(dist), expected_dist,f"Erreur Distances Sommet 0: obtenu {list(dist)} au lieu de {expected_dist}")
        self.assertEqual(list(peres), expected_peres,f"Erreur Pères Sommet 0: obtenu {list(peres)} au lieu de {expected_peres}")

    def test_dijkstra_TD1_case3(self):
        dist, peres = self.algo.calculPCCSommet(3)

        expected_dist = [math.inf, 4, 1, 0]
        expected_peres = [None, 2, 3, None]

        self.assertEqual(list(dist), expected_dist,f"Erreur Distances Sommet 3: obtenu {list(dist)} au lieu de {expected_dist}")
        self.assertEqual(list(peres), expected_peres,f"Erreur Pères Sommet 3: obtenu {list(peres)} au lieu de {expected_peres}")


if __name__ == '__main__':
    unittest.main()
