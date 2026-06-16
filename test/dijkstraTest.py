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


class PCCSommet(unittest.TestCase):

    def setUp(self):
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
        dist, peres = self.algo.calculPCCSommet(1)
        expected_dist = [math.inf, 0, math.inf, math.inf]
        expected_peres = [None, None, None, None]
        self.assertEqual(list(dist), expected_dist)
        self.assertEqual(list(peres), list(expected_peres))

    def test_pccsommet_TD1_case2(self):
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

    def setUp(self):
        self.matrice = [
            [math.inf, 10, 5],
            [10, math.inf, 20],
            [5, 20, math.inf]
        ]
        self.graphe = GrapheValueNonOriente(np.array(self.matrice))
        self.algo = AlgoDijkstra(self.graphe)

    def test_calcul_global(self):
        dist_mat, preds_mat = self.algo.calculPCCTousSommets()
        self.assertEqual(dist_mat[1][2], 15)
        self.assertEqual(preds_mat[1][2], 0)
        for i in range(self.graphe.nb_sommets()):
            self.assertEqual(dist_mat[i][i], 0)


if __name__ == '__main__':
    unittest.main()
