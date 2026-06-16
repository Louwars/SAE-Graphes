import numpy as np
import math
import random
import time
from graphe_non_oriente import GrapheValueNonOriente
from algo_plus_court_chemin import AlgoPlusCourtChemin

class AlgoBellmanFord(AlgoPlusCourtChemin):
    def __init__(self, g:GrapheValueNonOriente):
        super().__init__(g)

    def calculPCCSommet(self, s: int):
        n = self.graphe.nb_sommets()
        d = np.full(n, np.inf)
        pred = np.full(n, None)
        d[s] = 0
        aretes = self.graphe.get_liste_aretes()
        for k in range(1, n):
            for (u, v, poids) in aretes:
                if d[u] + poids < d[v]:
                    d[v] = d[u] + poids
                    pred[v] = u
                if d[v] + poids < d[u]:
                    d[u] = d[v] + poids
                    pred[u] = v
        for (u, v, poids) in aretes:
            if d[v] > d[u] + poids:
                print("Il existe un cycle absorbant !")
                break
        return d, pred

    def calculPCCTousSommets(self):
        n = self.graphe.nb_sommets()
        matrice_dist = np.full((n, n), np.inf)
        matrice_preds = np.full((n, n), None)
        for s in range(n):
            dist_s, pred_s = self.calculPCCSommet(s)
            matrice_dist[s] = dist_s
            matrice_preds[s] = pred_s
        return matrice_dist, matrice_preds

if __name__ == "__main__":
    matrice = np.array([[math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf],
                        [1, math.inf, 1, 1, math.inf, math.inf, math.inf],
                        [math.inf, 1, math.inf, 1, 1, math.inf, math.inf],
                        [1, 1, 1, math.inf, 1, math.inf, math.inf],
                        [math.inf, math.inf, 1, 1, math.inf, 1, math.inf],
                        [math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1],
                        [math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf],
                        ])
    g = GrapheValueNonOriente(matrice)
    print(g.nb_sommets())
    print(g.nb_aretes())
    t0 = time.perf_counter()
    algoBF = AlgoBellmanFord(g)
    t_exec = time.perf_counter() - t0
    print(algoBF.distances)
    print(algoBF.predecesseurs)
    print(f"Temps de calcul : {t_exec:.5f} s")

    matrice_m = np.array([
        [math.inf, 1, math.inf, 1, math.inf, 5, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 8],
        [1, math.inf, 1, 1, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 4, math.inf, math.inf],
        [math.inf, 1, math.inf, 1, 1, math.inf, math.inf, math.inf, math.inf, math.inf, 7, math.inf, math.inf, math.inf, math.inf, 3, math.inf, math.inf, math.inf, math.inf],
        [1, 1, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 6, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, 1, 1, math.inf, 1, math.inf, 2, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 5, math.inf],
        [5, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf, math.inf, 3, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, 2, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, 2, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 4],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, 2, math.inf, math.inf, 6, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, 4, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, 7, math.inf, math.inf, math.inf, 2, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 2, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, 6, math.inf, 3, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 4, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 6, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, 3, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf],
        [math.inf, 4, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf],
        [math.inf, math.inf, math.inf, math.inf, 5, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1],
        [8, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 4, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf]
    ])
    g_m = GrapheValueNonOriente(matrice_m)
    print(g_m.nb_sommets())
    print(g_m.nb_aretes())
    t0 = time.perf_counter()
    algo_m = AlgoBellmanFord(g_m)
    t_exec = time.perf_counter() - t0
    print(algo_m.distances)
    print(algo_m.predecesseurs)
    print(f"Temps de calcul : {t_exec:.5f} s")

    n = 200
    m = 350
    matrice_g = np.full((n, n), math.inf)
    edges = set()
    while len(edges) < m:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j and (i, j) not in edges and (j, i) not in edges:
            matrice_g[i][j] = 1
            matrice_g[j][i] = 1
            edges.add((i, j))
    g_g = GrapheValueNonOriente(matrice_g)
    print(g_g.nb_sommets())
    print(g_g.nb_aretes())
    t0 = time.perf_counter()
    algo_g = AlgoBellmanFord(g_g)
    t_exec = time.perf_counter() - t0
    print(algo_g.distances)
    print(algo_g.predecesseurs)
    print(f"Temps de calcul : {t_exec:.5f} s")