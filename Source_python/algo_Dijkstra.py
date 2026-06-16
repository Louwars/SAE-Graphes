import numpy as np
import math
import random
from graphe_non_oriente import GrapheValueNonOriente
from algo_plus_court_chemin import AlgoPlusCourtChemin

class AlgoDijkstra(AlgoPlusCourtChemin):
    def __init__(self, g:GrapheValueNonOriente):
        super().__init__(g)

    def calculPCCSommet(self, s:int):
        n = self.graphe.nb_sommets()
        dist = np.full(n, np.inf)
        dist[s] = 0
        preds = np.full(n, None, dtype=object)
        visites = [False] * n
        for _ in range(n):
            u = -1
            min_dist = np.inf
            for i in range(n):
                if not visites[i] and dist[i] < min_dist:
                    min_dist = dist[i]
                    u = i
            if u == -1:
                break
            visites[u] = True
            for v in range(n):
                poids_uv = self.graphe.matrice[u][v]
                if poids_uv != np.inf:
                    if dist[u] + poids_uv < dist[v]:
                        dist[v] = dist[u] + poids_uv
                        preds[v] = u
        return (dist, preds)

    def calculPCCTousSommets(self):
        n = self.graphe.nb_sommets()
        dist = np.full((n, n), np.inf)
        preds = np.full((n, n), None)
        for k in range(n):
            dist1, preds1 = self.calculPCCSommet(k)
            dist[k] = dist1
            preds[k] = preds1
        self.distances = dist
        self.predecesseurs = preds
        return (dist, preds)

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
    algoDijkstra = AlgoDijkstra(g)
    print(algoDijkstra.distances)
    print(algoDijkstra.predecesseurs)

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
    algo_m = AlgoDijkstra(g_m)
    print(algo_m.distances)
    print(algo_m.predecesseurs)

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
    algo_g = AlgoDijkstra(g_g)
    print(algo_g.distances)
    print(algo_g.predecesseurs)
