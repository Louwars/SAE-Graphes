import numpy as np
import math
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
