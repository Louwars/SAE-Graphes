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


# Fonction principale

if __name__ == "__main__":
    ###################################################
    ### Petit graphe : graphe de 7 sommets et 9 arêtes

    print("\n ##### Petit graphe #####\n")

    matrice = np.array([[math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf],
                        [1, math.inf, 1, 1, math.inf, math.inf, math.inf],
                        [math.inf, 1, math.inf, 1, 1, math.inf, math.inf],
                        [1, 1, 1, math.inf, 1, math.inf, math.inf],
                        [math.inf, math.inf, 1, 1, math.inf, 1, math.inf],
                        [math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1],
                        [math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf],
                        ])
    g = GrapheValueNonOriente(matrice)

    print("nb sommets : ", g.nb_sommets())
    print("nb arêtes : ", g.nb_aretes())

    # Utilisation de l'algorithme de Dijkstra
    algoD = AlgoDijkstra(g)
    print("\n Distances sur les plus courts chemins entre les sommets :\n", algoD.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algoD.predecesseurs)

    ###################################################
    ### Moyen graphe : graphe de 20 sommets et 35 arêtes

    print("\n ##### Moyen graphe #####\n")

    matrice = np.array([
        [math.inf, 1, math.inf, 1, math.inf, 5, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf,
         math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 8],
        [1, math.inf, 1, 1, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf,
         math.inf, math.inf, math.inf, math.inf, 4, math.inf, math.inf],
        [math.inf, 1, math.inf, 1, 1, math.inf, math.inf, math.inf, math.inf, math.inf, 7, math.inf, math.inf, math.inf,
         math.inf, 3, math.inf, math.inf, math.inf, math.inf],
        [1, 1, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 6, math.inf,
         math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, 1, 1, math.inf, 1, math.inf, 2, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf,
         math.inf, math.inf, math.inf, math.inf, 5, math.inf],
        [5, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf, math.inf, 3, math.inf,
         math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, 2, math.inf, math.inf,
         math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, 2, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf,
         math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 4],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, 2, math.inf,
         math.inf, 6, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf,
         math.inf, 4, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, 7, math.inf, math.inf, math.inf, 2, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf,
         math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 2, math.inf, 1, math.inf, 1,
         math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, 6, math.inf, 3, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1,
         math.inf, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 4, math.inf,
         math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 6, math.inf, math.inf,
         math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf, math.inf],
        [math.inf, math.inf, 3, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf,
         math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf],
        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf,
         math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf, math.inf],
        [math.inf, 4, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf,
         math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1, math.inf],
        [math.inf, math.inf, math.inf, math.inf, 5, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf,
         math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1],
        [8, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 4, math.inf, math.inf, math.inf, math.inf,
         math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf]
    ])
    g = GrapheValueNonOriente(matrice)

    print("nb sommets : ", g.nb_sommets())
    print("nb arêtes : ", g.nb_aretes())

    # Utilisation de l'algorithme de Dijkstra
    algoDijkstra = AlgoDijkstra(g)
    print("\n Distances sur les plus courts chemins entre les sommets :\n", algoDijkstra.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algoDijkstra.predecesseurs)

    ###################################################
    ### Grand graphe : graphe de 200 sommets et 350 arêtes

    print("\n ##### Grand graphe #####\n")
    n = 200  # sommets
    m = 350  # arêtes

    # Matrice remplie avec inf
    matrice = np.full((n, n), math.inf)

    edges = set()

    while len(edges) < m:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

        if i != j and (i, j) not in edges and (j, i) not in edges:
            matrice[i][j] = 1
            matrice[j][i] = 1  # graphe non orienté
            edges.add((i, j))
    g = GrapheValueNonOriente(matrice)

    print("nb sommets : ", g.nb_sommets())
    print("nb arêtes : ", g.nb_aretes())

    # Utilisation de l'algorithme de Dijkstra
    algoDijkstra = AlgoDijkstra(g)
    print("\n Distances sur les plus courts chemins entre les sommets :\n", algoDijkstra.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algoDijkstra.predecesseurs)
