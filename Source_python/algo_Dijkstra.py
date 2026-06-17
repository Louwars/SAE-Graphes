import numpy as np
import math
import random
import heapq

from graphe_non_oriente import GrapheValueNonOriente
from algo_plus_court_chemin import AlgoPlusCourtChemin


########################################
# Classe AlgoDijkstra
########################################
class AlgoDijkstra(AlgoPlusCourtChemin):
    """
    Classe qui représente l'algorithme de Dijkstra, pour le calcul de plus court chemin, sur un graphe valué.
    On stocke les résultats du calcul de la distance et du prédécesseur sur le plus court chemin,
    pour chacune des paires de sommets du graphe.
    """

    def __init__(self, g: GrapheValueNonOriente):
        """
        Constructeur à partir d'un graphe valué.
        Les 2 matrices contenant les distances et les prédécesseurs sur les plus courts
        chemins sont initialisées dans ce constructeur.
        Paramètres :
            g : graphe valué.
        """
        super().__init__(g)

    def calculPCCSommet(self, s: int):
        """
        Calcule le plus court chemin, en partant du sommet s, et
        en allant vers chacun des autres sommets du graphe.
        Retour :
            (matrice des distances, matrice des prédécesseurs sur le plus court chemin).
        """
        n = self.graphe.nb_sommets()
        # Build adjacency list locally if called individually
        adj = [[] for _ in range(n)]
        for u in range(n):
            for v in range(n):
                w = self.graphe.matrice[u][v]
                if w != np.inf:
                    adj[u].append((v, w))
        return self.calculPCCSommetWithAdj(s, adj)

    def calculPCCSommetWithAdj(self, s: int, adj):
        n = self.graphe.nb_sommets()
        dist = np.full(n, np.inf)
        dist[s] = 0
        preds = np.full(n, None)

        pq = [(0.0, s)]
        visited = [False] * n

        while pq:
            d, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True

            for v, w in adj[u]:
                if not visited[v]:
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        preds[v] = u
                        heapq.heappush(pq, (dist[v], v))

        return (dist, preds)

    def calculPCCTousSommets(self):
        """
        Calcule les plus courts chemins, en partant de chacun des sommets du graphe, et
        en allant vers chacun des sommets du graphe.
        Retour :
            (matrice des distances, matrice des prédécesseurs sur le plus court chemin).
        """
        n = self.graphe.nb_sommets()
        dist = np.full((n, n), np.inf)
        for i in range(n):
            dist[i][i] = 0
        preds = np.full((n, n), None)

        # Pre-build adjacency list once for all-pairs calculations
        adj = [[] for _ in range(n)]
        for u in range(n):
            for v in range(n):
                w = self.graphe.matrice[u][v]
                if w != np.inf:
                    adj[u].append((v, w))

        for k in range(n):
            dist1, preds1 = self.calculPCCSommetWithAdj(k, adj)
            dist[k] = dist1
            preds[k] = preds1

        return (dist, preds)


    # Méthode ajouté par moi même

    def sommet_dist_min(self, tab_distances, sommets_marques):
        x = -1

        for i in range(len(tab_distances)):
            if i not in sommets_marques:
                if x == -1:
                    x = i
                elif tab_distances[i] < tab_distances[x]:
                    x = i
        return x


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
    algoDijkstra = AlgoDijkstra(g)
    print("\n Distances sur les plus courts chemins entre les sommets :\n", algoDijkstra.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algoDijkstra.predecesseurs)

    ###################################################
    ### Moyen graphe : graphe de 20 sommets et 35 arêtes

    print("\n ##### Moyen graphe #####\n")

    g = GrapheValueNonOriente()

    g.lit_fichier_dot('../Donnees/moyen_graphe.dot')

    print("nb sommets : ", g.nb_sommets())
    print("nb arêtes : ", g.nb_aretes())

    # Utilisation de l'algorithme de Dijkstra
    algoDijkstra = AlgoDijkstra(g)
    print("\n Distances sur les plus courts chemins entre les sommets :\n", algoDijkstra.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algoDijkstra.predecesseurs)

    ###################################################
    ### Grand graphe : graphe de 200 sommets et 350 arêtes

    print("\n ##### Grand graphe #####\n")
    
    g = GrapheValueNonOriente()

    g.lit_fichier_dot('../Donnees/grand_graphe.dot')

    print("nb sommets : ", g.nb_sommets())
    print("nb arêtes : ", g.nb_aretes())

    # Utilisation de l'algorithme de Dijkstra
    algoDijkstra = AlgoDijkstra(g)
    print("\n Distances sur les plus courts chemins entre les sommets :\n", algoDijkstra.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algoDijkstra.predecesseurs)
