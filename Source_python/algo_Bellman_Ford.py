import numpy as np
import math
import random

from graphe_non_oriente import GrapheValueNonOriente
from algo_plus_court_chemin import AlgoPlusCourtChemin

       

########################################
# Classe AlgoBellmanFord
########################################
class AlgoBellmanFord(AlgoPlusCourtChemin):
    """
    Classe qui représente l'algorithme de Bellman-Ford, pour le calcul de plus court chemin, sur un graphe valué.
    On stocke les résultats du calcul de la distance et du prédécesseur sur le plus court chemin,
    pour chacune des paires de sommets du graphe.
    """
    
    def __init__(self, g:GrapheValueNonOriente):
        """
        Constructeur à partir d'un graphe valué non orienté.
        Les 2 matrices contenant les distances et les prédécesseurs sur les plus courts
        chemins sont initialisées dans ce constructeur.
        Paramètres :
            g : graphe valué.
        """
        super().__init__(g)

    
    def calculPCCTousSommets(self):
        """
        Calcule les plus courts chemins, en partant de chacun des sommets du graphe, et
        en allant vers chacun des sommets du graphe.
        Retour :
            (matrice des distances, matrice des prédécesseurs sur le plus court chemin).
        """
        dist = np.full((self.graphe.nb_sommets(), self.graphe.nb_sommets()), np.inf)
        for i in range(self.graphe.nb_sommets()):
            dist[i][i] = 0
        preds = np.full((self.graphe.nb_sommets(), self.graphe.nb_sommets()), None)
        liste_cc = self.graphe.calcul_cc()
        for k in range(1, self.graphe.nb_sommets()-1):
            for arc in liste_cc:
                if dist[arc[0]] + self.graphe[arc] < dist[arc[1]]:
                    dist[arc[1]] = dist[arc[0]] + self.graphe[arc]
                    preds[arc[1]] = arc[0]
        return dist, preds

# Fonction principale

if __name__ == "__main__":
    ###################################################
    ### Petit graphe : graphe de 7 sommets et 9 arêtes
    
    print("\n ##### Petit graphe #####\n")
    
    matrice = np.array([[math.inf, 1,math.inf,1,math.inf,math.inf,math.inf],
                        [1,math.inf,1,1,math.inf,math.inf,math.inf],
                        [math.inf,1,math.inf,1,1,math.inf,math.inf],
                        [1,1,1,math.inf,1,math.inf,math.inf],
                        [math.inf,math.inf,1,1,math.inf,1,math.inf],
                        [math.inf,math.inf,math.inf,math.inf,1,math.inf,1],
                        [math.inf,math.inf,math.inf,math.inf,math.inf,1,math.inf],
                        ])
    g = GrapheValueNonOriente(matrice)
    
    print("nb sommets : ", g.nb_sommets())
    print("nb arêtes : ", g.nb_aretes())
    
    
    # Utilisation de l'algorithme de Dijkstra
    algoBF = AlgoBellmanFord(g)
    print("\n Distances sur les plus courts chemins entre les sommets :\n", algoBF.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algoBF.predecesseurs)

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

    # Utilisation de l'algorithme de BellmanFord
    algoBellmanFord = AlgoBellmanFord(g)
    print("\n Distances sur les plus courts chemins entre les sommets :\n", algoBellmanFord.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algoBellmanFord.predecesseurs)

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

    # Utilisation de l'algorithme de BellmanFord
    algoBellmanFord = AlgoBellmanFord(g)
    print("\n Distances sur les plus courts chemins entre les sommets :\n", algoBellmanFord.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algoBellmanFord.predecesseurs)

    
        