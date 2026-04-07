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

    def calculPCCSommet(self, s: int):
        """
        Implémentation fidèle au pseudo-code pour un sommet source s.
        """
        n = self.graphe.nb_sommets()
        # Initialisation
        d = np.full(n, np.inf)
        pred = np.full(n, None)
        d[s] = 0

        # On récupère la liste des arêtes une seule fois
        # (Chaque arête (u, v) avec poids w)
        aretes = self.graphe.get_liste_aretes()

        # Boucle principale : k = 1 jusqu'à n-1
        for k in range(1, n):
            for (u, v, poids) in aretes:
                # On traite l'arc u -> v
                if d[u] + poids < d[v]:
                    d[v] = d[u] + poids
                    pred[v] = u

                # On traite l'arc v -> u (car le graphe est non-orienté)
                if d[v] + poids < d[u]:
                    d[u] = d[v] + poids
                    pred[u] = v

        # Détection de cycle absorbant (optionnel selon votre sujet)
        for (u, v, poids) in aretes:
            if d[v] > d[u] + poids:
                print("Il existe un cycle absorbant !")
                break

        return d, pred

    def calculPCCTousSommets(self):
        """
        Lance Bellman-Ford pour chaque sommet afin de remplir les matrices globales.
        """
        n = self.graphe.nb_sommets()
        matrice_dist = np.full((n, n), np.inf)
        matrice_preds = np.full((n, n), None)

        for s in range(n):
            dist_s, pred_s = self.calculPCCSommet(s)
            matrice_dist[s] = dist_s
            matrice_preds[s] = pred_s

        return matrice_dist, matrice_preds

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

    
        