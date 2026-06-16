import time

import numpy as np
import math
import os

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
    
    def __init__(self, g:GrapheValueNonOriente):
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
        """
        Calcule les plus courts chemins, en partant de chacun des sommets du graphe, et
        en allant vers chacun des sommets du graphe.
        Retour :
            (matrice des distances, matrice des prédécesseurs sur le plus court chemin).
        """
        n = self.graphe.nb_sommets()
        dist_totale = np.full((n, n), np.inf)
        preds_totale = np.full((n, n), None)
        for i in range(n):
            d, p = self.calculPCCSommet(i)
            dist_totale[i] = d
            preds_totale[i] = p
        self.distances = dist_totale
        self.predecesseurs = preds_totale

        return (dist_totale, preds_totale)

 

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
    t_debut = time.perf_counter() #Le chrono
    algoDijkstra = AlgoDijkstra(g)
    t_fin = time.perf_counter()
    temps_moyen = t_fin - t_debut
    print(f"Temps d'exécution (Petit) : {temps_moyen:.5f} secondes")
    algoDijkstra.calculPCCTousSommets()
    print("\n Distances sur les plus courts chemins entre les sommets :\n", algoDijkstra.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algoDijkstra.predecesseurs)

    ###################################################
    ### Moyen graphe : graphe de 20 sommets et 35 arêtes
    print("\n ##### Moyen graphe #####\n")
    g_moyen = GrapheValueNonOriente()
    g_moyen.lit_fichier_dot('moyen_graphe.dot')
    print("nb sommets : ", g_moyen.nb_sommets())
    print("nb arêtes : ", g_moyen.nb_aretes())
    t_debut_moyen = time.perf_counter()
    algo_moyen = AlgoDijkstra(g_moyen)
    t_fin_moyen = time.perf_counter()
    temps_moyen = t_fin_moyen - t_debut_moyen
    print(f"Temps d'exécution (Moyen) : {temps_moyen:.5f} secondes")
    algo_moyen.calculPCCTousSommets()
    print("Calcul Dijkstra moyen terminé.")
    print("\nDistances sur les plus courts chemins entre les sommets :\n", algo_moyen.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algo_moyen.predecesseurs)

    ###################################################
    ### Grand graphe : graphe de 200 sommets et 350 arêtes
    print("\n ##### Grand graphe #####\n")
    g_grand = GrapheValueNonOriente()
    g_grand.lit_fichier_dot('grand_graphe.dot')
    print("nb sommets : ", g_grand.nb_sommets())
    print("nb arêtes : ", g_grand.nb_aretes())
    t_debut_grand = time.perf_counter()
    algo_grand = AlgoDijkstra(g_grand)
    t_fin_grand = time.perf_counter()
    temps_moyen = t_fin_grand- t_debut_grand
    print(f"Temps d'exécution (Grand) : {temps_moyen:.5f} secondes")
    algo_grand.calculPCCTousSommets()
    print("\nDistances sur les plus courts chemins entre les sommets :\n", algo_grand.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algo_grand.predecesseurs)

    
        