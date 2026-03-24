import numpy as np
import math

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
 

    def calculPCCSommet(self, s:int):
        """
        Calcule le plus court chemin, en partant du sommet s, et
        en allant vers chacun des autres sommets du graphe.
        Retour :
            (matrice des distances, matrice des prédécesseurs sur le plus court chemin).
        """
        dist = np.full((self.graphe.nb_sommets()), np.inf)
        dist[s] = 0
        preds = np.full((self.graphe.nb_sommets()), None)
        
        M = []

        while len(M) < self.graphe.nb_sommets():
            x = self.sommet_dist_min(dist, M)
            M.append(x)

            for y in range(self.graphe.nb_sommets()):
                if y not in M :
                    vxy = self.graphe.matrice[x][y]

                    if dist[x] + vxy < dist[y]:
                        dist[y] = dist[x] + vxy
                        preds[y] = x
        
        return (dist, preds)
    
    
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
        
        for k in range(self.graphe.nb_sommets()):
            dist1, preds1 =self.calculPCCSommet(k)
            dist[k] = dist1
            preds[k] = preds1
        
        return (dist, preds)

    #Méthode ajouté par moi même

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
    algoDijkstra = AlgoDijkstra(g)
    print("\n Distances sur les plus courts chemins entre les sommets :\n", algoDijkstra.distances)
    print("\nPrédécesseurs sur les plus courts chemins entre les sommets :\n", algoDijkstra.predecesseurs)
    
    
    ###################################################
    ### Moyen graphe : graphe de 20 sommets et 35 arêtes
    
    print("\n ##### Moyen graphe #####\n")
    
    
    
    
    ###################################################
    ### Grand graphe : graphe de 200 sommets et 350 arêtes
    
    print("\n ##### Grand graphe #####\n")

