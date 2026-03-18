import numpy as np
import math

from graphe_non_oriente import GrapheValueNonOriente
from algo_plus_court_chemin import AlgoPlusCourtChemin
from algo_Dijkstra import AlgoDijkstra
from algo_Bellman_Ford import AlgoBellmanFord



########################################
# Classe ReseauSocial
########################################
class ReseauSocial:
    """
    Classe qui représente un réseau social, représenté par un graphe valué non orienté.
    """
    
    def __init__(self, g:GrapheValueNonOriente, algo:AlgoPlusCourtChemin):
        """
        Constructeur à partir d'un graphe valué et d'un algorithme de calcul de plus court chemin.
        Paramètres :
            g : graphe valué.
            algo : algorithme de calcul de plus court chemin.
        """
        self.graphe = g
        self.algoPCC = algo
 

    def __str__(self):
        """
        Représentation du réseau social
        Retour :
            chaîne de caractères représentant le graphe du réseau social.
        """
        return str(self.graphe)
    
    
    def densite_graphe(self):
        """
        Densité du graphe, c'est-à-dire ratio du nombre d'arêtes sur le nombre d'arêtes d'un graphe complet.
        Retour :
            densité du graphe.
        """
        #TODO : à compléter
        return 0  
    
    
    def degre_sommet(self, s:int):
        """
        Degré du sommet s donné.
        Paramètre :
            s : indice du sommet considéré.
        Retour :
            degré du sommet s.
        """
        #TODO : à compléter
        return 0
    
    
    def degre_moyen_graphe(self):
        """
        Degré moyen du graphe, c'est-à-dire moyenne des degrés du graphe.
        Retour :
            degré moyen du graphe.
        """
        #TODO : à compléter
        return 0       
        
    
    def proximite_sommet(self, s:int):
        """
        Proximité du sommet s donné, c'est-à-dire distance moyenne du sommet aux autres sommets du graphe.
        Paramètre :
            s : indice du sommet considéré.
        Retour :
            proximité du sommet s.
        """
        #TODO : à compléter
        return 0
    
    
    def diametre_graphe(self):
        """
        Diamètre du graphe, c'est-à-dire plus grande des distances entre deux sommets du graphe.
        Retour :
            diamètre du graphe.
        """
        #TODO : à compléter
        return 0    
    
    
    def longueur_moyenne_graphe(self):
        """
        Longueur moyenne du graphe, c'est-à-dire distance moyenne pour chaque paire de sommets du graphe.
        Retour :
            longueur moyenne du graphe.
        """
        #TODO : à compléter
        return 0   
    
    
    def afficher_metriques(self):
        """
        Affichage des différentes métriques du graphe. 
        Pour les métriques sur les sommets, on affiche les valeurs des métriques, pour chaque sommet
        (en triant les valeurs par valeur croissante ou décroissante, selon les métriques).
        Si le réseau n'est pas connexe, on l'indique et on affiche ces métriques sur la plus grosse composante
        connexe du réseau.
        """  
        print("Affichage des métriques sur le réseau :")
        #TODO : à compléter
        
        
        
    def visualiser_reseau(self):
        """
        Affichage du réseau social, en utilisant les métriques calculées pour modifier l'affichage de la taille
        des sommets et/ou arêtes.
        """
        print("Affichage du réseau social et de ses métriques")
        #TODO : à compléter



### Autres fonctions

def analyse_reseau_cours_Centrale_Supelec():
    """
    Création et analyse du réseau social de l'exemple donné dans les slides 
    du cours de Centrale Supélec, en affichant le résultat des différentes métriques
    sur ce réseau social.
    """
    matrice = np.array([[math.inf, 1,math.inf,1,math.inf,math.inf,math.inf],
                        [1,math.inf,1,1,math.inf,math.inf,math.inf],
                        [math.inf,1,math.inf,1,1,math.inf,math.inf],
                        [1,1,1,math.inf,1,math.inf,math.inf],
                        [math.inf,math.inf,1,1,math.inf,1,math.inf],
                        [math.inf,math.inf,math.inf,math.inf,1,math.inf,1],
                        [math.inf,math.inf,math.inf,math.inf,math.inf,1,math.inf],
                        ])
    #TODO : à compléter
    
    

def analyse_reseau_club_karate():
    """
    Création et analyse du réseau social de l'exemple du club de karaté, 
    en affichant le résultat des différentes métriques sur ce réseau social.
    """
    #TODO : à compléter





# Fonction principale

if __name__ == "__main__":
    matrice = np.array([[math.inf, 1,math.inf,1,math.inf,math.inf,math.inf],
                        [1,math.inf,1,1,math.inf,math.inf,math.inf],
                        [math.inf,1,math.inf,1,1,math.inf,math.inf],
                        [1,1,1,math.inf,1,math.inf,math.inf],
                        [math.inf,math.inf,1,1,math.inf,1,math.inf],
                        [math.inf,math.inf,math.inf,math.inf,1,math.inf,1],
                        [math.inf,math.inf,math.inf,math.inf,math.inf,1,math.inf],
                        ])
    g = GrapheValueNonOriente(matrice)
    r = ReseauSocial(g, AlgoDijkstra(g))
    print(r)
    r.afficher_metriques()
    r.visualiser_reseau()
