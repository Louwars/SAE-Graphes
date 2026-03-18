import numpy as np
import math

from graphe_non_oriente import GrapheValueNonOriente



########################################
# Classe AlgoPlusCourtChemin
########################################
class AlgoPlusCourtChemin:
    """
    Classe abstraite qui représente un algorithme de calcul de plus court chemin, sur un graphe valué.
    On stocke les résultats du calcul de la distance et du prédécesseur sur le plus court chemin,
    pour chacune des paires de sommets du graphe.
    --> cette classe n'est pas à modifier.
    """
    
    def __init__(self, g:GrapheValueNonOriente):
        """
        Constructeur à partir d'un graphe valué non orienté.
        Les 2 matrices contenant les distances et les prédécesseurs sur les plus courts
        chemins sont initialisées dans ce constructeur.
        Paramètres :
            g : graphe valué.
        """
        self.graphe = g
        (self.distances, self.predecesseurs) = self.calculPCCTousSommets()
 

    def __str__(self):
        """
        Représentation de l'algorithme de calcul de plus court chemin, par la matrice des distances.
        Retour :
            chaîne de caractères contenant les valeurs des distances.
        """
        return str(self.distances)

    
    def calculPCCTousSommets(self):
        """
        Calcule les plus courts chemins, en partant de chacun des sommets du graphe, et
        en allant vers chacun des sommets du graphe.
        Retour :
            (matrice des distances, matrice des prédécesseurs sur le plus court chemin).
        """
        


