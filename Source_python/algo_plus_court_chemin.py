import numpy as np
import math
from graphe_non_oriente import GrapheValueNonOriente

class AlgoPlusCourtChemin:
    def __init__(self, g:GrapheValueNonOriente):
        self.graphe = g
        (self.distances, self.predecesseurs) = self.calculPCCTousSommets()

    def __str__(self):
        return str(self.distances)

    def calculPCCTousSommets(self):
        dist = np.full((self.graphe.nb_sommets(), self.graphe.nb_sommets()), np.inf)
        for i in range(self.graphe.nb_sommets()):
            dist[i][i] = 0
        preds = np.full((self.graphe.nb_sommets(), self.graphe.nb_sommets()), None)
        return (dist, preds)
