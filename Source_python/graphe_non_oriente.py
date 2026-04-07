import numpy as np
import math
import re
import collections



########################################
# Classe GrapheValue
########################################
class GrapheValueNonOriente:
    """
    Classe qui représente des graphes valués non orientés.
    Les graphes valués sont représentés par leur matrice de valuation.
    Les sommets sont numérotés de 0 à n-1 (n étant le nombre de sommets).
    On peut également indiquer les noms des sommets du graphe, définis dans un dictionnaire.
    """
    
    def __init__(self, mat=[], noms=None):
        """
        Constructeur d'un graphe valué, à partir de sa matrice de valuation et des noms des sommets.
        Paramètres :
            mat : matrice de valuation du graphe.
            noms : dictionnaire qui associe son nom à chaque numéro de sommet.
        """
        self.matrice = mat
        if noms is None:
            self.noms_sommets = {x: x for x in range(len(self.matrice))}
        else:
            self.noms_sommets = noms
    
 

    def __str__(self):
        """
        Représentation du graphe valué, par une chaîne de caractères.
        Retour :
            chaîne de caractères contenant les noms des sommets et les valeurs de la matrice 
            de valuation du graphe.
        """
        res_str = "Noms des sommets :\n" + str(self.noms_sommets) + "\n"
        res_str += "Matrice de valuation du graphe :\n" + str(self.matrice) + "\n"
        return res_str
    
    
    def ecrit_dans_fichier_dot(self, nom_fichier:str):
        """
        Écrit le graphe dans un fichier au format DOT.
        Paramètres :
            nom_fichier (str) : nom du fichier DOT dans lequel écrire le graphe.
        """
        with open(nom_fichier, 'w') as f:
            print(nom_fichier)
            f.write("graph G {\n")

            for i in range(self.nb_sommets()):
                f.write(f'  {i} [label="{self.noms_sommets[i]}"];\n')
            f.write("\n")
            
            for i in range(self.nb_sommets()):
                for j in range(i, self.nb_sommets()):
                    print(i, " -- ", j)
                    if self.matrice[i, j] != math.inf:  # math.inf signifie absence d'arête
                        f.write(f'  {i} -- {j} [label="{self.matrice[i, j]}"];\n')
        
            f.write("}\n")

    
    def nb_sommets(self):
        """
        Calcule le nombre de sommets du graphe.
        Retour : 
            nombre de sommets du graphe.
        """
        return len(self.matrice)
    
    
    def nb_aretes(self):
        """
        Calcul le nombre d'arêtes du graphe.
        Retour : 
            nombre d'arêtes du graphe.
        """
        return ((self.nb_sommets())**2 - collections.Counter(self.matrice.flatten())[math.inf]) // 2
    
    
    def degre_sommet(self, s:int):
        """
        Calul du degré du sommet d'indice donné.
        Paramètres :
            s : indice du sommet considéré.
        Retour : 
            degré du sommet s.
        """
        return self.nb_sommets() - collections.Counter(self.matrice[s,:])[math.inf]


    def degres_sommets(self):
        """
        Calcul de la liste des degrés des sommets du graphe.
        Retour : 
            liste des degrés des sommets du graphe.
        """
        return [self.degre_sommet(i) for i in range(self.nb_sommets())]
    
    
    def construit_sous_graphe_induit(self, ens_sommets:set):
        """
        Construction du sous-graphe induit, à partir de l'ensemble de sommets donné.
        Paramètres :
            ens_sommets : ensemble de sommets à partir duquel construire le sous-graphe.
        """
        array = self.matrice
        array2 = np.zeros((len(ens_sommets), (len(ens_sommets))))
        i = 0
        for elem in ens_sommets:
            j = 0
            for ele2 in ens_sommets:
                array2[i][j] = array[elem][ele2]
                j +=1
            i+=1
        return GrapheValueNonOriente(array2)



    def graphe_symetrique(self, graphe):
        sym = np.zeros((self.nb_sommets(), self.nb_sommets()))
        for i in range(len(graphe)):
            for j in range(len(graphe)):
                if graphe[i][j] == 1:
                    sym[i][j] = 1
                    sym[j][i] = 1
        return GrapheValueNonOriente(sym)

    
    
    def calcule_cc(self):
        """
        Calcul des composantes connexes, retournées sous la forme d'une liste
        d'ensembles de numéros de sommets (chaque sous-ensemble correspond à une
        composante connexe).
        Retour:
            liste des ensembles de sommets correspondant à des composantes connexes.
        """
        tab = []
        for i in range(self.nb_sommets()):
            for j in range(self.nb_sommets()):
                if self.matrice[i][j] != math.inf:
                    tab.append([i, j])
        return tab

        
    
    def est_connexe(self):
        """
        Test de la connexité du graphe courant.
        Retour : 
            vrai si le graphe est connexe ; faux sinon.
        """
        # calcul des composantes connexes du graphes
        cc = self.calcule_cc()
        
        # graphe connexe si une seule composante connexe
        return len(cc) == 1
    
        
    def plus_grosse_cc(self):
        """
        Calcule les composantes connexes du graphe et retourne le sous-graphe 
        correspondant à la plus grosse d'entre elles (en termes de nombre de sommets).
        Retour :
            le sous-graphe correspondant à la plus grosse composante connexe 
            (la numérotation des sommets n'est plus la même que dans le graphe de départ).
        """
        # calcul des composantes connexes
        ccs = self.calcule_cc()
        
        # sélection de l'ensemble le plus grand
        cc_grande = None
        for cc in ccs:
            len_cc_cour = len(cc)
            if cc_grande == None or len(cc_grande) < len_cc_cour:
                cc_grande = cc
            
        # construction du sous-graphe induit correspondant
        return self.construit_sous_graphe_induit(cc)

    def get_liste_aretes(self):
        """
        Parcourt la matrice de valuation et retourne la liste de toutes les arêtes.
        Retour :
            liste de triplets (u, v, poids)
        """
        aretes = []
        n = self.nb_sommets()
        for i in range(n):
            # On commence à i+1 pour ne pas compter les arêtes en double (graphe non orienté)
            # et éviter la diagonale (distance d'un sommet à lui-même).
            for j in range(i + 1, n):
                poids = self.matrice[i][j]
                if poids != math.inf:
                    aretes.append((i, j, poids))
        return aretes



# Fonction principale

if __name__ == "__main__":
    test = np.full([3], math.inf)
    print(test, "\n")

    
    m = np.full([3,3], math.inf)
    g = GrapheValueNonOriente(m)
    print("Graphe g:\n", g)

    
    m2 = np.array([[math.inf, 3, 2.5, 8],
                   [3,math.inf,math.inf,7],
                   [2.5,math.inf,math.inf,1.5],
                   [8,7,1.5,math.inf],
                  ])
    g2 = GrapheValueNonOriente(m2, {0:"Teddy", 1:"Lisa", 2:"Mohamed", 3:"Levi"})
    print("Graphe g2:\n", g2)
    print(g2.calcule_cc())
    ''' print("\nGraphe g2:\n", g2)
    print("\t degré(0) :", g2.degre_sommet(0))
    print("\t degré(1) :", g2.degre_sommet(1))
    print("\t degrés des sommets :", g2.degres_sommets())
    print("\t nb sommets :", g2.nb_sommets())
    print("\t nb arêtes :", g2.nb_aretes())

    test = set([0,1,3])
    print("\t test", g2.construit_sous_graphe_induit(test))
    print("\nEcriture du graphe dans un fichier")
    g2.ecrit_dans_fichier_dot('graphe2.dot')'''

