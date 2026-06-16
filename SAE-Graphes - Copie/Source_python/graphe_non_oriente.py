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
        
    
    def lit_fichier_dot(self, nom_fichier:str):
        """
        Lit le graphe contenu dans le fichier dont le nom est donné (au format DOT)
        et met à jour les attributs du graphe courant avec les sommets et les arêtes du graphe lu.
        Paramètres:
            nom_fichier (str): nom du fichier DOT contenant la définition d'un graphe.
        """
        # Variables pour stocker les noms des sommets et les valuations des arêtes 
        # lus dans le fichier
        noms_sommets = {}       # {id (int): label (str)}
        valuation_aretes = []   # [(id1, id2, valuation)]
        max_num_sommet = -1     # numéro de sommet le plus grand parmi ceux lus
        
        # Patrons pour les expressions régulières des noms de sommets et des arêtes
        patron_sommet = re.compile(r'^\s*(\d+)\s*\[label="([^"]+)"\]')
        patron_arete = re.compile(r'\s*(\d+)\s*--\s*(\d+)\s*\[label="([^"]+)"\]')
        
        # Lecture des lignes du fichier et traitement des différents types de lignes        
        with open(nom_fichier, "r", encoding="utf-8") as f:
            for ligne in f:
                # si la ligne correspond à un nom de sommet, sous la forme : id [label="Nom"]
                m = patron_sommet.match(ligne)
                if m:
                    id_sommet = int(m.group(1))
                    nom_sommet = m.group(2)
                    noms_sommets[id_sommet] = nom_sommet
                    max_num_sommet = max(max_num_sommet, id_sommet)
                    continue
                # si la ligne correspond à la définition d'une arête, sous la forme : id1 -- id2 [label=valuation]
                m = patron_arete.match(ligne)
                if m:
                    id_sommet1 = int(m.group(1))
                    id_sommet2 = int(m.group(2))
                    val_arete = float(m.group(3))
                    valuation_aretes.append((id_sommet1, id_sommet2, val_arete))
                    max_num_sommet = max(max_num_sommet, id_sommet1, id_sommet2)

        #print("max_num_sommet :", max_num_sommet)
            
        # Mise à jour des noms de sommets du graphe courant
        self.noms_sommets = {}
        for num_sommet in range(max_num_sommet+1):
            if num_sommet in noms_sommets:
                self.noms_sommets[num_sommet] = noms_sommets[num_sommet]
            else:
                self.noms_sommets[num_sommet] = num_sommet
                    
        # Mise à jour de la matrice de valuation du graphe courant
        self.matrice = np.full((max_num_sommet+1, max_num_sommet+1), np.inf)
        for sommet_i, sommet_j, val in valuation_aretes:
            self.matrice[sommet_i][sommet_j] = val
            self.matrice[sommet_j][sommet_i] = val #car graphe non orienté

    
    def ecrit_dans_fichier_dot(self, nom_fichier:str):
        """
        Écrit le graphe dans un fichier au format DOT.
        Paramètres :
            nom_fichier (str) : nom du fichier DOT dans lequel écrire le graphe.
        """
        with open(nom_fichier, 'w') as f:
            #print(nom_fichier)
            f.write("graph G {\n")

            for i in range(self.nb_sommets()):
                f.write(f'  {i} [label="{self.noms_sommets[i]}"];\n')
            f.write("\n")
            
            for i in range(self.nb_sommets()):
                for j in range(i, self.nb_sommets()):
                    #print(i, " -- ", j)
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
        liste_sommets = sorted(list(ens_sommets))
        taille_sous_graphe = len(liste_sommets)
        nouvelle_matrice = np.zeros((taille_sous_graphe, taille_sous_graphe))
        for i in range(taille_sous_graphe):
            for j in range(taille_sous_graphe):
                u = liste_sommets[i]
                v = liste_sommets[j]
                nouvelle_matrice[i][j] = self.matrice[u][v]
        return GrapheValueNonOriente(nouvelle_matrice)
    
    
    def calcule_cc(self):
        """
        Calcul des composantes connexes, retournées sous la forme d'une liste
        d'ensembles de numéros de sommets (chaque sous-ensemble correspond à une
        composante connexe).
        Retour:
            liste des ensembles de sommets correspondant à des composantes connexes.
        """
        sommetutilise = set()
        cc = []
        for sommet in range(self.nb_sommets()):
            if sommet not in sommetutilise:
                composanteconnexe=set()
                stack=[sommet]
                sommetutilise.add(sommet)

                while stack: #Tant qu'il rest des sommets non visité on boucle
                    scourrant=stack.pop()
                    composanteconnexe.add(scourrant)

                    for voisin,valeur in enumerate(self.matrice[scourrant]):
                        if voisin not in sommetutilise:
                            sommetutilise.add(voisin)
                            stack.append(voisin)
                cc.append(composanteconnexe)
        return cc
        
    
    def est_connexe(self):
        """
        Test de la connexité du graphe courant.
        Retour : 
            vrai si le graphe est connexe ; faux sinon.
        """
        cc = self.calcule_cc()
        return len(cc) == 1
    
        
    def plus_grosse_cc(self):
        """
        Calcule les composantes connexes du graphe et retourne le sous-graphe 
        correspondant à la plus grosse d'entre elles (en termes de nombre de sommets).
        Retour :
            le sous-graphe correspondant à la plus grosse composante connexe 
            (la numérotation des sommets n'est plus la même que dans le graphe de départ).
        """
        ccs = self.calcule_cc()
        cc_grande = None
        for cc in ccs:
            len_cc_cour = len(cc)
            if cc_grande == None or len(cc_grande) < len_cc_cour:
                cc_grande = cc
        return self.construit_sous_graphe_induit(cc)



#Fonction principale

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
    print("\nGraphe g2:\n", g2)
    print("\t degré(0) :", g2.degre_sommet(0))
    print("\t degré(1) :", g2.degre_sommet(1))
    print("\t degrés des sommets :", g2.degres_sommets())
    print("\t nb sommets :", g2.nb_sommets())
    print("\t nb arêtes :", g2.nb_aretes())
    
    print("\nEcriture du graphe dans un fichier")
    g2.ecrit_dans_fichier_dot('graphe2.dot')
    
    print("\nLecture du graphe à partir d'un fichier")
    g3 = GrapheValueNonOriente()
    g3.lit_fichier_dot('graphe2.dot')
    print(g3)