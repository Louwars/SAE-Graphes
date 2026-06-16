import numpy as np
import math
import re
import collections

class GrapheValueNonOriente:
    def __init__(self, mat=[], noms=None):
        self.matrice = mat
        if noms is None:
            self.noms_sommets = {x: x for x in range(len(self.matrice))}
        else:
            self.noms_sommets = noms

    def __str__(self):
        res_str = "Noms des sommets :\n" + str(self.noms_sommets) + "\n"
        res_str += "Matrice de valuation du graphe :\n" + str(self.matrice) + "\n"
        return res_str

    def lit_fichier_dot(self, nom_fichier:str):
        noms_sommets = {}
        valuation_aretes = []
        max_num_sommet = -1
        patron_sommet = re.compile(r'^\s*(\d+)\s*\[label="?([^"\]\s;]+)"?\]')
        patron_arete = re.compile(r'\s*(\d+)\s*--\s*(\d+)\s*\[label="?([^"\]\s;]+)"?\]')
        with open(nom_fichier, "r", encoding="utf-8") as f:
            for ligne in f:
                m = patron_sommet.match(ligne)
                if m:
                    id_sommet = int(m.group(1))
                    nom_sommet = m.group(2)
                    noms_sommets[id_sommet] = nom_sommet
                    max_num_sommet = max(max_num_sommet, id_sommet)
                    continue
                m = patron_arete.match(ligne)
                if m:
                    id_sommet1 = int(m.group(1))
                    id_sommet2 = int(m.group(2))
                    val_arete = float(m.group(3))
                    valuation_aretes.append((id_sommet1, id_sommet2, val_arete))
                    max_num_sommet = max(max_num_sommet, id_sommet1, id_sommet2)
        self.noms_sommets = {}
        for num_sommet in range(max_num_sommet+1):
            if num_sommet in noms_sommets:
                self.noms_sommets[num_sommet] = noms_sommets[num_sommet]
            else:
                self.noms_sommets[num_sommet] = num_sommet
        self.matrice = np.full((max_num_sommet+1, max_num_sommet+1), np.inf)
        for sommet_i, sommet_j, val in valuation_aretes:
            self.matrice[sommet_i][sommet_j] = val
            self.matrice[sommet_j][sommet_i] = val

    def ecrit_dans_fichier_dot(self, nom_fichier:str):
        with open(nom_fichier, 'w') as f:
            f.write("graph G {\n")
            for i in range(self.nb_sommets()):
                f.write(f'  {i} [label="{self.noms_sommets[i]}"];\n')
            f.write("\n")
            for i in range(self.nb_sommets()):
                for j in range(i, self.nb_sommets()):
                    if self.matrice[i, j] != math.inf:
                        f.write(f'  {i} -- {j} [label="{self.matrice[i, j]}"];\n')
            f.write("}\n")

    def nb_sommets(self):
        return len(self.matrice)

    def nb_aretes(self):
        return ((self.nb_sommets())**2 - collections.Counter(self.matrice.flatten())[math.inf]) // 2

    def degre_sommet(self, s:int):
        return self.nb_sommets() - collections.Counter(self.matrice[s,:])[math.inf]

    def degres_sommets(self):
        return [self.degre_sommet(i) for i in range(self.nb_sommets())]

    def construit_sous_graphe_induit(self, ens_sommets:set):
        liste_sommets = sorted(list(ens_sommets))
        taille_sous_graphe = len(liste_sommets)
        nouvelle_matrice = np.zeros((taille_sous_graphe, taille_sous_graphe))
        for i in range(taille_sous_graphe):
            for j in range(taille_sous_graphe):
                u = liste_sommets[i]
                v = liste_sommets[j]
                nouvelle_matrice[i][j] = self.matrice[u][v]
        return GrapheValueNonOriente(nouvelle_matrice)

    def graphe_symetrique(self, graphe):
        sym = np.zeros((self.nb_sommets(), self.nb_sommets()))
        for i in range(len(graphe)):
            for j in range(len(graphe)):
                if graphe[i][j] == 1:
                    sym[i][j] = 1
                    sym[j][i] = 1
        return GrapheValueNonOriente(sym)

    def calcule_cc(self):
        sommetutilise = set()
        cc = []
        for sommet in range(self.nb_sommets()):
            if sommet not in sommetutilise:
                composanteconnexe=set()
                stack=[sommet]
                sommetutilise.add(sommet)
                while stack:
                    scourrant=stack.pop()
                    composanteconnexe.add(scourrant)
                    for voisin, valeur in enumerate(self.matrice[scourrant]):
                        if valeur != math.inf and voisin not in sommetutilise:
                            sommetutilise.add(voisin)
                            stack.append(voisin)
                cc.append(composanteconnexe)
        return cc

    def est_connexe(self):
        cc = self.calcule_cc()
        return len(cc) == 1

    def plus_grosse_cc(self):
        ccs = self.calcule_cc()
        cc_grande = None
        for cc in ccs:
            len_cc_cour = len(cc)
            if cc_grande == None or len(cc_grande) < len_cc_cour:
                cc_grande = cc
        return self.construit_sous_graphe_induit(cc_grande)

    def get_liste_aretes(self):
        aretes = []
        n = self.nb_sommets()
        for i in range(n):
            for j in range(i + 1, n):
                poids = self.matrice[i][j]
                if poids != math.inf:
                    aretes.append((i, j, poids))
        return aretes

if __name__ == "__main__":
    m2 = np.array([[math.inf, 3, 2.5, 8],
                   [3,math.inf,math.inf,7],
                   [2.5,math.inf,math.inf,1.5],
                   [8,7,1.5,math.inf],
                  ])
    g2 = GrapheValueNonOriente(m2, {0:"Teddy", 1:"Lisa", 2:"Mohamed", 3:"Levi"})
    print(g2)
    g2.ecrit_dans_fichier_dot('graphe2.dot')
