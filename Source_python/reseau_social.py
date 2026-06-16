import numpy as np
import math

from graphe_non_oriente import GrapheValueNonOriente
from algo_plus_court_chemin import AlgoPlusCourtChemin
from algo_Dijkstra import AlgoDijkstra
from algo_Bellman_Ford import AlgoBellmanFord
import networkx as nx
import matplotlib.pyplot as plt


########################################
# Classe ReseauSocial
########################################
class ReseauSocial:
    """
    Classe qui représente un réseau social, représenté par un graphe valué non orienté.
    """

    def __init__(self, g: GrapheValueNonOriente, algo: AlgoPlusCourtChemin):
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
        n = self.graphe.nb_sommets()
        if n <= 1:
            return 0
        m = self.graphe.nb_aretes()
        return (2 * m) / (n * (n - 1))

    def degre_sommet(self, s: int):
        """
        Degré du sommet s donné.
        Paramètre :
            s : indice du sommet considéré.
        Retour :
            degré du sommet s.
        """
        return self.graphe.degre_sommet(s)

    def degre_moyen_graphe(self):
        """
        Degré moyen du graphe, c'est-à-dire moyenne des degrés du graphe.
        Retour :
            degré moyen du graphe.
        """
        n = self.graphe.nb_sommets()
        if n == 0:
            return 0
        return sum(self.graphe.degres_sommets()) / n

    def proximite_sommet(self, s: int):
        """
        Proximité du sommet s donné, c'est-à-dire distance moyenne du sommet aux autres sommets du graphe.
        Paramètre :
            s : indice du sommet considéré.
        Retour :
            proximité du sommet s.
        """
        n = self.graphe.nb_sommets()
        if n <= 1:
            return 0
        distances = self.algoPCC.distances[s]
        dist_valides = [d for d in distances if d != np.inf and d != 0]
        if not dist_valides:
            return np.inf
        return sum(dist_valides) / len(dist_valides)

    def diametre_graphe(self):
        """
        Diamètre du graphe, c'est-à-dire plus grande des distances entre deux sommets du graphe.
        Retour :
            diamètre du graphe.
        """
        distances = self.algoPCC.distances.flatten()
        dist_valides = [d for d in distances if d != np.inf]
        if not dist_valides:
            return 0
        return max(dist_valides)

    def longueur_moyenne_graphe(self):
        """
        Longueur moyenne du graphe, c'est-à-dire distance moyenne pour chaque paire de sommets du graphe.
        Retour :
            longueur moyenne du graphe.
        """
        distances = self.algoPCC.distances.flatten()
        dist_valides = [d for d in distances if d != np.inf and d != 0]
        if not dist_valides:
            return 0
        return sum(dist_valides) / len(dist_valides)

    def afficher_metriques(self):
        """
        Affichage des différentes métriques du graphe. 
        Pour les métriques sur les sommets, on affiche les valeurs des métriques, pour chaque sommet
        (en triant les valeurs par valeur croissante ou décroissante, selon les métriques).
        Si le réseau n'est pas connexe, on l'indique et on affiche ces métriques sur la plus grosse composante
        connexe du réseau.
        """
        print("Affichage des métriques sur le réseau :")
        if not self.graphe.est_connexe():
            print("⚠️ Attention : Le réseau n'est pas connexe.")
            print("Calcul des métriques sur la plus grosse composante connexe...\n")
            g_cc = self.graphe.plus_grosse_cc()

            # On recrée une instance du même algorithme et un sous-réseau
            nouvel_algo = self.algoPCC.__class__(g_cc)
            reseau_cc = ReseauSocial(g_cc, nouvel_algo)
            reseau_cc.afficher_metriques()
            return

            # Métriques globales
        print(f"-> Densité du graphe      : {self.densite_graphe():.4f}")
        print(f"-> Degré moyen            : {self.degre_moyen_graphe():.4f}")
        print(f"-> Diamètre               : {self.diametre_graphe()}")
        print(f"-> Longueur moy. chemins  : {self.longueur_moyenne_graphe():.4f}\n")

        # Métriques individuelles (Degrés)
        n = self.graphe.nb_sommets()
        degres = [(i, self.degre_sommet(i)) for i in range(n)]
        degres.sort(key=lambda x: x[1], reverse=True)  # Tri décroissant

        print("--- Degrés des sommets (Tri décroissant) ---")
        for s, d in degres:
            print(f"Sommet {s:2d} : {d}")

        # Métriques individuelles (Proximité)
        proximites = [(i, self.proximite_sommet(i)) for i in range(n)]
        proximites.sort(key=lambda x: x[1])  # Tri croissant (plus la distance est faible, plus il est "proche")

        print("\n--- Proximité des sommets (Distance moyenne, Tri croissant) ---")
        for s, p in proximites:
            print(f"Sommet {s:2d} : {p:.4f}")

    def visualiser_reseau(self):
        """
        Affichage du réseau social, en utilisant les métriques calculées pour modifier l'affichage de la taille
        des sommets et/ou arêtes.
        """
        print("Affichage du réseau social et de ses métriques")

        G = nx.Graph()
        n = self.graphe.nb_sommets()

        #Ajout des sommets et des arêtes
        for i in range(n):
            G.add_node(i)
            for j in range(i + 1, n):
                poids = self.graphe.matrice[i][j]
                if poids != np.inf:
                    G.add_edge(i, j, weight=poids)

        #Calcul de la taille des nœuds basée sur leur degré (ex: degré * 300)
        node_sizes = [self.degre_sommet(i) * 300 for i in range(n)]
        #Les nœuds de degré 0 auront une taille minimale
        node_sizes = [size if size > 0 else 100 for size in node_sizes]

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)  #Layout esthétique

        nx.draw(G, pos,
                with_labels=True,
                node_size=node_sizes,
                node_color='lightblue',
                edge_color='gray',
                font_size=10,
                font_weight='bold')

        plt.title(f"Visualisation du Réseau Social (Densité: {self.densite_graphe():.2f})")
        plt.show()


### Autres fonctions

def analyse_reseau_cours_Centrale_Supelec():
    """
    Création et analyse du réseau social de l'exemple donné dans les slides 
    du cours de Centrale Supélec, en affichant le résultat des différentes métriques
    sur ce réseau social.
    """
    matrice = np.array([[math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf],
                        [1, math.inf, 1, 1, math.inf, math.inf, math.inf],
                        [math.inf, 1, math.inf, 1, 1, math.inf, math.inf],
                        [1, 1, 1, math.inf, 1, math.inf, math.inf],
                        [math.inf, math.inf, 1, 1, math.inf, 1, math.inf],
                        [math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1],
                        [math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf],
                        ])
    g = GrapheValueNonOriente(matrice)
    algo = AlgoDijkstra(g)
    rs = ReseauSocial(g, algo)
    rs.afficher_metriques()
    rs.visualiser_reseau()


def analyse_reseau_club_karate():
    """
    Création et analyse du réseau social de l'exemple du club de karaté, 
    en affichant le résultat des différentes métriques sur ce réseau social.
    """
    print("\nAnalyse du réseau du club de karaté")
    g = GrapheValueNonOriente()
    g.lit_fichier_dot('../Donnees/soc-karate.dot')
    algo = AlgoDijkstra(g)
    rs = ReseauSocial(g, algo)
    rs.afficher_metriques()
    rs.visualiser_reseau()


def analyse_reseau_deezer():
    """
    Création et analyse du réseau social issu de Deezer, en affichant le résultat 
    des différentes métriques sur ce réseau social (uniquement les métriques globales
    au graphe et les métriques individuelles pour les 10 sommets ayant les degrés les 
    plus élevés).
    """
    print("\n--- Analyse du réseau issu de Deezer ---")
    g = GrapheValueNonOriente()
    g.lit_fichier_dot('../Donnees/reseau_deezer_SAE_3-1.dot')
    algo = AlgoDijkstra(g)
    rs = ReseauSocial(g, algo)

    print(f"Densité du graphe : {rs.densite_graphe():.6f}")
    print(f"Degré moyen : {rs.degre_moyen_graphe():.4f}")
    print(f"Diamètre : {rs.diametre_graphe()}")
    print(f"Longueur moyenne des chemins : {rs.longueur_moyenne_graphe():.4f}\n")

    n = rs.graphe.nb_sommets()
    degres = [(i, rs.degre_sommet(i)) for i in range(n)]
    degres.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 des sommets (Degrés les plus élevés) :")
    for i in range(min(10, n)):
        s, d = degres[i]
        nom = rs.graphe.noms_sommets.get(s, s)
        proximite = rs.proximite_sommet(s)
        print(f"Sommet {nom} : Degré = {d} | Proximité = {proximite:.4f}")


def analyse_reseau_git():
    """
    Création et analyse du réseau social issu de GitHub, en affichant le résultat 
    des différentes métriques sur ce réseau social (uniquement les métriques globales
    au graphe et les métriques individuelles pour les 10 sommets ayant les degrés les 
    plus élevés).
    """
    print("\n--- Analyse du réseau issu de GitHub ---")
    g = GrapheValueNonOriente()
    g.lit_fichier_dot('../Donnees/reseau_github_SAE_3-1.dot')
    algo = AlgoDijkstra(g)
    rs = ReseauSocial(g, algo)

    print(f"Densité du graphe : {rs.densite_graphe():.6f}")
    print(f"Degré moyen : {rs.degre_moyen_graphe():.4f}")
    print(f"Diamètre : {rs.diametre_graphe()}")
    print(f"Longueur moyenne des chemins : {rs.longueur_moyenne_graphe():.4f}\n")

    n = rs.graphe.nb_sommets()
    degres = [(i, rs.degre_sommet(i)) for i in range(n)]
    degres.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 des sommets (Degrés les plus élevés) :")
    for i in range(min(10, n)):
        s, d = degres[i]
        nom = rs.graphe.noms_sommets.get(s, s)
        proximite = rs.proximite_sommet(s)
        print(f"Sommet {nom} : Degré = {d} | Proximité = {proximite:.4f}")


def analyse_reseau_twitch():
    """
    Création et analyse du réseau social issu de Twitch, en affichant le résultat 
    des différentes métriques sur ce réseau social (uniquement les métriques globales
    au graphe et les métriques individuelles pour les 10 sommets ayant les degrés les 
    plus élevés).
    """
    print("\n--- Analyse du réseau issu de Twitch ---")
    g = GrapheValueNonOriente()
    g.lit_fichier_dot('../Donnees/reseau_twitch_SAE_3-1.dot')
    algo = AlgoDijkstra(g)
    rs = ReseauSocial(g, algo)

    print(f"Densité du graphe : {rs.densite_graphe():.6f}")
    print(f"Degré moyen : {rs.degre_moyen_graphe():.4f}")
    print(f"Diamètre : {rs.diametre_graphe()}")
    print(f"Longueur moyenne des chemins : {rs.longueur_moyenne_graphe():.4f}\n")

    n = rs.graphe.nb_sommets()
    degres = [(i, rs.degre_sommet(i)) for i in range(n)]
    degres.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 des sommets (Degrés les plus élevés) :")
    for i in range(min(10, n)):
        s, d = degres[i]
        nom = rs.graphe.noms_sommets.get(s, s)
        proximite = rs.proximite_sommet(s)
        print(f"Sommet {nom} : Degré = {d} | Proximité = {proximite:.4f}")


# Fonction principale

if __name__ == "__main__":
    matrice = np.array([[math.inf, 1, math.inf, 1, math.inf, math.inf, math.inf],
                        [1, math.inf, 1, 1, math.inf, math.inf, math.inf],
                        [math.inf, 1, math.inf, 1, 1, math.inf, math.inf],
                        [1, 1, 1, math.inf, 1, math.inf, math.inf],
                        [math.inf, math.inf, 1, 1, math.inf, 1, math.inf],
                        [math.inf, math.inf, math.inf, math.inf, 1, math.inf, 1],
                        [math.inf, math.inf, math.inf, math.inf, math.inf, 1, math.inf],
                        ])
    g = GrapheValueNonOriente(matrice)
    r = ReseauSocial(g, AlgoDijkstra(g))
    print(r)
    r.afficher_metriques()
    r.visualiser_reseau()
    analyse_reseau_club_karate()
    analyse_reseau_deezer()
    analyse_reseau_git()
    analyse_reseau_twitch()
