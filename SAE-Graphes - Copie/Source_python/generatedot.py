import random


def generer_fichier_dot(nom_fichier: str, nb_sommets: int, nb_aretes: int, poids_max: int = 20):
    """
    Génère un fichier au format DOT contenant un graphe non orienté aléatoire.
    """
    # Vérification de sécurité pour ne pas demander plus d'arêtes que possible
    max_aretes_possibles = (nb_sommets * (nb_sommets - 1)) // 2
    if nb_aretes > max_aretes_possibles:
        nb_aretes = max_aretes_possibles
        print(f"Attention: Nombre d'arêtes réduit au maximum possible ({nb_aretes})")

    with open(nom_fichier, 'w', encoding='utf-8') as f:
        f.write("graph G {\n")

        # 1. Écriture des sommets
        for i in range(nb_sommets):
            f.write(f'  {i} [label="Sommet_{i}"];\n')

        f.write("\n")

        # 2. Génération des arêtes (en évitant les doublons et les boucles)
        aretes_creees = set()

        while len(aretes_creees) < nb_aretes:
            # On tire deux sommets au hasard
            u = random.randint(0, nb_sommets - 1)
            v = random.randint(0, nb_sommets - 1)

            # Pour un graphe non orienté, on ordonne toujours u et v
            # pour éviter d'avoir (1, 2) et (2, 1) comptés comme deux arêtes différentes
            if u > v:
                u, v = v, u

            # On vérifie que ce n'est pas une boucle (u != v) et que l'arête n'existe pas déjà
            if u != v and (u, v) not in aretes_creees:
                aretes_creees.add((u, v))
                # On génère un poids aléatoire entre 1 et poids_max
                poids = random.randint(1, poids_max)
                f.write(f'  {u} -- {v} [label="{poids}"];\n')

        f.write("}\n")

    print(f"Fichier '{nom_fichier}' généré avec succès ! ({nb_sommets} sommets, {len(aretes_creees)} arêtes)")


# ==========================================
# Génération de vos fichiers
# ==========================================
if __name__ == "__main__":
    # Moyen graphe : 20 sommets et 35 arêtes
    generer_fichier_dot("moyen_graphe.dot", 20, 35)

    # Grand graphe : 200 sommets et 350 arêtes
    generer_fichier_dot("grand_graphe.dot", 200, 350)