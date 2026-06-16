import random

def generer_fichier_dot(nom_fichier: str, nb_sommets: int, nb_aretes: int, poids_max: int = 20):
    max_aretes_possibles = (nb_sommets * (nb_sommets - 1)) // 2
    if nb_aretes > max_aretes_possibles:
        nb_aretes = max_aretes_possibles
        print(f"Attention: Nombre d'arêtes réduit au maximum possible ({nb_aretes})")
    with open(nom_fichier, 'w', encoding='utf-8') as f:
        f.write("graph G {\n")
        for i in range(nb_sommets):
            f.write(f'  {i} [label="Sommet_{i}"];\n')
        f.write("\n")
        aretes_creees = set()
        while len(aretes_creees) < nb_aretes:
            u = random.randint(0, nb_sommets - 1)
            v = random.randint(0, nb_sommets - 1)
            if u > v:
                u, v = v, u
            if u != v and (u, v) not in aretes_creees:
                aretes_creees.add((u, v))
                poids = random.randint(1, poids_max)
                f.write(f'  {u} -- {v} [label="{poids}"];\n')
        f.write("}\n")
    print(f"Fichier '{nom_fichier}' généré avec succès ! ({nb_sommets} sommets, {len(aretes_creees)} arêtes)")

if __name__ == "__main__":
    generer_fichier_dot("moyen_graphe.dot", 20, 35)
    generer_fichier_dot("grand_graphe.dot", 200, 350)
