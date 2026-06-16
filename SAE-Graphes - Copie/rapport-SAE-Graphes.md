# Rapport de la SAE Graphes -- Initiation à l'analyse de réseaux sociaux

Numéro du groupe de SAE (choisi sur Madoc) : 3-12
Nom et prénom des étudiants du binôme :

1. Outin Maxim
2. Je suis un monôme.

## Réponses aux questions

_Répondez, ci-dessous, aux questions du sujet._

### Question du (b) 4. : utilisation des 2 algorithmes de calcul de plus courts chemins, sur 3 tailles de graphes

_Commentez ici les résultats obtenus, en temps d'exécution, selon l'algorithme de calcul de plus courts chemins choisi et selon les tailles des graphes (vous pouvez ajouter des tableaux donnant les temps d'exécution de chaque type d'algorithme de calcul de plus courts chemins, pour chaque taille de graphes considérée)._

Afin de comparer les performances des algorithmes de Dijkstra et de Bellman-Ford, nous avons testé l'implémentation de la recherche des plus courts chemins pour **toutes les paires de sommets** sur trois graphes de tailles différentes :

* **Petit graphe :** 7 sommets, 9 arêtes (Matrice fournie)
* **Moyen graphe :** 20 sommets, 35 arêtes (Fichier `moyen_graphe.dot`)
* **Grand graphe :** 200 sommets, 350 arêtes (Fichier `grand_graphe.dot`)

Les mesures ont été réalisées en Python à l'aide de la fonction `time.perf_counter()`.

## Résultats


| Graphe    | Taille(Nb Sommet) | Temps Dijkstra (s) | Temps Bellman-Ford (s) |
| :-------- | :---------------- | :----------------- | :--------------------- |
| **Petit** | 7                 | 0.00013            | 0.00018                |
| **Moyen** | 20                | 0.00144            | 0.00346                |
| **Grand** | 200               | 1.18164            | 4.64411                |

## Analyse et conclusion

**Sur les petits graphes :** La différence de temps est négligeable, les deux algorithmes s'exécutent de manière quasi-instantanée.

**Sur le grand graphe :** L'algorithme de Dijkstra se montre nettement supérieur en termes de vitesse.

**Conclusion :**
Bien que Bellman-Ford soit plus polyvalent (il est capable de gérer les graphes avec des poids d'arêtes négatifs),
Dijkstra est largement préférable pour des graphes à pondération positive (comme les distances ou les temps de trajet), en particulier lorsque le nombre de sommets devient important.

### Question du (d) : analyse du réseau du club de karaté

_Commentez ici les résultats des métriques d'analyse de réseaux sociaux, pour le réseau du club de karaté ("soc-karate.dot")._

#### Métriques globales

* **Densité du graphe :** ~0.1390 Cela indique un réseau moyennement dense : les membres ne sont pas tous amis entre eux.
* **Degré moyen :** ~4.5882 En moyenne, chaque membre du club possède un peu plus de 4,5 connexions directes d'amitié au sein du groupe.
* **Diamètre :** 5 La distance maximale séparant les deux membres les plus éloignés du réseau est de 5 personnes.
* **Longueur moyenne du graphe :** ~2.408 En moyenne, deux membres du club sont séparés par environ 2,4 intermédiaires.

#### Métriques individuelles et sommets d'importance

* Le sommet 33 possède le degré le plus élevé du graphe avec 17 connexions directes. Sa proximité est très forte, beaucoup d'information passe par lui.
* Le sommet 0 présente le deuxième degré le plus élevé avec 16 connexions directes.
* Le sommet 32 avec un degré de 12 s'impose également dans ce réseau social.

#### Conclusion

On observe dans ce club de karaté deux groupes, le groupe du sommet 33 et 32 qui apparaissent commes les leaders de leur groupe.
De l'autre coté le groupe du sommet 0 qui possède également un nombre conséquent de relation.
Les autres sommets quant à eux possède se greffe entre eux selon leur lien d'amitié.

Analyse des 3 réseaux attribués

_Commentez ici les résultats des métriques d'analyse des réseaux sociaux étudiés (comment interpréter les valeurs des métriques calculées ? Quel sommet semble le plus important dans le réseau et pour quelle raison ?...)._

## Analyse du réseau social issu de Deezer : [SAE_3-12]

#### Métriques globales

- **Densité du graphe :** 0.0172 Cela indique un réseau très peu dense. Cela semble cohérent comme nous étudions une plateforme musicale, les utilisateurs se regroupent par niches et ne sont connectés qu'à une infime fraction de la base totale.
- **Degré moyen:** 8.5840 En moyenne, chaque utilisateur de cet échantillon possède un peu plus de 8.5840 connexions directes d'amitié/d'abonnement.
- **Diamètre :** 7.0 La distance maximale séparant les deux membres les plus éloignés du réseau.
- **Longueur moyenne du graphe :** 3.2647 En moyenne, deux membres quelconques du réseau sont séparés par environ 3.2647 intermédiaires.

#### Métriques individuelles et sommets d'importance

- Le sommet 14771 possède le degré le plus élevé du graphe avec 40 connexions directes. Sa proximité est très forte, beaucoup d'information passe par lui.
- Le sommet 20162 présente le deuxième degré le plus élevé avec 36 connexions.
- Le sommet 7205 avec un degré de 34 s'impose également dans ce réseau comme leader.

#### Conclusion

On observe dans ce réseau Deezer que les sommets 14771, 20162, 7205 et 9465 apparaissent comme les "leaders" de la communauté : ils représentent probablement,des artistes populaires ou des influenceurs musique. La grande majorité des autres utilisateurs ont beaucoup moins de relations et se greffent en périphérie de ces pôles majeurs, possiblement selon leurs affinités musicales.

## Analyse du réseau social issu de GitHub : [SAE_3-12]

#### Métriques globales

- **Densité du graphe :** 0.086958 La densité du graphe de GitHub est faible. Les développeurs travaillent en petites équipes sur des dépôts de code spécifiques, ce qui limite les interconnexions globales.
- **Degré moyen :** 43.392 Chaque développeur collabore ou interagit en moyenne avec ~43 autres personnes dans cet échantillon.
- **Diamètre :** 4.0 On remarque que la distance qui sépare les 2 membres les plus éloignés du réseau est de 4. On a donc un graphe très resseré.
- **Longueur moyenne du graphe :** 1.9317 La longueur moyenne des chemins nous confirme que les utilisateurs sont très peu éloignés les uns des autres. 

#### Métriques individuelles et sommets d'importance

- Le sommet nfultz est le contributeur central du réseau avec 454 collaborations. C'est énorme car il a une connexion avec la quasi-totalité du réseau.
- Le sommet dalinhuang99 le suit avec  278 connexions directes.
- Le sommet Bunlong (degré de 265) agit également comme un grand contributeur.

#### Conclusion

Le réseau GitHub est centralisé autour d'une poignée d'individus. Le sommet nfultz se détache avec 454 collaborations directes. Ce chiffre est élevé comparé au degré moyen du graphe, qui n'est que de 43.

Ce sommet principal accompagné de dalinhuang99 et Bunlong agit comme un point de connexion central pour la quasi-totalité du réseau. Cette hyper-centralisation explique les distances très courtes mesurées par l'algorithme de Dijkstra : avec un diamètre de seulement 4 et une longueur moyenne de 1,93, n'importe quel développeur de cet échantillon peut atteindre un autre membre très rapidement, le plus souvent en passant simplement par l'un des trois sommets dominants.

## Analyse du réseau social issu de Twitch : [SAE_3-12]

#### Métriques globales

- **Densité du graphe :** 0.11006 Les "viewers" d'un même streamer interagissent beaucoup entre eux dans le chat ce qui peut expliquer la densité moyenne du graphe.
- **Degré moyen :** 54.92
- **Diamètre :** 3.0
- **Longueur moyenne du graphe :** 1.9713

#### Métriques individuelles et sommets d'importance

- Le sommet 2161 est le plus populaire de ce graphe avec 253 connexions directes.
- Le sommet 771 présente le deuxième degré le plus élevé avec 226 connexions.
- Le sommet 5310 ferme le podium avec 219 relations.

#### Conclusion

Les quelques sommets qui monopolisent les degrés de connexion les plus élevés correspondent aux gros streamers de la plateforme. La quasi-totalité du reste du graphe représente les spectateurs. Contrairement à GitHub où la relation est une collaboration mutuelle, la relation ici est basée sur la diffusion de contenu, créant des communautés regroupées autour de ces quelques sommets dominant.
