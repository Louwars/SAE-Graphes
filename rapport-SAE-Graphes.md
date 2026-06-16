# Rapport de la SAE Graphes -- Initiation à l'analyse de réseaux sociaux

Nom et prénom des étudiants du binôme :
1. Cheminard Noah
2. Leleu Louis 

## Réponses aux questions

### Question du (b) 4. : utilisation des 2 algorithmes de calcul de plus courts chemins, sur 3 tailles de graphes

Afin de comparer les performances des algorithmes de Dijkstra et de Bellman-Ford, nous avons testé l'implémentation de la recherche des plus courts chemins pour toutes les paires de sommets sur trois graphes de tailles différentes :

* Petit graphe : 7 sommets, 9 arêtes (Matrice fournie)
* Moyen graphe : 20 sommets, 35 arêtes (Fichier graph-moyen.dot)
* Grand graphe : 200 sommets, 350 arêtes

Les mesures ont été réalisées en Python à l'aide de la fonction time.perf_counter().

## Résultats

| Graphe | Taille (Nb Sommets) | Temps Dijkstra (s) | Temps Bellman-Ford (s) |
| :--- | :--- | :--- | :--- |
| Petit | 7 | 0.00010 | 0.00011 |
| Moyen | 20 | 0.00128 | 0.00282 |
| Grand | 200 | 0.94150 | 2.53079 |

## Analyse et conclusion

Sur les petits graphes : La différence de temps est négligeable, les deux algorithmes s'exécutent de manière quasi-instantanée.

Sur le grand graphe : L'algorithme de Dijkstra se montre nettement supérieur en termes de vitesse.

Conclusion :
Bien que Bellman-Ford soit plus polyvalent car capable de gérer les graphes avec des poids d'arêtes négatifs, Dijkstra est largement préférable pour des graphes à pondération positive, en particulier lorsque le nombre de sommets devient important.

### Question du (d) : analyse du réseau du club de karaté

#### Métriques globales

* Densité du graphe : 0.1390
Cela indique un réseau moyennement dense : les membres ne sont pas tous amis entre eux.
* Degré moyen : 4.5882
En moyenne, chaque membre du club possède un peu plus de 4.5 connexions directes d'amitié au sein du groupe.
* Diamètre : 5.0
La distance maximale séparant les deux membres les plus éloignés du réseau est de 5 personnes.
* Longueur moyenne du graphe : 2.4082
En moyenne, deux membres du club sont séparés par environ 2.4 intermédiaires.

#### Métriques individuelles et sommets d'importance

* Le sommet 33 possède le degré le plus élevé du graphe avec 17 connexions directes. Sa proximité est très forte, beaucoup d'informations passent par lui.
* Le sommet 0 présente le deuxième degré le plus élevé avec 16 connexions directes.
* Le sommet 32 avec un degré de 12 s'impose également dans ce réseau social.

#### Conclusion

On observe dans ce club de karaté deux groupes, le groupe du sommet 33 et 32 qui apparaissent comme les leaders de leur groupe. De l'autre côté, le groupe du sommet 0 possède également un nombre conséquent de relations. Les autres sommets se greffent selon leurs liens d'amitié.

### Question du (e) : analyse des 3 réseaux attribués

## Analyse du réseau social issu de Deezer : [SAE_3-6]

#### Métriques globales

- Densité du graphe : 0.017403
Cela indique un réseau très peu dense. Comme nous étudions une plateforme musicale, les utilisateurs se regroupent par niches et ne sont connectés qu'à une infime fraction de la base totale.
- Degré moyen : 8.6840
En moyenne, chaque utilisateur de cet échantillon possède environ 8.6 connexions directes d'amitié/d'abonnement.
- Diamètre : 7.0
La distance maximale séparant les deux membres les plus éloignés du réseau.
- Longueur moyenne du graphe : 3.2462
En moyenne, deux membres quelconques du réseau sont séparés par environ 3.2 intermédiaires.

#### Métriques individuelles et sommets d'importance

- Le sommet 14771 possède le degré le plus élevé du graphe avec 40 connexions directes. Sa proximité est très forte (distance moyenne de 2.3464), beaucoup d'informations passent par lui.
- Le sommet 7205 présente le deuxième degré le plus élevé avec 36 connexions.
- Le sommet 20162 avec un degré de 36 s'impose également dans ce réseau comme leader.

#### Conclusion

On observe dans ce réseau Deezer que les sommets 14771, 7205 et 20162 apparaissent comme les leaders de la communauté. La grande majorité des autres utilisateurs ont beaucoup moins de relations et se greffent en périphérie de ces pôles majeurs, possiblement selon leurs affinités musicales.

## Analyse du réseau social issu de GitHub : [SAE_3-6]

#### Métriques globales

- Densité du graphe : 0.086309
La densité du graphe de GitHub est faible. Les développeurs travaillent en petites équipes sur des dépôts de code spécifiques, ce qui limite les interconnexions globales.
- Degré moyen : 43.0680
Chaque développeur collabore ou interagit en moyenne avec environ 43 autres personnes dans cet échantillon.
- Diamètre : 4.0
La distance qui sépare les deux membres les plus éloignés du réseau est de 4. On a donc un graphe très resserré.
- Longueur moyenne du graphe : 1.9338
La longueur moyenne des chemins nous confirme que les utilisateurs sont très peu éloignés les uns des autres.

#### Métriques individuelles et sommets d'importance

- Le sommet nfultz est le contributeur central du réseau avec 452 collaborations. Il a une connexion avec la quasi-totalité du réseau.
- Le sommet dalinhuang99 le suit avec 276 connexions directes.
- Le sommet Bunlong (degré de 267) agit également comme un grand contributeur.

#### Conclusion

Le réseau GitHub est centralisé autour d'une poignée d'individus. Le sommet nfultz se détache avec 452 collaborations directes. Ce sommet principal, accompagné de dalinhuang99 et Bunlong, agit comme un point de connexion central pour la quasi-totalité du réseau. Cette centralisation explique les distances très courtes mesurées par l'algorithme de Dijkstra (diamètre de 4 et longueur moyenne de 1.93).

## Analyse du réseau social issu de Twitch : [SAE_3-6]

#### Métriques globales

- Densité du graphe : 0.109499
Les viewers d'un même streamer interagissent beaucoup entre eux dans le chat, ce qui explique la densité moyenne du graphe.
- Degré moyen : 54.6400
- Diamètre : 3.0
- Longueur moyenne du graphe : 1.9708

#### Métriques individuelles et sommets d'importance

- Le sommet 2161 est le plus populaire de ce graphe avec 251 connexions directes.
- Le sommet 771 présente le deuxième degré le plus élevé avec 227 connexions.
- Le sommet 5310 ferme le podium avec 216 relations.

#### Conclusion

Les quelques sommets qui monopolisent les degrés de connexion les plus élevés correspondent aux gros streamers de la plateforme. La quasi-totalité du reste du graphe représente les spectateurs. Contrairement à GitHub où la relation est une collaboration mutuelle, la relation ici est basée sur la diffusion de contenu, créant des communautés regroupées autour de ces quelques sommets dominants.
