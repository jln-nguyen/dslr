# DSLR : Data Science x Logistic Regression

Recréation du Choixpeau magique de Poudlard : à partir des notes d'un élève dans plusieurs cours, le programme prédit sa maison (Gryffindor, Hufflepuff, Ravenclaw ou Slytherin).

Le projet se déroule en deux temps :

1. **Explorer les données** (`describe`, `histogram`, `scatter_plot`, `pair_plot`) pour comprendre quels cours sont utiles.
2. **Entraîner un classifieur** (`logreg_train`, `logreg_predict`) : une régression logistique one-vs-all, entraînée par descente de gradient.

## Structure

```
.
├── dataset/
│   ├── dataset_train.csv      # élèves avec leur maison (1600 lignes)
│   └── dataset_test.csv       # élèves sans maison (à prédire)
├── describe.py                # statistiques descriptives, recodées à la main
├── histogram.py               # quel cours est homogène entre les maisons ?
├── scatter_plot.py            # quelles deux features sont similaires ?
├── pair_plot.py               # quelles features garder pour le modèle ?
├── logreg_train.py            # entraînement, génère weights.json
├── logreg_predict.py          # prédiction, génère houses.csv
├── weights.json               # (généré) modèle entraîné
└── houses.csv                 # (généré) prédictions
```

## Prérequis

- Python 3
- `numpy`, `pandas`, `matplotlib`

```
pip install numpy pandas matplotlib
```

Aucune bibliothèque d'apprentissage automatique n'est utilisée. Les statistiques (moyenne, écart-type, quartiles, corrélation) et la descente de gradient sont codées à la main. `numpy` ne sert qu'aux produits matriciels.

## Utilisation

```
python3 describe.py dataset/dataset_train.csv
python3 histogram.py dataset/dataset_train.csv
python3 scatter_plot.py dataset/dataset_train.csv
python3 pair_plot.py dataset/dataset_train.csv

python3 logreg_train.py dataset/dataset_train.csv
python3 logreg_predict.py dataset/dataset_test.csv weights.json
```

Option de validation (entraîne sur 80 % du fichier, mesure sur les 20 % restants, n'écrit pas `weights.json`) :

```
python3 logreg_train.py dataset/dataset_train.csv --validate
```

Tous les programmes quittent avec le code 1 et un message `Error: ...` en cas de mauvais arguments, de mauvaise extension ou de fichier illisible.

## Partie 1 : exploration des données

### describe

Recode `DataFrame.describe()` sans l'utiliser. Pour chaque cours numérique, il affiche : **Count, Mean, Std, Min, 25 %, 50 %, 75 %, Max**.

- Les valeurs manquantes sont retirées **colonne par colonne** avant chaque calcul : le count diffère donc d'un cours à l'autre.
- **Écart-type** : diviseur `n - 1` (correction de Bessel), comme pandas.
- **Quartiles** : interpolation linéaire. Sur la liste triée, position `(n - 1) x q`, puis mélange des deux valeurs voisines selon la partie décimale.
- Affichage aligné avec 6 décimales (`f'{value:>15.6f}'`). Les noms de cours trop longs sont tronqués avec `...`.

### histogram

Question : **quel cours a une distribution de notes homogène entre les quatre maisons ?**

Un histogramme par cours, avec les 4 maisons superposées en transparence. Le programme calcule aussi, pour chaque cours, un score d'homogénéité :

```
score = écart-type des 4 moyennes de maison / écart-type global du cours
```

Plus le score est bas, plus les maisons ont les mêmes notes. Le cours au score le plus bas est affiché dans le terminal.

**Réponse : Care of Magical Creatures** (Arithmancy est très proche).

### scatter_plot

Question : **quelles sont les deux features similaires ?**

Le programme calcule la corrélation de Pearson de chacune des 78 paires de cours (13 x 12 / 2), garde celle dont la valeur absolue est la plus grande, et trace son nuage de points.

```
covariance(X, Y) = somme((x - moy_x) x (y - moy_y)) / (n - 1)
r = covariance(X, Y) / (std(X) x std(Y))
```

La comparaison se fait sur `abs(r)` : une corrélation de -0,99 est aussi forte que +0,99.

**Réponse : Astronomy et Defense Against the Dark Arts**, avec r proche de -1 (les points forment une droite).

### pair_plot

Question : **quelles features utiliser pour la régression logistique ?**

Grille 13 x 13 :

- **hors diagonale** : nuage de points d'un cours contre un autre ;
- **diagonale** : histogramme du cours ;
- **couleur** : la maison de l'élève.

On garde les cours où les couleurs se séparent, on écarte ceux où elles se mélangent, et on ne garde qu'un cours dans chaque paire redondante.

### Conclusion : choix des features

| Cours | Décision | Raison |
|---|---|---|
| Arithmancy | écarté | les 4 maisons ont la même distribution |
| Care of Magical Creatures | écarté | idem |
| Defense Against the Dark Arts | écarté | redondant avec Astronomy (corrélation proche de -1) |
| Potions | écarté | sépare très peu les maisons |
| Herbology, Astronomy, Divination, Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Charms, Flying | **gardés (9)** | chacun isole une ou deux maisons |

## Partie 2 : régression logistique

### Principe

Une régression logistique répond à une question oui/non. Pour 4 maisons, on entraîne **4 modèles indépendants** (one-vs-all) : « est-ce un Gryffindor ? », « un Hufflepuff ? », etc. Chaque modèle est une liste de **10 nombres** : un poids par cours et un biais.

Pour un élève :

```
score       z = biais + w1 x note1 + ... + w9 x note9
probabilité h = 1 / (1 + e^(-z))          (sigmoïde)
```

La maison prédite est celle dont la probabilité est la plus haute.

### Entraînement (`logreg_train.py`)

1. Lecture du CSV, suppression des élèves sans maison.
2. Calcul de la moyenne et de l'écart-type de chaque cours, sur les vraies notes.
3. Remplacement des notes manquantes par la moyenne du cours, puis **standardisation** `(note - moyenne) / écart-type`.
4. Ajout d'une colonne de 1 en première position (le biais) : `X` de forme `(1600, 10)`.
5. Pour chaque maison, `y = 1` si l'élève en fait partie, `0` sinon, puis **descente de gradient** à partir de poids nuls :

```
h        = sigmoïde(X @ poids)
gradient = X.T @ (h - y) / m
poids    = poids - learning_rate x gradient
```

6. Sauvegarde dans `weights.json`.

| Paramètre | Valeur | Raison |
|---|---|---|
| Learning rate | 0,1 | convient aux données standardisées |
| Itérations | 1000 | le coût est quasi plat ensuite (3000 itérations ne changent pas la précision) |

Le coût (log-loss) part de 0,693 (modèle au hasard) et descend à environ 0,05 à 0,07. Il sert uniquement à vérifier la convergence.

### Fichier de poids (`weights.json`)

```json
{
    "features": ["Herbology", "Astronomy", "..."],
    "means":    {"Herbology": 0.0, "...": 0.0},
    "stds":     {"Herbology": 0.0, "...": 0.0},
    "weights":  {"Gryffindor": [10 valeurs], "Hufflepuff": [...], "Ravenclaw": [...], "Slytherin": [...]}
}
```

Le fichier contient tout ce dont la prédiction a besoin : liste et ordre des cours, statistiques du jeu d'entraînement, poids de chaque maison (le premier est le biais).

### Prédiction (`logreg_predict.py`)

1. Lecture de `weights.json` et du CSV de test.
2. Sélection des **mêmes cours, dans le même ordre**.
3. Remplacement des notes manquantes par les **moyennes du jeu d'entraînement**, puis standardisation avec les **moyennes et écarts-types du jeu d'entraînement** (jamais ceux du test).
4. Ajout de la colonne de 1 en première position.
5. Calcul des 4 probabilités par élève, choix de la plus grande.
6. Écriture de `houses.csv` :

```
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
...
```

Aucun élève n'est supprimé : une prédiction est produite pour chaque ligne du test.

## Résultats

| Mesure | Résultat |
|---|---|
| Précision sur le jeu d'entraînement complet (1600 élèves) | 98,19 % |
| `logreg_predict` sur `dataset_train.csv`, comparé aux vraies maisons | 98,19 % (même valeur : la chaîne de prédiction est correcte) |
| Entraînement sur 80 % (1280), mesure sur 20 % (320) | 98,44 % / 97,19 % |

La mesure sur 320 élèves est bruitée : chaque erreur pèse environ 0,31 point. Le plafond vers 98 % vient des élèves dont le profil est vraiment ambigu avec ces 9 cours, pas de la durée d'entraînement.

## Choix de conception

- **Colonne de 1 pour le biais** plutôt qu'un biais séparé : une seule formule de gradient pour tous les paramètres.
- **Statistiques du train sauvegardées** dans le fichier de poids, pour que la prédiction applique exactement la même transformation.
- **Liste des cours lue depuis le fichier de poids** et non réécrite dans `logreg_predict`, pour garantir le même ordre de colonnes.
- **Moyenne plutôt que suppression** pour les valeurs manquantes : on garde tous les élèves, ce qui est obligatoire pour le test.
- **Épsilon de 1e-15** dans les logarithmes du coût, pour éviter `log(0)`.

## Limites

- Un modèle linéaire ne sépare pas les élèves dont les notes ressemblent à celles d'une autre maison.
- Le remplacement par la moyenne rapproche un élève sans note du profil moyen.
- Le choix des cours repose sur une lecture graphique, complétée par la mesure de la précision.