# Premier_modele_K-NN

Projet réalisé dans le cadre de la formation Microsoft IA x Simplon - Brest +.

_ _ _

## Création du DataSet

Le dossier DataSet regroupe les différents jeux de données collectés (fichier _*.csv*_), chacun des fichiers contient normalement 10 observations.

Le fichier `create_DataSet.py`, contient la fonction `create_DataSet()` permetant de concaténer les différentes observations et retourne un DataFrame.

_ _ _

## KNN from Scratch

Le calcul du KNN from scratch se faire à l'aide de la class KNN contenu dans le fichier *KNN.py*.

### Utilisation

/!\ le modèle ne fonctionne qu'avec des *array* composé d'*interger* !

1. Importer la class **KNN** : `from KNN import *`
2. Instancier la class : `modele = KNN()`
3. Entrainer le modèle : `modele.train(X_train, y_train)`
    Il est possible de spécfier la metric à utiliser (`metric=`) avec les valeur `'euclidean' or 'manhattan' or 'minkowski'`
4. Réaliser une prédiction : `y_pred = modele.prediction(X_test, k)`
5. Calculer l'accuracy du modèle : `acc = modele.accuracy(y_test, y_pred)[0]`
    Il est nécéssaire de spécifier l'indice *0*, sinon la fonction renvois également le taux d'erreur.
    En ajoutant le paramètre `resume=True`, les valeurs d'accuracy ainsi que d'autres estimation de performance s'afficherons dans la console.

### Composition