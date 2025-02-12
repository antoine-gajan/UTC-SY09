from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y


def accuracy(X_train, y_train, X_val, y_val, n_neighbors):
    """Taux de bonne classification d'un modèle Knn pour un jeu de données
    d'apprentissage et de validation fournis.
    """

    # Définition, apprentissage et prédiction par la méthode des
    # plus proches voisins avec `n_neighbors` voisins
    cls = KNeighborsClassifier(n_neighbors=n_neighbors)
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_val)

    # Calcul du taux de bonne classification avec `accuracy_score`
    acc = accuracy_score(y_val, y_pred)

    return acc


def knn_simple_validation(X_train, y_train, X_val, y_val, n_neighbors_list):
    """Pour chaque nombre de voisins dans `n_neighbors_list`, génère
    les triplets nombres de voisins, taux de bonne classification
    correspondant sur l'ensemble de validation avec `accuracy` et
    degré de liberté du Knn (taille de l'ensemble d'apprentissage
    divisé par le nombre de voisins).

    """

    for n in n_neighbors_list:
        acc = accuracy(X_train, y_train, X_val, y_val, n)
        yield n, acc, len(X_train) / n



def knn_multiple_validation(X, y, n_splits, train_size, n_neighbors_list):
    """Pour chaque nombre de voisins dans `n_neighbors_list` et un
    nombre `n_splits` de fois, génère les triplets nombres de voisins,
    taux de bonne classification correspondant sur l'ensemble de
    validation avec `accuracy` et degré de liberté du Knn (taille de
    l'ensemble d'apprentissage divisé par le nombre de voisins).

    """

    # Conversion en tableau numpy si on fournit des DataFrame Pandas par exemple
    X, y = check_X_y(X, y)

    def models_accuracies(train_index, val_index, n_neighbors_list):
        """Taux de bonne classification de tous les modèles pour un jeu de données fixé."""

        # Extraction des jeux de données et étiquettes d'apprentissage
        # et de validation de `X` et `y`
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

        yield from knn_simple_validation(X_train, y_train, X_val, y_val, n_neighbors_list)

    # Définition de `n_splits` jeu de données avec `ShuffleSplit`
    ms = ShuffleSplit(n_splits=n_splits, train_size=train_size)

    for train_index, test_index in ms.split(X):
        yield from models_accuracies(train_index, test_index, n_neighbors_list)
