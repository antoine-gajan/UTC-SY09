from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from src.knn_validation import accuracy
from sklearn.utils import check_X_y


def knn_cross_validation(X, y, n_folds, n_neighbors_list):
    """Génère les couples nombre de voisins et taux de bonne classification correspondant."""

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

        for n in n_neighbors_list:
            acc = accuracy(X_train, y_train, X_val, y_val, n)
            yield n, acc, len(X_train) / n

    ms = KFold(n_splits=n_folds)
    for train_index, val_index in ms.split(X):
        yield from models_accuracies(train_index, val_index, n_neighbors_list)


def knn_cross_validation2(X, y, n_folds, n_neighbors_list):
    # Générer la même sortie de `knn_cross_validation` en utilisant
    # `cross_val_score`

    # Conversion en tableau numpy si on fournit des DataFrame Pandas par exemple
    X, y = check_X_y(X, y)

    for n in n_neighbors_list:
        cls = KNeighborsClassifier(n_neighbors=n)
        accs = cross_val_score(cls, X, y, cv=n_folds)
        for acc in accs:
            yield n, acc, len(X) / n
