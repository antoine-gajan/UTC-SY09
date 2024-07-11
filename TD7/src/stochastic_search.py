import math


class StochasticProtList:
    """Tirage aléatoire des hyperparamètres `n1` et `n2` en fonction de
    `n_neighbors` et `A`.

    """

    def __init__(self, n_neighbors, A):
        self.n_neighbors = n_neighbors
        self.A = A

    def rvs(self, *args, **kwargs):
        # Création de `n1` et `n2` vérifiant les 2e et 3e conditions
        n1 = ...
        n2 = ...

        # Retour du couple de prototypes si la 4e condition est
        # vérifiée ou rejet de ce couple et appel récursif de `rvs`
        if ...:
            return ...
        else:
            return ...


A = 100

param_grid = [
    {
        "n_neighbors": [n_neighbors],
        "n_prototypes_list": StochasticProtList(n_neighbors, A),
    }
    for n_neighbors in range(1, math.floor(math.sqrt(A)) + 1)
]
