import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from numpy.random import default_rng
rng = default_rng(42)


def sample_from_M(delta, c, n=1000):
    # Génération d'un échantillon de taille `n` suivant le mélange entre une loi
    # normale centrée réduite et une loi exponentielle décentrée.
    ...


def sample_from_Xy(delta, c, n=1000):
    # Génération d'un échantillon de taille `n` de loi celle de `X` avec les
    # étiquettes correspondantes `y`.
    ...


def gen(deltas, cs):
    for delta, c in zip(deltas, cs):
        # Estimation de l'erreur du 1-NN pour un jeu de
        # données avec `delta` et `c`.

        # On entraine un 1-NN sur un jeu de donnée de taille 1000.
        ...

        # On estime son erreur avec un jeu de données de test de
        # taille 10000.
        ...

        # On génère `delta`, `c` et l'erreur du 1-NN
        yield delta, c, err
