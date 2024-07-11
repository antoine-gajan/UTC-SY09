import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

def kmeans_dataset(dataset, n_clusters_list, strategies, tries):
    for n_clusters in n_clusters_list:
        for strategy in strategies:
            for rs in range(tries):  # On utilisera `rs` pour fixer le `random_state`
                km = KMeans(n_clusters=n_clusters, init=strategy, random_state=rs, n_init=1).fit(dataset)
                inertia = km.inertia_
                yield rs, strategy, n_clusters, inertia

