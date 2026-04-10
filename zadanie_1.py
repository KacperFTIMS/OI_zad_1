import pandas as pd
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score
from sklearn.datasets import load_iris, load_wine


def super_funkacja(X, y):
    k_values = range(2, 10)
    scores = []
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    for k in k_values:
        kmeans = cluster.KMeans(n_clusters=k)

        y_kmeans = kmeans.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, y_kmeans)

        scores.append(score)

        ari_km = adjusted_rand_score(y, y_kmeans)
        hom_km = homogeneity_score(y, y_kmeans)
        com_km = completeness_score(y, y_kmeans)

        print(f"Wyniki dla n {k} - K-Means: ARI={ari_km:.3f}, HOM={hom_km:.3f}, COM={com_km:.3f}, ")

    plt.figure(figsize=(8, 4))
    plt.plot(k_values, scores, marker='o', color='r')
    plt.xlabel("Liczba klastrow")
    plt.ylabel("silhouette score")
    plt.grid(True)

    plt.show()

def dbscan(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    eps_values = numpy.arange(0.1, 1.6, 0.1)

    scores = []
    n_clusters_list = []

    for eps in eps_values:
        db = cluster.DBSCAN(eps=eps, min_samples=1)
        y_db = db.fit_predict(X_scaled)

        n_clusters = len(set(y_db))
        n_clusters_list.append(n_clusters)

        if 1 < n_clusters < len(X_scaled):
            score = silhouette_score(X_scaled, y_db)
        else:
            score = -1

        scores.append(score)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.set_xlabel('Parametr eps')
    ax1.set_ylabel('Silhouette Score', color='tab:red')
    ax1.plot(eps_values, scores, marker='o', color='tab:red', label='Silhouette')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Liczba klastrów', color='tab:blue')
    ax2.plot(eps_values, n_clusters_list, marker='x', linestyle='--', color='tab:blue', label='Liczba klastrów')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.show()

def woronoi(X, y):
    pass

lista_zbiorow = ["2_1", "2_2", "2_3", "3_1", "3_2", "3_3"]

for nazwa in lista_zbiorow:
    zbior = pd.read_csv(f"{nazwa}.csv", sep=";", header=None)

    X = zbior.iloc[:, :-1].values
    y = zbior.iloc[:, -1].values

    #super_funkacja(X, y)
    dbscan(X, y)

irisx, irisy = load_iris(return_X_y=True)

# print(irisx, irisy)
