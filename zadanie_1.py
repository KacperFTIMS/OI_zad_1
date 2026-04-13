import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score
from sklearn.datasets import load_iris, load_wine
from scipy.spatial import Voronoi, voronoi_plot_2d


def super_funkacja(X, y, name, ax_s, ax_b, ax_w, pg):
    k_values = range(2, 10)
    scores = []
    all_ari = []
    all_homo = []
    all_com = []
    all_predictions = []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    for k in k_values:
        kmeans = cluster.KMeans(n_clusters=k)

        y_kmeans = kmeans.fit_predict(X_scaled)

        all_predictions.append(y_kmeans)

        score = silhouette_score(X_scaled, y_kmeans)

        ari_km = adjusted_rand_score(y, y_kmeans)
        hom_km = homogeneity_score(y, y_kmeans)
        com_km = completeness_score(y, y_kmeans)

        scores.append(score)
        all_ari.append(ari_km)
        all_homo.append(hom_km)
        all_com.append(com_km)

        print(f"Wyniki dla n {k}, zbioru {nazwa} - K-Means: ARI={ari_km:.3f}, HOM={hom_km:.3f}, COM={com_km:.3f}, ")


    if pg == 1:
        idx_best = np.argmax(scores)
        idx_worst = np.argmin(scores)

        ax_s.plot(k_values, scores, marker='o', color='r')
        ax_s.set_title(f"Zbiór {name}")
        ax_s.set_xlabel("Wartości")
        ax_s.set_ylabel("silhouette score")
        ax_s.grid(True)

        voronoi_buff(X_scaled, y, name, all_predictions, ax_b, ax_w,
                     ax_s, k_values, scores, idx_best, idx_worst)
    if pg == 3:
        idx_best = np.argmax(all_ari)
        idx_worst = np.argmin(all_ari)

        ax_s.plot(k_values, all_ari, marker='o', label='ARI', color='tab:blue')
        ax_s.plot(k_values, all_homo, marker='x', label='HOM', color='tab:orange')
        ax_s.plot(k_values, all_com, marker='^', label='COM', color='tab:green')
        ax_s.set_title(f"Zbiór {name} - Metryki zewn.")
        ax_s.set_xlabel("k")
        ax_s.legend()
        ax_s.grid(True)

        voronoi_buff(X_scaled, y, name, all_predictions, ax_b, ax_w,
                     ax_s, k_values, all_ari, idx_best, idx_worst)

    if pg == 5:
        idx_best = np.argmax(all_ari)
        print(
            f"{name.upper()} | K-Means (k={k_values[idx_best]}): Silhouette={scores[idx_best]:.3f}, ARI={all_ari[idx_best]:.3f}, HOM={all_homo[idx_best]:.3f}, COM={all_com[idx_best]:.3f}")

        # Rysowanie wykresów liniowych (Silhouette + Metryki zewn.) dla pg=5 na przekazanym obiekcie ax_s
        if ax_s is not None:
            # Silhouette na innej osi (bliźniak) lub na tym samym z dopiskiem
            ax_s.plot(k_values, all_ari, marker='o', label='ARI', color='tab:blue')
            ax_s.plot(k_values, all_homo, marker='x', label='HOM', color='tab:orange')
            ax_s.plot(k_values, all_com, marker='^', label='COM', color='tab:green')
            # Dodajemy Silhouette (przerywaną czerwoną linią) do tego samego wykresu
            ax_s.plot(k_values, scores, marker='s', linestyle='--', label='Silhouette', color='tab:red')

            ax_s.set_title(f"{name.upper()} - K-Means Metryki")
            ax_s.set_xlabel("k (liczba klastrów)")
            ax_s.legend()
            ax_s.grid(True)

        # Rysowanie punktów tylko dla zbioru Iris
        if name == "iris":
            fig_rz, ax_rz = plt.subplots(1, 2, figsize=(10, 4))
            ax_rz[0].scatter(X[:, 2], X[:, 3], c=y, cmap='viridis', edgecolor='k')
            ax_rz[0].set_xlabel("Długość płatka (Petal Length)")
            ax_rz[0].set_ylabel("Szerokość płatka (Petal Width)")
            ax_rz[0].set_title("Zbiór Iris: Wyraźna separacja")

            ax_rz[1].scatter(X[:, 0], X[:, 2], c=y, cmap='viridis', edgecolor='k')
            ax_rz[1].set_xlabel("Długość działki (Sepal Length)")
            ax_rz[1].set_ylabel("Długość płatka (Petal Length)")
            ax_rz[1].set_title("Zbiór Iris: Nakładanie się klas")

            fig_rz.tight_layout()
            # ZAPISUJEMY UŻYWAJĄC OBIEKTU fig_rz:
            fig_rz.savefig("iris_projections.png")
            # ZAMYKAMY TEN WYKRES, ABY NIE BLOKOWAŁ METRYK:
            plt.close(fig_rz)


def dbscan(X, y, name, ax_s, ax_b, ax_w, pg):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    eps_values = np.arange(0.1, 1.6, 0.1)

    scores = []
    all_ari = []
    all_homo = []
    all_com = []
    n_clusters_list = []
    all_predictions = []

    for eps in eps_values:
        db = cluster.DBSCAN(eps=eps, min_samples=1)
        y_db = db.fit_predict(X_scaled)

        all_predictions.append(y_db)

        n_clusters = len(set(y_db))
        n_clusters_list.append(n_clusters)

        if 1 < n_clusters < len(X_scaled):
            score = silhouette_score(X_scaled, y_db)
        else:
            score = -1

        ari_km = adjusted_rand_score(y, y_db)
        hom_km = homogeneity_score(y, y_db)
        com_km = completeness_score(y, y_db)

        scores.append(score)
        all_ari.append(ari_km)
        all_homo.append(hom_km)
        all_com.append(com_km)

        print(f"Wyniki dla eps {eps}, zbioru {nazwa} - DBSCAN: ARI={ari_km:.3f}, HOM={hom_km:.3f}, COM={com_km:.3f}, ")

    if pg == 2:
        idx_best = np.argmax(scores)
        idx_worst = np.argmin(scores)

        ax_s.plot(eps_values, scores, marker='o', color='r')
        ax_s.set_title(f"Zbiór {name} - Silhouette")
        ax_s.set_xlabel("eps")
        ax_s.grid(True)

        voronoi_buff(X_scaled, y, name, all_predictions, ax_b, ax_w,
                     ax_s, eps_values, scores, idx_best, idx_worst)

    if pg == 4:
        idx_best = np.argmax(all_ari)
        idx_worst = np.argmin(all_ari)

        ax_s.plot(eps_values, all_ari, marker='o', label='ARI', color='tab:blue')
        ax_s.plot(eps_values, all_homo, marker='x', label='HOM', color='tab:orange')
        ax_s.plot(eps_values, all_com, marker='^', label='COM', color='tab:green')
        ax_s.set_title(f"Zbiór {name} - Metryki zewn.")
        ax_s.set_xlabel("eps")
        ax_s.legend()
        ax_s.grid(True)

        voronoi_buff(X_scaled, y, name, all_predictions, ax_b, ax_w,
                     ax_s, eps_values, all_ari, idx_best, idx_worst)

    if pg == 5:
        idx_best = np.argmax(all_ari)
        if all_ari[idx_best] > 0.0:
            print(
                f"{name.upper()} | DBSCAN (eps={eps_values[idx_best]:.1f}): Silhouette={scores[idx_best]:.3f}, ARI={all_ari[idx_best]:.3f}, HOM={all_homo[idx_best]:.3f}, COM={all_com[idx_best]:.3f}")
        else:
            print(f"{name.upper()} | DBSCAN nie znalazł klastrów.")

        ax_s.plot(eps_values, all_ari, marker='o', label='ARI', color='tab:blue')
        ax_s.plot(eps_values, all_homo, marker='x', label='HOM', color='tab:orange')
        ax_s.plot(eps_values, all_com, marker='^', label='COM', color='tab:green')
        ax_s.plot(eps_values, scores, marker='s', linestyle='--', label='Silhouette', color='tab:red')

        ax_s.set_title(f"{name.upper()} - DBSCAN Metryki")
        ax_s.set_xlabel("eps")
        ax_s.legend()
        ax_s.grid(True)

def voronoi_buff(X_scaled, y, name, all_predictions, ax_b, ax_w, ax_s, values, scores, idx_best, idx_worst):
    # Woronoj najlepszy
    plot_voronoi_diagram(X_scaled, y, all_predictions[idx_best], ax_b)

    ax_b.set_title(f"Best: {values[idx_best]:.2f}, score: {scores[idx_best]:.3f}")

    # Woronoj najgorszy
    plot_voronoi_diagram(X_scaled, y, all_predictions[idx_worst], ax_w)
    ax_w.set_title(f"Worst: {values[idx_worst]:.2f}, score: {scores[idx_worst]:.3f}")



def plot_voronoi_diagram(X, y_true, y_pred, ax):
    X = X[:, :2]

    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(X, y_pred)

    step = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    custom_cmap = 'tab10'
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=custom_cmap)

    vor = Voronoi(X)
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False,
                    line_colors='k', line_width=1.0, line_alpha=0.4)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if y_true is not None:
        ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap=custom_cmap, edgecolor='k', s=35)
    else:
        ax.scatter(X[:, 0], X[:, 1], c='k', s=15)

    # ax.title("Diagram Woronoja")
    # ax.tight_layout()


lista_zbiorow = ["2_1", "2_2", "2_3", "3_1", "3_2", "3_3"]
lista_real = ["iris", "wine"]

for pg in [5]:
    if pg in [1, 2, 3, 4]:

        fig, axes = plt.subplots(6, 3, figsize=(18, 30))

        for i, nazwa in enumerate(lista_zbiorow):
            zbior = pd.read_csv(f"{nazwa}.csv", sep=";", header=None)

            X = zbior.iloc[:, :-1].values
            y = zbior.iloc[:, -1].values

            ax_silhouette = axes[i, 0]
            ax_best = axes[i, 1]
            ax_worst = axes[i, 2]

            if pg % 2 == 1:
                super_funkacja(X, y, nazwa, ax_silhouette, ax_best, ax_worst, pg)
            elif pg % 2 == 0:
                dbscan(X, y, nazwa, ax_silhouette, ax_best, ax_worst, pg)

            # dbscan(X, y, nazwa)

        plt.tight_layout()
        plt.savefig(f"exp_{pg}")
        plt.show()

    elif pg == 5:

        for nazwa in lista_real:
            if nazwa == "iris":
                X, y = load_iris(return_X_y=True)
            else:
                X, y = load_wine(return_X_y=True)

            fig_metryki, ax_metryki = plt.subplots(1, 2, figsize=(12, 5))
            # Ponieważ dla pg=5 nie rysujemy siatki z Woronojem,
            # przekazujemy do funkcji wartości None zamiast obiektów osi ax
            super_funkacja(X, y, nazwa, ax_metryki[0], None, None, pg)
            dbscan(X, y, nazwa, ax_metryki[1], None, None, pg)

            plt.tight_layout()
            plt.savefig(f"analiza_{nazwa}_metryki.png")
            plt.show()

# irisx, irisy = load_iris(return_X_y=True)

# print(irisx, irisy)
