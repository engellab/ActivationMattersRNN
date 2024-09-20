import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from kneed import DataGenerator, KneeLocator

def intra_cluster_variance():
    pass

#TODO: elbow method, gap statistics

n_centers = 11
X, y = make_blobs(n_samples=3000, n_features=95, centers=n_centers)
print(X.shape)
# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(4,20), metric='silhouette')
# visualizer.fit(X)        # Fit the data to the visualizer
# visualizer.show()        # Finalize and render the figure


# now, do the clustering with KMeans
n_repeats = 5
max_clusters = 20
min_clusters = 3

score_matrix = np.zeros((max_clusters, n_repeats))
for n in range(min_clusters, max_clusters):
    cl = KMeans(n_clusters=n, n_init='auto')
    for i in range(n_repeats):
        labels = cl.fit_predict(X)
        score = silhouette_score(X, labels)
        # score = davies_bouldin_score(X, labels)
        # score = calinski_harabasz_score(X, labels)
        # score = cl.inertia_
        score_matrix[n, i] = score


score_matrix = score_matrix[min_clusters:, :]

best_scores = np.min(score_matrix, axis=1)
kneedle = KneeLocator(np.arange(min_clusters, max_clusters), best_scores, S=1.0, curve="concave", direction="increasing")
print(kneedle.knee)

fig = plt.figure(figsize = (5, 5))
plt.imshow(score_matrix, cmap='bwr', interpolation='bicubic')
plt.show()


fig = plt.figure(figsize = (7, 3))
plt.plot(np.arange(min_clusters, max_clusters), best_scores, color = 'red', label = 'silhouette_score')
plt.axvline(n_centers, color = 'k', linestyle = '--')
plt.grid(True)
plt.show()

fig = plt.figure(figsize = (7, 3))
plt.plot(np.arange(min_clusters, max_clusters - 1), np.diff(best_scores), color = 'blue', label = 'silhouette_score')
plt.axvline(n_centers, color = 'k', linestyle = '--')
plt.grid(True)
plt.show()


fig = plt.figure(figsize = (7, 3))
plt.plot(np.arange(min_clusters, max_clusters - 2), np.diff(np.diff(best_scores)), color = 'green', label = 'silhouette_score')
plt.axvline(n_centers, color = 'k', linestyle = '--')
plt.grid(True)
plt.show()

# pca = PCA(n_components=3)
# pca.fit(X)
# points_pr = pca.components_.T
# print(points_pr.shape)
# fig = plt.figure(figsize=(10, 10))
# plt.scatter(points_pr[:, 0], points_pr[:, 1], color = 'orange', edgecolors='k')
# plt.show()