from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARS
from itertools import chain
from tqdm.auto import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import DataGenerator, KneeLocator
import ray

def get_feature_extension(W_rec, F, labels, n_clusters):
    feature_list = []
    for lbl in range(n_clusters):
        lbl_inds = np.where(labels == lbl)[0]
        if len(lbl_inds) == 0:
            D = np.zeros((W_rec.shape[0], F.shape[1]))
            U = np.zeros((W_rec.shape[0], F.shape[1]))
        else:
            # downstream features
            D = W_rec[lbl_inds, :].T @ F[lbl_inds, :]
            # upstream features
            U = W_rec[:, lbl_inds] @ F[lbl_inds, :]
        feature_list.append(np.hstack([D, U]))
    F_extention = np.hstack(feature_list)
    return F_extention

def get_feature_extension_list(W_recs, feature_list, labels_list, n_clusters):
    F_extention_list = []
    for j in range(len(W_recs)):
        W_rec = W_recs[j]
        F = feature_list[j]
        labels = labels_list[j]
        F_extention = get_feature_extension(W_rec, F, labels, n_clusters)
        F_extention_list.append(F_extention)
    return F_extention_list


class CustomClusterer():
    def __init__(self, n_clusters, max_iter=15, gamma = 0.25, increase_gamma=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.increase_gamma = increase_gamma
        self.gamma = gamma
        self.cluster_centers = None

    def get_aux_features(self, W_recs, feature_list, labels_list, n_clusters):
        return get_feature_extension_list(W_recs, feature_list, labels_list, n_clusters)

    def fit(self, feature_list, W_recs):
        feature_array = np.vstack(feature_list)
        cl = KMeans(n_clusters=self.n_clusters, n_init='auto')
        if self.increase_gamma:
            gammas = np.linspace(0, 1, self.max_iter)
        else:
            gammas = self.gamma * np.ones(self.max_iter)

        cl.fit(feature_array)
        labels_list = [cl.predict(feature_list[i]) for i in range(len(feature_list))]
        scores = []

        for i in range(self.max_iter):
            if i == 0:
                n_features = feature_list[0].shape[1]
                padding = np.zeros((cl.cluster_centers_.shape[0], 2 * self.n_clusters * n_features))
                self.cluster_centers = np.hstack([cl.cluster_centers_, padding])
            else:
                self.cluster_centers = cl_new.cluster_centers_

            aux_feature_list = self.get_aux_features(W_recs, feature_list, labels_list, n_clusters=self.n_clusters)
            # tmp1 =  [f.shape[1] for f in aux_feature_list]
            # print(min(tmp1), max(tmp1))
            features_combined_list = [np.hstack([f, gammas[i] * af]) for f, af in zip(feature_list, aux_feature_list)]
            cl_new = KMeans(n_clusters=self.n_clusters, init=self.cluster_centers, n_init=1)
            features_combined = np.vstack(features_combined_list)
            cl_new.fit(features_combined)
            labels_list_new = [cl_new.predict(features_combined_list[i]) for i in range(len(features_combined_list))]

            # need to calculate missmatch between clusters
            labels_unraveled = list(chain.from_iterable(labels_list))
            labels_unraveled_new = list(chain.from_iterable(labels_list_new))
            score = ARS(labels_unraveled, labels_unraveled_new)
            scores.append(score)
            labels_list = labels_list_new

            if score == 1.0:
                return labels_list

        return labels_list

class ModelSelector():
    def __init__(self, clusterer_bldr_fn, scoring_method,
                 min_clusters, max_clusters,
                 n_repeats, selection_criteria, visualize = True, parallel=True):
        self.clusterer_bld_fn = clusterer_bldr_fn
        self.scoring_method = scoring_method
        if self.scoring_method == 'DaviesBouldin':
            self.cluster_scoring = davies_bouldin_score
        elif self.scoring_method == "Silhouette":
            self.cluster_scoring = silhouette_score
        elif self.scoring_method == 'CalinskiHarabasz':
            self.cluster_scoring = calinski_harabasz_score
        elif self.scoring_method == 'Inertia':
            self.cluster_scoring = inertia_score

        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_repeats = n_repeats
        self.visualize = visualize
        self.selection_criteria = selection_criteria #
        self.parallel = parallel

    def select_n_clusters(self, feature_list, W_recs, visualize=False):
        if self.parallel:
            ray.shutdown()
            ray.init()
        score_matrix = np.zeros((self.max_clusters, self.n_repeats))
        for n_clusters in tqdm(range(self.min_clusters, self.max_clusters)):
            if self.parallel:
                score_matrix[n_clusters, :] = np.array(
                    ray.get([get_score.remote(cl=self.clusterer_bld_fn(n_clusters),
                                              feature_list=feature_list,
                                              W_recs=W_recs,
                                              scoring_function=self.cluster_scoring) for i in range(self.n_repeats)]))
            else:
                score_matrix[n_clusters, :] = np.array([get_score_sequential(cl=self.clusterer_bld_fn(n_clusters),
                                                        feature_list=feature_list,
                                                        W_recs=W_recs,
                                                        scoring_function=self.cluster_scoring)
                                                        for i in range(self.n_repeats)])
        score_matrix = score_matrix[self.min_clusters:, :]
        if self.scoring_method in ["Inertia", "DaviesBouldin"]:
            scores_best = np.nanmin(score_matrix, axis=1)
            if self.selection_criteria == 'extremum':
                optimal_n_clusters = np.nanargmin(scores_best) + self.min_clusters
            elif self.selection_criteria == 'knee':
                kneedle = KneeLocator(np.arange(self.min_clusters, self.max_clusters), scores_best, S=1,
                                      curve="convex",
                                      direction="decreasing")
                knee_point = kneedle.knee

                # choose extremum
                if knee_point is None:
                    knee_point = np.nanargmin(scores_best) + self.min_clusters
                optimal_n_clusters = int(knee_point)
        else:
            scores_best = np.nanmax(score_matrix, axis=1)
            if self.selection_criteria == 'extremum':
                optimal_n_clusters = np.nanargmax(scores_best) + self.min_clusters
            elif self.selection_criteria == 'knee':
                kneedle = KneeLocator(np.arange(self.min_clusters, self.max_clusters), scores_best, S=1,
                                      curve="concave",
                                      direction="increasing")
                knee_point = kneedle.knee
                # choose extremum
                if knee_point is None:
                    knee_point = np.nanargmax(scores_best) + self.min_clusters
                optimal_n_clusters = int(knee_point)

        if visualize:
            fig = plt.figure(figsize=(7, 3))
            plt.plot(np.arange(self.min_clusters, self.max_clusters), scores_best, color='red', label=self.scoring_method)
            plt.axvline(optimal_n_clusters, color='k', linestyle='--')
            plt.grid(True)
            plt.show()
        return optimal_n_clusters


@ray.remote
def get_score(cl, feature_list, W_recs, scoring_function, return_labels=False):
    n_samples = sum([len(f) for f in feature_list])
    if cl.n_clusters > n_samples - 1:
        score = np.nan
        labels_list = None
        print("Too few samples!")
    else:
        labels_list = cl.fit(feature_list=feature_list, W_recs=W_recs)
        labels_unraveled = list(chain.from_iterable(labels_list))
        aux_feature_list = get_feature_extension_list(W_recs, feature_list, labels_list, n_clusters=cl.n_clusters)
        features_combined_list = [np.hstack([f, cl.gamma * af]) for f, af in zip(feature_list, aux_feature_list)]
        feature_array_combined = np.vstack(features_combined_list)
        score = scoring_function(feature_array_combined, labels_unraveled)
    if return_labels:
        return score, labels_list
    else:
        return score

def get_score_sequential(cl, feature_list, W_recs, scoring_function, return_labels=False):
    n_samples = sum([len(f) for f in feature_list])
    if n_samples <= cl.n_clusters:
        score = np.nan
        labels_list = None
        print("Too few samples!")
    else:
        labels_list = cl.fit(feature_list=feature_list, W_recs=W_recs)
        labels_unraveled = list(chain.from_iterable(labels_list))
        aux_feature_list = get_feature_extension_list(W_recs, feature_list, labels_list, n_clusters=cl.n_clusters)
        features_combined_list = [np.hstack([f, cl.gamma * af]) for f, af in zip(feature_list, aux_feature_list)]
        feature_array_combined = np.vstack(features_combined_list)
        score = scoring_function(feature_array_combined, labels_unraveled)
    if return_labels:
        return score, labels_list
    else:
        return score

def get_best_clustering(clusterer, feature_list, W_recs, scoring_method, n_repeats, parallel=True):
    if scoring_method == 'DaviesBouldin':
        cluster_scoring = davies_bouldin_score
    elif scoring_method == "Silhouette":
        cluster_scoring = silhouette_score
    elif scoring_method == 'CalinskiHarabasz':
        cluster_scoring = calinski_harabasz_score
    elif scoring_method == 'Inertia':
        cluster_scoring = inertia_score
    if parallel:
        results = ray.get([get_score.remote(cl=clusterer,
                                            feature_list=feature_list,
                                            W_recs=W_recs,
                                            scoring_function=cluster_scoring,
                                            return_labels=True) for i in range(n_repeats)])
    else:
        results = [get_score_sequential(cl=clusterer,
                                        feature_list=feature_list,
                                        W_recs=W_recs,
                                        scoring_function=cluster_scoring,
                                        return_labels=True) for i in range(n_repeats)]

    scores = [tpl[0] for tpl in results]
    labellings = [tpl[1] for tpl in results]
    if scoring_method in ["Inertia", "DaviesBouldin"]:
        ind = np.argmin(scores)
        best_score = scores[ind]
        best_labelling = labellings[ind]
    else:
        ind = np.argmax(scores)
        best_score = scores[ind]
        best_labelling = labellings[ind]
    return best_score, best_labelling


def inertia_score(features, labels):
    lbls_unique = np.unique(labels)
    res = 0
    for lbl in lbls_unique:
        mask = labels == lbl
        f = features[mask]
        center = np.mean(features, axis=0)
        diff = f - center
        res += np.sum(np.square(diff))
    return res



