import numpy as np
np.set_printoptions(suppress=True)
import os
import warnings
warnings.filterwarnings('ignore')
os.system('python ../../style/style_setup.py')
from src.utils.graph_clustering import CustomClusterer, ModelSelector, get_best_clustering
from src.utils.get_intercluster_connectivity import get_intercluster_connectivity
from src.plots.ploting_utils import *
from src.utils.feautre_extraction_utils import reduce_dimensionality


# taskname = "DMTS"
taskname = "DMTS"
# taskname = "HalfAdder"
# taskname = "XOR"

activation = 'relu'
cluster_scoring_method = ["Silhouette", "DaviesBouldin", "CalinskiHarabasz", "Inertia"][0]
n_repeats = 30
min_clusters = 4
max_clusters = 18
point_selection = 'knee'
gamma = 0.3
increase_gamma = False
max_iter = 12
thr = 0.2
typeIF = 2
connIF = 1

best_labelling = None
data_path = f"../../data/features_datasets/{taskname}_{activation}_typeif={typeIF}_connif={connIF}.pkl"
Feature_data = pickle.load(open(data_path, 'rb+'))
Feature_array = Feature_data["Features"]
feature_list = Feature_data["Features_by_RNN"]
W_inps = Feature_data["W_inps"]
W_recs = Feature_data["W_recs"]
W_outs = Feature_data["W_outs"]

clusterer_params = {"max_iter": max_iter, "gamma": gamma, "increase_gamma": increase_gamma}
clusterer_bldr_fn = lambda n_clusters: CustomClusterer(n_clusters, **clusterer_params)
model_selector = ModelSelector(clusterer_bldr_fn,
                               scoring_method=cluster_scoring_method,
                               min_clusters=min_clusters,
                               max_clusters=max_clusters,
                               n_repeats=n_repeats,
                               point_selection=point_selection)
optimal_n_clusters = model_selector.select_n_clusters(feature_list=feature_list, W_recs=W_recs, visualize=True)
print(f"Optimal number of clusters: {optimal_n_clusters}")

clusterer = CustomClusterer(n_clusters=optimal_n_clusters, **clusterer_params)
best_score, labels_list = get_best_clustering(clusterer=clusterer,
                                                feature_list=feature_list,
                                                W_recs=W_recs,
                                                scoring_method=cluster_scoring_method,
                                                n_repeats=n_repeats)
labels_unraveled = list(chain.from_iterable(labels_list))

ic_W_inp, ic_W_rec, ic_W_out = get_intercluster_connectivity(W_inps, W_recs, W_outs, labels_list)
print(f"Best clustering Score: {best_score}")
Features_projected, features_list_projected = reduce_dimensionality(Feature_array=Feature_data["Features"],
                                                                     Feature_list=Feature_data["Features_by_RNN"],
                                                                     var_thr=0.99)
plot_RNN_distribution(labels_list)
plot_feature_array(Features_projected, labels_unraveled)
plot_intercluster_connectivity(ic_W_inp=ic_W_inp,
                               ic_W_rec=ic_W_rec,
                               ic_W_out=ic_W_out,
                               labels_unraveled=labels_unraveled,
                               th=thr)

# pickle.dump(connectivity_dict, open(f"../data/effective_connectivity_{activation_type}_trajectoriesonly.pkl", "wb+"))
# # Need to assemble the trajectories
# Trajectories = np.vstack(Feature_data["Trajectories_by_RNN"])
# Mean_trajectories = np.zeros((n_clusters, Trajectories.shape[1], Trajectories.shape[2]))
# for lbl in labels_permuted:
#     Mean_trajectories[lbl, ...] = np.mean(Trajectories[np.where(labels_unraveled_permuted == lbl)[0], ...], axis = 0)
# data = {}
# data["traces"] = Mean_trajectories
# data["W_rec"] = ic_W_rec
# data["W_inp"] = ic_W_inp
# data["W_out"] = ic_W_out
# pickle.dump(data, open(f"../../data/effective_trajectories/traj_DMTS_{tag}.pkl", "wb+"))

