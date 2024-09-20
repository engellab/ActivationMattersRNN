# import numpy as np
# import umap
# import sys
# sys.path.insert(0, '../../')
# sys.path.insert(0, '../../../')
# sys.path.insert(0, 'Users/tolmach/Documents/GitHub')
# sys.path.insert(0, 'Users/tolmach/Documents/GitHub/rnn_coach/src')
# from utils import gini, mean_downsample, get_ordering, permute_input_matrix, permute_output_matrix, \
#     permute_recurrent_matrix
#
# np.set_printoptions(suppress=True)
# from matplotlib import pyplot as plt
# import os
# import pickle
# import json
# from copy import deepcopy
# import pandas as pd
# from copy import deepcopy
# from sklearn.decomposition import PCA
# from sklearn.metrics import silhouette_score
# from sklearn.cluster import DBSCAN, KMeans
# from scipy.interpolate import interp1d
# from tqdm.auto import tqdm
# import warnings
# warnings.filterwarnings('ignore')
# from style import style_setup
# from scipy.stats import zscore
# os.system('python ../style/style_setup.py')
#
# def get_ordering(T, th=0.2):
#     res = 0
#     inds = []
#     inds_left = np.arange(T.shape[0])
#     while True:
#         o = np.argsort(T[:, 0])[::-1]
#         inds_left = inds_left[o]
#         T_ = T[o, :]
#         res += T_[0, 0]
#
#         if (T_.shape[0] == 1):
#             inds.append(inds_left[0])
#             return inds, res + (T_[0, 0])
#         elif (T_.shape[1] == 1):
#             inds.extend(inds_left)
#             return inds, res + np.sum(T_[:, 0])
#         else:
#             if T_[1, 0] + th >= np.max(T_[1:, 1]) :
#                 T = T_[1:, :]
#             else:
#                 T = T_[1:, 1:]
#             inds.append(inds_left[0])
#             inds_left = inds_left[1:]
#
#     return inds, res
#
# taskname = "CDDM"
# mfr_thr = 0.06
# std_thr = 0.05
# n_steps = 300
# downsample_window = 10
# task_params = {}
# task_params["seed"] = 0
# task_params["n_steps"] = n_steps
# task_params["cue_on"] = 0
# task_params["cue_off"] = n_steps
# task_params["stim_on"] = n_steps // 3
# task_params["stim_off"] = n_steps
# task_params["dec_on"] = 2 * n_steps // 3
# task_params["dec_off"] = n_steps
# task_params["coherences"] = [-0.8, -0.4, -0.1, 0, 0.1, 0.4, 0.8]
# num_inputs = 6
# num_outputs = 2
#
# # taskname = 'DMTS'
# # n_inputs = num_inputs = 3
# # n_outputs = n_outputs = 2
# # mfr_thr = 0.01
# # std_thr = 0.01
# # T = 120
# # dt = 1
# # downsample_window = 10
# # n_steps = int(T / dt)
# # task_params = dict()
# # task_params["n_steps"] = n_steps
# # task_params["n_inputs"] = n_inputs
# # task_params["n_outputs"] = n_outputs
# # task_params["stim_on_sample"] = n_steps // 10
# # task_params["stim_off_sample"] = 2 * n_steps // 10
# # task_params["stim_on_match"] = 3 * n_steps // 10
# # task_params["stim_off_match"] = 4 * n_steps // 10
# # task_params["dec_on"] = 5 * n_steps // 10
# # task_params["dec_off"] = n_steps
# # task_params["random_window"] = 0
# # task_params["seed"] = 0
#
#
# # taskname = 'MemoryAntiAngle'
# # n_inputs = num_inputs = 4
# # n_outputs = num_outputs = 3
# # mfr_thr = 0.01
# # std_thr = 0.01
# # downsample_window = 10
# # score_constraint = 0.0015
# # T = 120
# # dt = 1
# # n_steps = int(T / dt)
# # task_params = dict()
# # task_params["stim_on"] = 2 * n_steps // 12
# # task_params["stim_off"] = 3 * n_steps // 12
# # task_params["random_window"] = 0
# # task_params["recall_on"] = 8 * n_steps // 16
# # task_params["recall_off"] = n_steps
# # task_params["seed"] = 0
#
# # taskname = 'AngleIntegration'
# # n_inputs = num_inputs = 3
# # n_outputs = num_outputs = 6
# # mfr_thr = 0.01
# # std_thr = 0.01
# # downsample_window = 10
# # score_constraint = 0.0015
# # T = 320
# # dt = 1
# # n_steps = int(T / dt)
# # task_params = dict()
# # task_params["w"] = 0.1 / (2 * np.pi)
# # task_params["seed"] = 0
#
# df = pickle.load(open(f"../data/{taskname}.pkl", 'rb+'))
#
#
# for activation_type in ["tanh"]:
#     filter = (df.activation == activation_type)
#     df_filtered = df.loc[filter]
#
#     if activation_type == 'relu':
#         filter = (df_filtered.constrained == True) & (df_filtered.RNN_score <= 0.009)
#         df_filtered = df_filtered.loc[filter]
#         if taskname == "CDDM":
#             n_stepss = [l["n_steps"] for l in df_filtered['task_params'].tolist()]
#             filter = [(True if el == 300 else False) for el in n_stepss]
#             df_filtered = df_filtered.loc[filter]
#
#     if activation_type == 'tanh':
#         if taskname == "CDDM":
#             filter = (df_filtered.weight_decay == 0.000005)
#             df_filtered = df_filtered[filter]
#             # filter = (df_filtered.max_iter_RNN == 6000)
#             # df_filtered = df_filtered[filter]
#             # filter = (df_filtered.lr_RNN == 0.005)
#             # df_filtered = df_filtered[filter]
#
#     print(f"Number of {activation_type} RNNs: {len(df_filtered)}")
#
#     print(f"weight_decay options: {np.unique(df_filtered['weight_decay'].tolist())}")
#     print(f"same_batch options: {np.unique(df_filtered['same_batch'].tolist())}")
#     print(f"max_iter_RNN options: {np.unique(df_filtered['max_iter_RNN'].tolist())}")
#     print(f"sigma_rec_RNN options: {np.unique(df_filtered['sigma_rec_RNN'].tolist())}")
#     print(f"lr_RNN options: {np.unique(df_filtered['lr_RNN'].tolist())}")
#     print(f"activation options: {np.unique(df_filtered['activation'].tolist())}")
#     print(f"constrained options: {np.unique(df_filtered['constrained'].tolist())}")
#     print(f"orth_input_only options: {np.unique(df_filtered['orth_input_only'].tolist())}")
#
#     RNN_trajectories = df_filtered["RNN_trajectories"].tolist()
#     RNNs_conditionss = df_filtered["conditions"].tolist()
#     W_inps = df_filtered["W_inp_RNN"].tolist()
#     W_recs = df_filtered["W_rec_RNN"].tolist()
#     W_outs = df_filtered["W_out_RNN"].tolist()
#     W_inps_filtered = []
#     W_recs_filtered = []
#     W_outs_filtered = []
#     num_RNNs = len(RNN_trajectories)
#     TrajectoryFeature_array_list = []
#     Connectivity_feature_array_list = []
#     for i in range((num_RNNs)):
#         K = RNN_trajectories[i].shape[-1]
#         feature_array_list = []
#         for k in range(K):
#             features = mean_downsample(RNN_trajectories[i][:, :, k], window = downsample_window)
#             feature_array_list.append(features)
#         TrajectoryFeature_array_one_RNN = np.hstack(feature_array_list)
#
#         ConnectivityFeature_array_one_RNN = np.hstack([W_inps[i], W_outs[i].T])
#         Input_feature_array_list = W_inps[i]
#         Output_feature_array_list = W_outs[i].T
#         # remove low contributing neurons and neurons with low variance!
#         mfr = np.mean(np.abs(TrajectoryFeature_array_one_RNN), axis = 1)
#         std = np.std(np.abs(TrajectoryFeature_array_one_RNN), axis = 1)
#         inds_mfr = np.where((mfr > mfr_thr))[0]
#         inds_std =  np.where((std > std_thr))[0]
#         inds = list(set(inds_mfr) & set(inds_std))
#
#         TrajectoryFeature_array_one_RNN = TrajectoryFeature_array_one_RNN[inds, :]
#         ConnectivityFeature_array_one_RNN = ConnectivityFeature_array_one_RNN[inds, :]
#         TrajectoryFeature_array_list.append(TrajectoryFeature_array_one_RNN)
#         Connectivity_feature_array_list.append(ConnectivityFeature_array_one_RNN)
#
#         W_inp = W_inps[i]
#         W_rec = W_recs[i]
#         W_out = W_outs[i]
#         W_rec_filtered = W_rec[inds, :]
#         W_rec_filtered = W_rec_filtered[:, inds]
#         W_inps_filtered.append(deepcopy(W_inp[inds, :]))
#         W_recs_filtered.append(deepcopy(W_rec_filtered))
#         W_outs_filtered.append(deepcopy(W_out[:, inds]))
#
#
#     TrajectoryFeature_array_list_scaled = []
#     for f in TrajectoryFeature_array_list:
#         TrajectoryFeature_array_list_scaled.append((f - np.min(f, axis = 1, keepdims=True))
#                                          / (np.max(f, axis = 1, keepdims=True) - np.min(f, axis = 1, keepdims=True)))
#
#     TrajectoryFeature_array = np.vstack(TrajectoryFeature_array_list)
#     TrajectoryFeature_array_scaled = np.vstack(TrajectoryFeature_array_list_scaled)
#     ConnectivityFeature_array = np.vstack(Connectivity_feature_array_list)
#
#
#     Feature_array = TrajectoryFeature_array
#     Feature_array_list = TrajectoryFeature_array_list
#
#     # Feature_array = TrajectoryFeature_array_scaled
#     # Feature_array_list = Feature_array_list_scaled
#
#     # Feature_array = np.hstack([TrajectoryFeature_array, 35*ConnectivityFeature_array])
#     # Feature_array_list = [np.hstack([TrajectoryFeature_array_list[i], 35 * Connectivity_feature_array_list[i]]) for i in range(len(TrajectoryFeature_array_list))]
#
#     # Feature_array = np.hstack([TrajectoryFeature_array_scaled, 20*ConnectivityFeature_array])
#     # Feature_array_list = [np.hstack([TrajectoryFeature_array_list_scaled[i], 20 * Connectivity_feature_array_list[i]]) for i in range(len(TrajectoryFeature_array_list_scaled))]
#
#     # Feature_array = np.hstack([TrajectoryFeature_array, 20*ConnectivityFeature_array])
#     # Feature_array = zscore(Feature_array, axis=0)
#     # Feature_array_list = [np.hstack([TrajectoryFeature_array_list[i], 20 * Connectivity_feature_array_list[i]]) for i in range(len(TrajectoryFeature_array_list))]
#     # Feature_array_list = [zscore(F, axis=0) for F in Feature_array_list]
#
#     # Feature_array = zscore(TrajectoryFeature_array, axis=0)
#     # Feature_array_list = [zscore(F, axis=0) for F in TrajectoryFeature_array_list]
#
#     # Feature_array = ConnectivityFeature_array
#     # Feature_array_list = Connectivity_feature_array_list
#
#     n = 16
#     s = 2.5
#     pca = PCA(n_components=n)
#     pca.fit(Feature_array)
#     P = pca.components_.T
#     Feature_array_pr = Feature_array @ P
#
#     # for i in range(5):
#     #     reducer = umap.UMAP()
#     #     embedding = reducer.fit_transform(Feature_array_pr)
#     #     fig, ax = plt.subplots(1, 1, figsize=(2, 2))
#     #     data = embedding
#     #     ax.scatter(data[:, 0], data[:, 1], edgecolor='b', linewidth=0.1, color='orange', s=s, alpha = 1)
#     #     fig.savefig(f"../img/representations_UMAP_{activation_type}_{i}.png", transparent=True, dpi=300)
#     #     plt.show()
#
#     # Feature_array_pr = Feature_array_pr[:10000,:]
#     #
#     # fig, axs = plt.subplots(n, n, figsize=(n * 2, n * 2))
#     # data = Feature_array_pr
#     # for i in range(n):
#     #     for j in range(n):
#     #         axs[i, j].scatter(data[:, i], data[:, j], edgecolor='gray', linewidth=0.1, color='red', s=s, alpha = 0.5)
#     # plt.show()
#     #
#     if activation_type == 'relu':
#         n_clusters = 11
#     else:
#         n_clusters = 11
#
#     #
#     # Feature_array = np.hstack([TrajectoryFeature_array, 35*ConnectivityFeature_array])
#     # Feature_array_list = [np.hstack([TrajectoryFeature_array_list[i], 35 * Connectivity_feature_array_list[i]]) for i in range(len(TrajectoryFeature_array_list))]
#
#     cl = KMeans(n_clusters=n_clusters, max_iter=3000)
#     cl.fit(Feature_array)
#     lbls = np.unique(cl.labels_)
#     colors = ["red", "green", "blue", "darkorange", "magenta", "cyan", "lime", "gray", "yellow", "deepskyblue", "olive", "purple", "black"]
#     #
#     # fig, axs = plt.subplots(n, n, figsize=(n * 1.5, n * 1.5))
#     # data = Feature_array_pr
#     # for i in range(n):
#     #     for j in range(n):
#     #         for c, k in enumerate(lbls):
#     #             inds = np.where(cl.labels_ == k)[0]
#     #             axs[i, j].scatter(data[inds, i], data[inds, j],
#     #                               edgecolor='b',
#     #                               linewidth=0.05,
#     #                               color=colors[k],
#     #                               s=s,
#     #                               alpha=0.5)
#     # plt.suptitle(f"Clustering of neural responses, {activation_type} RNN")
#     # plt.show()
#     # fig.savefig(f"../img/representations_{activation_type}.png", transparent=True, dpi=300)
#
#     # fig, axs = plt.subplots(1, 1, figsize=(2, 2))
#     # colors = ["red", "green", "blue", "yellow",
#     #           "magenta", "cyan",
#     #           "purple", 'gray', "orange",
#     #           "plum", "lime", "olive", "bisque", 'pink', "lightblue", "lightcyan"]
#     # data = Feature_array_pr
#     # axs.scatter(data[:, 0], data[:, 1], edgecolor='b', linewidth=0.075, color='orange', s=s)
#     # # axs[1].scatter(data[:, 0], data[:, 2], edgecolor='b', linewidth=0.25, color='orange', s=s)
#     # # axs[2].scatter(data[:, 1], data[:, 2], edgecolor='b', linewidth=0.25, color='orange', s=s)
#     # plt.tight_layout()
#     # plt.subplots_adjust(wspace=0, hspace=0)
#     # fig.savefig(f"../img/representations_{activation_type}.png", transparent=True, dpi=300)
#     # plt.show()
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     data = Feature_array_pr
#     ax.scatter(data[:, 0], data[:, 1], data[:, 2], edgecolor='b', linewidth=0.2, color='orange', s=s)
#     plt.suptitle(f"Clustering of neural responses, {activation_type} RNN, 3D")
#     plt.show()
#
#     # cnt_list = []
#     # for i in range(len(RNN_trajectories)):
#     #     F = Feature_array_list_scaled[i]
#     #     labels_ = cl.predict(F)
#     #
#     #     cluster_counts = []
#     #     for c in range(cl.n_clusters):
#     #         cluster_counts.append(len(np.where(labels_ == c)[0]))
#     #
#     #     cluster_counts = np.array(cluster_counts)
#     #     cnt_list.append(deepcopy(cluster_counts))
#     #
#     # cl_count_array = np.array(cnt_list)
#     # cl_count_array_z = cl_count_array - np.mean(cl_count_array, axis=0)
#     # pca = PCA(n_components=2)
#     # pca.fit(cl_count_array)
#     # a = cl_count_array @ pca.components_.T
#     # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
#     # ax.scatter(a[:, 0], a[:, 1], color='m', edgecolor='b')
#     # plt.suptitle("Cluster-configuration similarity")
#     # plt.show()
#
#     # plotting the effective connectivity
#     input_to_cluster_dict = {}
#     cluster_to_cluster_dict = {}
#     cluster_to_output_dict = {}
#     for i in range(num_inputs):
#         for j in range(n_clusters):
#             input_to_cluster_dict[f"({j},{i})"] = []
#
#     for i in range(n_clusters):
#         for j in range(num_outputs):
#             cluster_to_output_dict[f"({j},{i})"] = []
#
#     for i in range(n_clusters):
#         for j in range(n_clusters):
#             cluster_to_cluster_dict[f"({j},{i})"] = []
#
#     # for each RNN in the list:
#     for num_RNN, RNN_features in enumerate(Feature_array_list):
#         # predict labels for each neuron
#         labels = cl.predict(RNN_features)
#         W_inp = W_inps_filtered[num_RNN]
#         W_rec = W_recs_filtered[num_RNN]
#         W_out = W_outs_filtered[num_RNN]
#         for nrn in range(len(labels)):
#             for inp_ch in range(W_inp.shape[1]):
#                 input_to_cluster_dict[f"({labels[nrn]},{inp_ch})"].append(W_inp[nrn, inp_ch])
#             for out_ch in range(W_out.shape[0]):
#                 cluster_to_output_dict[f"({out_ch},{labels[nrn]})"].append(W_out[out_ch, nrn])
#             for nrn_receiver in range(W_rec.shape[0]):
#                 cluster_to_cluster_dict[f"({labels[nrn_receiver]},{labels[nrn]})"].append(W_rec[nrn_receiver, nrn])
#     eff_W_inp = np.zeros((n_clusters, num_inputs))
#     eff_W_rec = np.zeros((n_clusters, n_clusters))
#     eff_W_out = np.zeros((num_outputs, n_clusters))
#
#     for i in range(n_clusters):
#         for j in range(num_inputs):
#             eff_W_inp[i, j] = np.mean(input_to_cluster_dict[f"({i},{j})"])
#
#     for i in range(n_clusters):
#         for j in range(n_clusters):
#             eff_W_rec[i, j] = np.mean(cluster_to_cluster_dict[f"({i},{j})"])
#
#     for i in range(num_outputs):
#         for j in range(n_clusters):
#             eff_W_out[i, j] = np.mean(cluster_to_output_dict[f"({i},{j})"])
#
#     effective_connectivity_dict = {"W_inp_eff" : eff_W_inp,
#                                    "W_rec_eff" : eff_W_rec,
#                                    "W_out_eff" : eff_W_out}
#
#     pickle.dump(effective_connectivity_dict, open(f"../data/effective_connectivity_{activation_type}_trajectoriesonly.pkl", "wb+"))
#     # effective_connectivity_dict = pickle.load(open(f"../data/effective_connectivity_{activation_type}.pkl", "rb+"))
#     W_inp = effective_connectivity_dict["W_inp_eff"]
#     W_rec = effective_connectivity_dict["W_rec_eff"]
#     W_out = effective_connectivity_dict["W_out_eff"]
#
#     if activation_type == 'relu':
#         idxs = [1, 5, 3, 2, 0, 4, 8, 9, 6, 10, 7]
#     else:
#         idxs = [9, 1, 5, 7, 6, 8, 2, 3, 4, 0, 10]
#         idxs, res = get_ordering(W_inp, th = 0.12)
#     W_inp = permute_input_matrix(W_inp, idxs)
#     W_rec = permute_recurrent_matrix(W_rec, idxs)
#     W_out = permute_output_matrix(W_out, idxs)
#
#     fig, ax = plt.subplots(3, 1, figsize=(2, 3), constrained_layout=False,
#                            gridspec_kw={'height_ratios': [0.5, W_rec.shape[0] / 6.0, 2.0 / 6.0]})
#     ax[0].imshow(W_inp.T, cmap='bwr', vmin=-0.5, vmax=0.5, aspect=0.3)
#     ax[1].imshow(W_rec, cmap='bwr', vmin=-0.5, vmax=0.5, aspect='equal')
#     ax[2].imshow(W_out, cmap='bwr', vmin=-0.5, vmax=0.5, aspect='equal')
#     ax[0].set_xticks([])
#     ax[1].set_xticks([])
#     ax[2].set_xticks(np.arange(W_rec.shape[0]))
#     ax[1].set_yticks(np.arange(W_rec.shape[0]))
#
#     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
#     plt.tight_layout()
#     plt.savefig(f"../img/inp_rec_out_matrices_{activation_type}_trajectoriesonly.png", dpi=300, transparent=True, bbox_inches='tight')
#     plt.show()
#
#     #
#     # Feature_array = TrajectoryFeature_array
#     # Feature_array_list = TrajectoryFeature_array_list
#     #
#     # pca = PCA(n_components=n)
#     # pca.fit(Feature_array)
#     # P = pca.components_.T
#     # Feature_array_pr = Feature_array @ P
#
#     fig, ax = plt.subplots(1, 1, figsize=(2, 2))
#     data = Feature_array_pr
#     for i, j in enumerate(lbls):
#         inds = np.where(cl.labels_ == j)[0]
#         print(np.round(np.mean(ConnectivityFeature_array[inds],axis = 0),2), colors[i])
#         ax.scatter(data[inds, 0], data[inds, 1], edgecolor='b', linewidth=0.00, color=colors[i], s=s, alpha = 1)
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     fig.savefig(f"../img/representations_{activation_type}_xy_trajectoriesonly.png", transparent=True, dpi=300)
#     plt.show()
#
#     fig, ax = plt.subplots(1, 1, figsize=(2, 2))
#     for i, j in enumerate(lbls):
#         inds = np.where(cl.labels_ == j)[0]
#         ax.scatter(data[inds, 0], data[inds, 2], edgecolor='b', linewidth=0.00, color=colors[i], s=s, alpha = 1)
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     fig.savefig(f"../img/representations_{activation_type}_xz_trajectoriesonly.png", transparent=True, dpi=300)
#     plt.show()