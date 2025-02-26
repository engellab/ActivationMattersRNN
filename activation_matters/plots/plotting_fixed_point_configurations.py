import os
from scipy.linalg import orthogonal_procrustes
from scipy.stats import ortho_group
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from copy import copy
import pickle
from activation_matters.plots.ploting_utils import normalize_color, plot_embedding, plot_aligned_FPs, plot_similarity_matrix, \
    plot_fixed_points
from style.style_setup import set_up_plotting_styles
import numpy as np
import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import ray
import re
# OmegaConf.register_new_resolver("eval", eval)

@ray.remote
def run_mds_and_plot(cfg, attempt, img_name, Mat, img_save_folder, inds_list, legends, colors, hatch, markers, save, show):
    set_up_plotting_styles(cfg.paths.style_path)
    # Run MDS
    img_name = re.sub(r"XXX", str(attempt), img_name)
    mds = MDS(n_components=2, dissimilarity='precomputed', n_init=101, eps=1e-6, max_iter=1000)
    mds.fit(Mat)
    embedding = mds.embedding_

    # Save the plot
    path = os.path.join(img_save_folder, img_name)
    plot_embedding(embedding, inds_list, legends, colors, hatch, markers,
                   show_legends=False, save=save, path=path, show=show)
    return embedding

@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def plot_fixed_point_configurations(cfg):
    n_dim = 2
    show = False
    save = True
    normalized = True
    taskname = cfg.task.taskname
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    img_save_folder = os.path.join(cfg.paths.img_folder, taskname)
    set_up_plotting_styles(cfg.paths.style_path)
    path_to_folder = os.path.join(cfg.paths.fixed_points_data_folder, taskname)
    aux_datasets_folder = os.path.join(cfg.paths.auxilliary_datasets_path, taskname)
    n_components = cfg.task.dynamical_topology_analysis.n_PCs
    net_types = ["relu", "sigmoid", "tanh"]
    data_dict = get_fp_data_dict(path_to_folder, net_types, n_components)
    n_fp_types = len(cfg.task.dynamical_topology_analysis.colors)
    colors = [list(cfg.task.dynamical_topology_analysis.colors[k]) for k in range(n_fp_types)]
    colors_stable = [normalize_color(el[0]) for el in colors]
    colors_unstable = [normalize_color(el[1]) for el in colors]

    if OmegaConf.select(cfg.task.dynamical_topology_analysis, "markers") is not None:
        markers = [list(cfg.task.dynamical_topology_analysis.markers[k]) for k in range(n_fp_types)]
    else:
        markers = None
    if OmegaConf.select(cfg.task.dynamical_topology_analysis, "edgecolors") is None or cfg.task.dynamical_topology_analysis.edgecolors == 'None':
        edgecolors = None
    else:
        edgecolors = [list(cfg.task.dynamical_topology_analysis.edgecolors[k]) for k in
                      range(n_fp_types)]

    # for net_type in net_types:
    #     for constrained in [True, False]:
    #         for shuffle in [False, True]:
    #             print(f"net_type = {net_type}; constrained = {constrained}; shuffle = {shuffle}")
    #             fp_dict = data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]
    #             # find a target network (it has to have maximum number of fp)
    #             Qs = []
    #             fp_list = fp_dict["fp_list"]
    #             labels_list = fp_dict["labels_list"]
    #             for i in range(len(fp_list)):
    #                 Q, score = ICP_registration(points_source=fp_list[i], labels_source=labels_list[i],
    #                                             points_target=fp_list[0], labels_target=labels_list[0],
    #                                             max_iter=1000, tol=1e-10)
    #                 Qs.append(np.copy(Q))
    #             path = os.path.join(img_save_folder, f"registered_fp_strcut_{net_type}_shuffle={shuffle}_constrained={constrained}_source_{n_dim}D")
    #
    #             plot_aligned_FPs(fp_list=fp_dict["fp_list"],
    #                              labels_list=fp_dict["labels_list"],
    #                              transforms_list=Qs,
    #                              colors_stable=colors_stable,
    #                              colors_unstable=colors_unstable,
    #                              markers=markers,
    #                              edgecolors=edgecolors,
    #                              n_dim=n_dim,
    #                              save=save,
    #                              show=show,
    #                              path=path)
    #
    # fp_dict_combined = {}
    # fp_dict_combined["fp_list"] = []
    # fp_dict_combined["labels_list"] = []
    # legends = []
    # inds_list = []
    # cnt = 0
    # for net_type in net_types:
    #     for constrained in [True, False]:
    #         for shuffle in [False]:
    #             fp_dict = data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]
    #             # take only n_nets_MDS
    #             n_nets = len(fp_dict["fp_list"])
    #             n_nets_taken = min([n_nets_MDS, n_nets])
    #             fp_dict_combined["fp_list"].extend([fp_dict["fp_list"][i] for i in range(n_nets_taken)])
    #             fp_dict_combined["labels_list"].extend([fp_dict["labels_list"][i] for i in range(n_nets_taken)])
    #             inds_list.append(cnt + np.arange(n_nets_taken))
    #             cnt += n_nets_taken
    #             legends.append(f"{net_type}_constrained={constrained}")
    #             # plot fixed points individually:
    #             for i in range(n_nets):
    #                 path = os.path.join(img_save_folder, f"fp_struct_{net_type}_constrained={constrained}_shuffle={shuffle}_net={i}.pdf")
    #                 plot_fixed_points(fixed_point_struct=fp_dict["fp_list"][i], fp_labels=fp_dict["labels_list"][i],
    #                                   colors=colors,
    #                                   markers=markers,
    #                                   edgecolors=edgecolors,
    #                                   n_dim=2, show=show, save=save, path=path)
    #                 path = os.path.join(img_save_folder, f"fp_struct3D_{net_type}_constrained={constrained}_shuffle={shuffle}_net={i}.pdf")
    #                 plot_fixed_points(fixed_point_struct=fp_dict["fp_list"][i], fp_labels=fp_dict["labels_list"][i],
    #                                   colors=colors,
    #                                   markers=markers,
    #                                   edgecolors=edgecolors,
    #                                   n_dim=3, show=show, save=save, path=path)
    # fp_dict_combined["inds_list"] = inds_list
    # fp_dict_combined["legends"] = legends
    # pickle.dump(fp_dict_combined, open(os.path.join(aux_datasets_folder, "FP_similarity.pkl"), "wb+"))

    fp_dict_combined = pickle.load(open(os.path.join(aux_datasets_folder, f"FP_similarity_normalized={normalized}_{dataSegment}{n_nets}.pkl"), "rb+"))
    fp_list = fp_dict_combined["fp_list"]
    labels_list = fp_dict_combined["labels_list"]
    inds_list = fp_dict_combined["inds_list"]
    legends = fp_dict_combined["legends"]

    Mat = pickle.load(open(os.path.join(aux_datasets_folder, f"FP_similarity_matrix_normalized={normalized}_{dataSegment}{n_nets}.pkl"), "rb"))
    path = os.path.join(img_save_folder, f"fp_struct_similarity_matrix_normalized={normalized}_{dataSegment}{n_nets}.pdf")
    if show:
        plot_similarity_matrix(Mat, save=save, path=path)

    colors = ["red", "red",
              "orange", "orange",
              "blue", "blue",
              "deepskyblue", "deepskyblue",
              "green", "green",
              "lightgreen", "lightgreen"]
    hatch = ["", "", "", "", "", "", "", "", "", "", "", ""]
    markers = ["o", "v", "o", "v", "o", "v", "o", "v", "o", "v", "o", "v"]
    np.fill_diagonal(Mat, 0)

    # for attempt in range(3):
    #     mds = MDS(n_components=2, dissimilarity='precomputed', n_init=101, eps=1e-6, max_iter=1000)
    #     mds.fit(Mat)
    #     embedding = mds.embedding_
    #     path = os.path.join(img_save_folder, f"MDS_fp_attempt={attempt}_nPCs={n_components}_normalized={normalized}_{dataSegment}{n_nets}.pdf")
    #     plot_embedding(embedding, inds_list, legends, colors, hatch, markers,
    #                    show_legends=False, save=save, path=path, show=show)

    img_name = f"MDS_fp_attempt=XXX_nPCs={n_components}_normalized={normalized}_{dataSegment}{n_nets}.pdf"
    ray.init()
    # Launch tasks in parallel
    results = [
        run_mds_and_plot.remote(cfg, attempt, img_name, Mat, img_save_folder,
                                inds_list, legends, colors, hatch, markers, save, show)
        for attempt in range(3)]
    embeddings = ray.get(results)
    ray.shutdown()
    return None



def get_fp_data_dict(path_to_folder, net_types, n_components):
    folder = os.listdir(path_to_folder)
    data_dict = {}
    for net_type in net_types:
        data_dict[net_type] = {}
        for constrained in [True, False]:
            data_dict[net_type][f"constrained={constrained}"] = {}
            for shuffle in [False, True]:
                data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"] = {}
                data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["fp_list"] = []
                data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["labels_list"] = []
                data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["fp_dict_list"] = []

        for file in folder:
            if net_type in file:
                for constrained in [True, False]:
                    if f"constrained={constrained}" in file:
                        for shuffle in [False, True]:
                            if f"shuffle={shuffle}" in file:
                                path = os.path.join(path_to_folder, file)
                                opened_file = open(path, "rb+")
                                fixed_points_dict = pickle.load(opened_file)
                                fixed_points = fixed_points_dict["fps"]
                                labels = fixed_points_dict["labels"]

                                if type(fixed_points) == list:
                                    fixed_points_pr = np.zeros((1, n_components))
                                else:
                                    pca = PCA(n_components=n_components)
                                    pca.fit(fixed_points)
                                    fixed_points_pr = fixed_points @ pca.components_.T

                                d = {}
                                input_IDs = np.unique(np.array([int(l.split("_")[-1]) for l in labels]))
                                for i in input_IDs:
                                    for point_type in ["sfp", "ufp"]:
                                        if f"{point_type}_{i}" in np.unique(labels):
                                            inds = np.where(np.array(labels) == f"{point_type}_{i}")
                                            d[f"{point_type}_{i}"] = fixed_points_pr[inds]
                                        else:
                                            d[f"{point_type}_{i}"] = np.array([])

                                data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["fp_list"].append(np.copy(fixed_points_pr))
                                data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["labels_list"].append(np.copy(labels))
                                data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["fp_dict_list"].append(copy(d))
    return data_dict

if __name__ == '__main__':
    plot_fixed_point_configurations()