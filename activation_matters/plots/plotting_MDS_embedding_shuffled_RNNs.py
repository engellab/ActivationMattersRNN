import numpy as np
import os
from trainRNNbrain.training.training_utils import prepare_task_arguments
from sklearn.manifold import MDS
from activation_matters.plots.ploting_utils import interpolate_color, plot_projected_trajectories, plot_similarity_matrix, \
    plot_embedding
import hydra
from omegaconf import OmegaConf
from style.style_setup import set_up_plotting_styles
import pickle
# OmegaConf.register_new_resolver("eval", eval)
import ray
import re
from itertools import chain

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

# @hydra.main(version_base="1.3", config_path=f"../../configs/task/", config_name=f'{taskname}')
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def plot_MDS_embedding_connectivity_shuffled_RNNs(cfg: OmegaConf):
    show = False
    save = True
    modality = cfg.modality
    taskname = cfg.task.taskname
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    set_up_plotting_styles (cfg.paths.style_path)
    aux_datasets_folder = os.path.join(cfg.paths.auxilliary_datasets_path, taskname)
    img_save_folder = os.path.join(cfg.paths.img_folder, taskname)

    print(dataSegment)

    data_dict = pickle.load(open(os.path.join(aux_datasets_folder, f"{modality}_{dataSegment}{n_nets}.pkl"), "rb+"))
    legends = data_dict["legends"]
    inds_list = data_dict["inds_list"]
    # # VISUALIZE MDS EMBEDDING OF TRAJECTORIES
    # print("Computing the trajectory embedding")
    colors = ["red", "red",
              "orange", "orange",
              "blue", "blue",
              "deepskyblue", "deepskyblue",
              "green", "green",
              "lightgreen", "lightgreen"]
    hatch = ["", "", "", "", "", "", "", "", "", "", "", ""]
    markers = ["o", "v", "o", "v", "o", "v", "o", "v", "o", "v", "o", "v"]

    #getting shuffled networks only
    legends_sh, inds_list_sh, colors_sh, hatch_sh, markers_sh = zip(*[(l, i, c, h, m) for l, i, c, h, m in zip(legends, inds_list, colors, hatch, markers) if "shuffle=True" in l])
    inds_to_take = list(chain(*inds_list_sh))
    inds_list_new = [len(inds_list_sh[i]) for i in range(len(inds_list_sh))]
    inds_list_new = [np.arange(el) for el in inds_list_new]

    cnt = 0
    for i, inds in enumerate(inds_list_new):
        inds_list_new[i] += cnt
        cnt += len(inds_list_new[i])
    inds_list_sh = inds_list_new

    Mat = pickle.load(open(os.path.join(aux_datasets_folder, f"{modality}_similarity_matrix_{dataSegment}{n_nets}.pkl"), "rb"))
    Mat_subsampled = Mat[np.ix_(inds_to_take, inds_to_take)]
    path = os.path.join(img_save_folder, f"{modality}_matrix_shuffled_connectivity_{dataSegment}{n_nets}.pdf")
    if show:
        plot_similarity_matrix(Mat_subsampled, save=save, path=path)

    np.fill_diagonal(Mat_subsampled, 0)

    img_name = f"MDS_{modality}_attempt=XXX_{dataSegment}{n_nets}_shuffled_connectivity.pdf"
    ray.init()
    # Launch tasks in parallel
    results = [
        run_mds_and_plot.remote(cfg, attempt, img_name, Mat_subsampled, img_save_folder, inds_list_sh,
                                legends_sh, colors_sh, hatch_sh, markers_sh, save, show)
        for attempt in range(3)]
    # Retrieve results (if needed)
    embeddings = ray.get(results)
    ray.shutdown()
    return None

def get_plotting_params_CDDM(conditions):
    contexts = np.array([1 if conditions[i]['context'] == 'motion' else -1 for i in range(len(conditions))])
    relevant_coherences = [conditions[i]["color_coh"]
                           if conditions[i]["context"] == "color"
                           else conditions[i]["motion_coh"] for i in range(len(conditions))]
    irrelevant_coherences = [conditions[i]["motion_coh"]
                             if conditions[i]["context"] == "color"
                             else conditions[i]["color_coh"] for i in range(len(conditions))]
    primary_colors = np.array([[0.3, 0.4, 0.8], [0.8, 0.8, 0.8], [0.8, 0.2, 0.3]])

    line_colors = []
    low_val = np.min(relevant_coherences)
    high_val = np.max(relevant_coherences)
    mid_val = np.mean(relevant_coherences)
    for coherence in relevant_coherences:
        color = interpolate_color(low_color=primary_colors[0],
                                  mid_color=primary_colors[1],
                                  high_color=primary_colors[2],
                                  low_val=low_val,
                                  mid_val=mid_val,
                                  high_val=high_val,
                                  val=coherence)
        line_colors.append(color)

    point_colors = []
    low_val = np.min(irrelevant_coherences)
    high_val = np.max(irrelevant_coherences)
    mid_val = np.mean(irrelevant_coherences)
    for coherence in irrelevant_coherences:
        color = interpolate_color(low_color=primary_colors[0],
                                  mid_color=primary_colors[1],
                                  high_color=primary_colors[2],
                                  low_val=low_val,
                                  mid_val=mid_val,
                                  high_val=high_val,
                                  val=coherence)
        point_colors.append(color)
    markers = ["p" if context == 1 else "*" for context in contexts]
    linewidth = 0.75
    return point_colors, line_colors, markers, linewidth

def get_plotting_params_GoNoGo(conditions):
    line_colors = []
    primary_colors = np.array([[0.3, 0.4, 0.8], [0.8, 0.8, 0.8], [0.8, 0.2, 0.3]])
    values = [float(str(condition).split(",")[0].split(":")[1]) for condition in conditions]
    low_val = np.min(values)
    high_val = np.max(values)
    mid_val = np.mean(values)
    for value in values:
        color = interpolate_color(low_color=primary_colors[0],
                                  mid_color=primary_colors[1],
                                  high_color=primary_colors[2],
                                  low_val=low_val,
                                  mid_val=mid_val,
                                  high_val=high_val,
                                  val=value)
        line_colors.append(color)
    markers = ["o" for val in values]
    linewidth = 1.5
    return None, line_colors, markers, linewidth

if __name__ == '__main__':
    plot_MDS_embedding_connectivity_shuffled_RNNs()