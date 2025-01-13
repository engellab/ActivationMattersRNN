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
OmegaConf.register_new_resolver("eval", eval)


taskname = "CDDM"
show = False
save = True
n_nets = 5

# @hydra.main(version_base="1.3", config_path=f"../../configs/task/", config_name=f'{taskname}')
@hydra.main(version_base="1.3", config_path=f"../../configs/task/", config_name=f'{taskname}')
def plot_trajectories(cfg: OmegaConf):
    set_up_plotting_styles(cfg.task.paths.style_path)
    aux_datasets_folder = cfg.task.paths.auxilliary_datasets_path
    img_folder = cfg.task.paths.img_folder

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    if taskname == "CDDM":
        task.coherences = np.array([-1, -0.5, 0, 0.5, 1.0])
    if hasattr(task, 'random_window'):
        task.random_window = 0
    task.seed = 0 # for consistency
    inputs, targets, conditions = task.get_batch()

    data_dict = pickle.load(open(os.path.join(aux_datasets_folder, "trajectories_data.pkl"), "rb+"))
    if taskname == "CDDM":
        point_colors, line_colors, markers, linewidth = get_plotting_params_CDDM(conditions)
    elif taskname == "GoNoGo" or taskname == "MemoryNumber":
        point_colors, line_colors, markers, linewidth = get_plotting_params_GoNoGo(conditions)

    legends = data_dict["legends"]
    inds_list = data_dict["inds_list"]

    # plotting the trajectories
    for k, legend in enumerate(legends):
        if "shuffle=False" in legend:
            RNN_trajectories_projected = data_dict["RNN_trajectories_projected"]
            for i in range(n_nets):
                for axes in [[0, 1, 2], [0, 2, 3], [1, 2, 3]]:
                    path = os.path.join(img_folder, f"{taskname}_trajectories_{legend}_{axes}_{i}.pdf")
                    trajectories_projected = RNN_trajectories_projected[inds_list[k][i]]

                    plot_projected_trajectories(trajectories_projected=trajectories_projected,
                                                axes=axes[:2],
                                                legend=legend,
                                                save=save, path=path,
                                                line_colors=line_colors,
                                                point_colors=point_colors,
                                                markers=markers,
                                                linewidth=linewidth,
                                                show=False,
                                                n_dim=2)

    # VISUALIZE MDS EMBEDDING OF TRAJECTORIES
    Mat = pickle.load(open(os.path.join(aux_datasets_folder, "trajectories_similarity_matrix.pkl"), "rb"))
    path = os.path.join(img_folder, f"trajectory_similarity_matrix.pdf")
    if show:
        plot_similarity_matrix(Mat, save=save, path=path)
    print("Computing the trajectory embedding")
    colors = ["red", "red",
              "orange", "orange",
              "blue", "blue",
              "deepskyblue", "deepskyblue",
              "green", "green",
              "lightgreen", "lightgreen"]
    hatch = ["", "", "", "", "", "", "", "", "", "", "", ""]
    markers = ["o", "v", "o", "v", "o", "v", "o", "v", "o", "v", "o", "v"]

    np.fill_diagonal(Mat, 0)
    for attempt in range(5):
        mds = MDS(n_components=2, dissimilarity='precomputed', n_init=101, eps=1e-6, max_iter=1000)
        mds.fit(Mat)
        embedding = mds.embedding_
        path = os.path.join(img_folder, f"MDS_trajectory_attempt={attempt}_horizontal.pdf")
        plot_embedding(embedding, inds_list, legends, colors, hatch, markers,
                       show_legends=False, save=save, path=path, show=show)

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
    plot_trajectories()