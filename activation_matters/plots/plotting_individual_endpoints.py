import numpy as np
import os
from trainRNNbrain.training.training_utils import prepare_task_arguments
from activation_matters.plots.ploting_utils import interpolate_color, plot_endpoints
import hydra
from activation_matters.plots.style.style_setup import set_up_plotting_styles
import pickle


@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def plot_individual_endpoints(cfg):
    show = False
    save = True
    n_nets_to_plot = 10
    control_type = cfg.control_type
    taskname = cfg.task.taskname
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    set_up_plotting_styles(cfg.paths.style_path)
    aux_datasets_folder = os.path.join(cfg.paths.auxilliary_datasets_path, taskname)
    img_save_folder = os.path.join(cfg.paths.img_folder, taskname)
    seed = cfg.seed

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    if taskname == "CDDM":
        task.coherences = np.array([-1, -0.5, 0, 0.5, 1.0])
    if hasattr(task, 'random_window'):
        task.random_window = 0
    task.seed = seed  # for consistency
    inputs, targets, conditions = task.get_batch()

    data_dict = pickle.load(
        open(os.path.join(aux_datasets_folder, f"trajectory_endpoints_{dataSegment}{n_nets}_{control_type}.pkl"), "rb+"))
    legends = data_dict["legends"]
    inds_list = data_dict["inds_list"]

    if taskname == "CDDM":
        point_colors, line_colors, markers, linewidth = get_plotting_params_CDDM(conditions)
    elif taskname == "GoNoGo" or taskname == "MemoryNumber":
        point_colors, line_colors, markers, linewidth = get_plotting_params_GoNoGo(conditions)

    # plotting the endpoints
    for k, legend in enumerate(legends):
        if not "shuffled=True" in legend:
            RNN_endpoints_processed = data_dict["RNN_trajectory_endpoints_processed"]
            for i in range(n_nets_to_plot):
                # for axes in [[0, 1, 2], [0, 2, 3], [1, 2, 3]]:
                for axes in [[0, 1]]:
                    endpoints = RNN_endpoints_processed[inds_list[k][i]]
                    path = os.path.join(img_save_folder, f"{taskname}_endpoints_{legend}_{axes}_{i}.pdf")
                    plot_endpoints(endpoints=endpoints,
                                   axes=axes,
                                   face_colors=point_colors,
                                   edge_colors=line_colors,
                                   markers = markers,
                                   save=save,
                                   show=show,
                                   path=path,
                                   n_dim=2)
                    # path = os.path.join(img_save_folder, f"{taskname}_endpoints3D_{legend}_{axes}_{i}.pdf")
                    # plot_endpoints(endpoints=endpoints,
                    #                axes=axes,
                    #                face_colors=point_colors,
                    #                edge_colors=line_colors,
                    #                markers = markers,
                    #                save=save,
                    #                show=show,
                    #                path=path,
                    #                n_dim=3)
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
    markers = ["p" if context == 1 else "o" for context in contexts]
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
    plot_individual_endpoints()