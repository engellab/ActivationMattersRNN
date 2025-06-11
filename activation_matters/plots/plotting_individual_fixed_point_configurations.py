import os

from activation_matters.analysis.fixed_point_analysis import ICP_registration, get_fp_data_dict
from activation_matters.plots.ploting_utils import normalize_color, plot_aligned_FPs, plot_fixed_points
from activation_matters.plots.style.style_setup import set_up_plotting_styles
import numpy as np
import hydra
from omegaconf import OmegaConf
# OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def plot_fixed_point_configurations(cfg):
    n_dim = 2
    n_nets_to_plot = 10
    seed = cfg.seed
    show = False
    save = True
    taskname = cfg.task.taskname 
    img_save_folder = os.path.join(cfg.paths.img_folder, taskname)
    set_up_plotting_styles(cfg.paths.style_path)
    path_to_folder = os.path.join(cfg.paths.fixed_points_data_folder, taskname)
    aux_datasets_folder = os.path.join(cfg.paths.auxilliary_datasets_path, taskname)
    control_type = cfg.control_type
    n_components = cfg.task.dynamical_topology_analysis.n_PCs
    net_types = ["relu", "sigmoid", "tanh"]
    dataSegment = cfg.dataSegment
    data_dict = get_fp_data_dict(path_to_folder, net_types, control_type, dataSegment, n_nets_to_plot, n_components)
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

    for net_type in net_types:
        for constrained in [True, False]:
            for control in [False, True]:
                print(f"net_type = {net_type}; constrained = {constrained}; control = {control}")
                fp_dict = data_dict[net_type][f"constrained={constrained}"][f"control={control}"]
                # find a target network (it has to have maximum number of fp)
                Qs = []
                fp_list = fp_dict["fp_list"]
                labels_list = fp_dict["labels_list"]
                for i in range(len(fp_list)):
                    Q, score = ICP_registration(points_source=fp_list[i], labels_source=labels_list[i],
                                                points_target=fp_list[0], labels_target=labels_list[0],
                                                max_iter=1000, tol=1e-10, seed=seed + i)
                    Qs.append(np.copy(Q))
                path = os.path.join(img_save_folder, f"registered_fp_struct_{net_type}_controlType={control_type}_control={control}_constrained={constrained}_source_{n_dim}D")

                plot_aligned_FPs(fp_list=fp_dict["fp_list"],
                                 labels_list=fp_dict["labels_list"],
                                 transforms_list=Qs,
                                 colors_stable=colors_stable,
                                 colors_unstable=colors_unstable,
                                 markers=markers,
                                 edgecolors=edgecolors,
                                 n_dim=n_dim,
                                 save=save,
                                 show=show,
                                 path=path)

    fp_dict_combined = {}
    fp_dict_combined["fp_list"] = []
    fp_dict_combined["labels_list"] = []
    legends = []
    inds_list = []
    for net_type in net_types:
        for constrained in [True, False]:
            for control in [True, False]:
                print(f"net_type={net_type}; control={control}; controlType={control_type}; constrained={constrained}")
                fp_dict = data_dict[net_type][f"constrained={constrained}"][f"control={control}"]
                # take only n_nets_MDS
                n_nets = len(fp_dict["fp_list"])
                legends.append(f"{net_type}_constrained={constrained}")
                # plot fixed points individually:
                for i in range(n_nets_to_plot):
                    path = os.path.join(img_save_folder, f"fp_struct_{net_type}_constrained={constrained}_controlType={control_type}_control={control}_net={i}.pdf")
                    plot_fixed_points(fixed_point_struct=fp_dict["fp_list"][i], fp_labels=fp_dict["labels_list"][i],
                                      colors=colors,
                                      markers=markers,
                                      edgecolors=edgecolors,
                                      n_dim=2, show=show, save=save, path=path)
                    path = os.path.join(img_save_folder, f"fp_struct3D_{net_type}_constrained={constrained}_controlType={control_type}_control={control}_net={i}.pdf")
                    plot_fixed_points(fixed_point_struct=fp_dict["fp_list"][i], fp_labels=fp_dict["labels_list"][i],
                                      colors=colors,
                                      markers=markers,
                                      edgecolors=edgecolors,
                                      n_dim=3, show=show, save=save, path=path)
    fp_dict_combined["inds_list"] = inds_list
    fp_dict_combined["legends"] = legends
    # pickle.dump(fp_dict_combined, open(os.path.join(aux_datasets_folder, f"FP_controlType={control_type}_control={control}.pkl"), "wb+"))
    return None


if __name__ == '__main__':
    plot_fixed_point_configurations()