import re

import numpy as np
import os

from sklearn.decomposition import PCA
from trainRNNbrain.training.training_utils import prepare_task_arguments

from activation_matters.analysis.selectivity_analysis import ICP_registration
from activation_matters.plots.ploting_utils import interpolate_color, plot_trajectories, plot_selectivities
import hydra
from activation_matters.plots.style.style_setup import set_up_plotting_styles
import pickle

@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def plot_individual_selectivities(cfg):
    show = False
    save = True
    n_nets_to_plot = 3
    n_dims_to_take = 3
    n_nets_to_aggregate = 30
    control_type = cfg.control_type
    taskname = cfg.task.taskname
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    set_up_plotting_styles (cfg.paths.style_path)
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
    task.seed = seed # for consistency

    data_dict = pickle.load(open(os.path.join(aux_datasets_folder, f"selectivities_{dataSegment}{n_nets}_{control_type}.pkl"), "rb+"))
    legends = data_dict["legends"]
    inds_list = data_dict["inds_list"]

    # plot individual selectivities:
    # for k, legend in enumerate(legends):
    #     if not "shuffled=True" in legend:
    #         RNN_selectivities_processed = data_dict["RNN_selectivities_processed"]
    #         for i in range(n_nets_to_plot):
    #             for axes in [[0, 1, 2], [0, 2, 3], [1, 2, 3]]:
    #                 path = os.path.join(img_save_folder, f"{taskname}_selectivities_{legend}_{axes}_{i}.pdf")
    #                 selectivities_projected = RNN_selectivities_processed[inds_list[k][i]]
    #                 plot_selectivities(selectivities_projected,
    #                                    axes=axes[:2],
    #                                    show=False,
    #                                    save=save,
    #                                    path=path,
    #                                    n_dim=2)

    # plot aggregated selectivities:
    selectivities_aggregated_dict = {}
    for k, legend in enumerate(legends):
        if not "shuffled=True" in legend:
            RNN_selectivities_processed = data_dict["RNN_selectivities_processed"]
            selectivities_list = [RNN_selectivities_processed[inds_list[k][i]][:, :n_dims_to_take] for i in range(n_nets_to_aggregate)]
            #need to register selectivities together:
            template_configuration = selectivities_list[1]
            Qs = []
            for i in range(len(selectivities_list)):
                Q, score = ICP_registration(points_source=selectivities_list[i],
                                            points_target=template_configuration,
                                            max_iter=1000, tol=1e-10, seed=seed + i)
                Qs.append(np.copy(Q))
            selectivities_list_aligned = [selectivities_list[i] @ Qs[i] for i in range(len(Qs))]
            aggregated_selectivities = np.vstack(selectivities_list_aligned)
            # leave only top contributing neurons:
            row_norms = np.linalg.norm(aggregated_selectivities, axis=1)
            threshold = np.quantile(row_norms, 0.5)
            mask = row_norms > threshold
            aggregated_selectivities = aggregated_selectivities[mask]
            pca = PCA(n_components=aggregated_selectivities.shape[1])
            aggregated_selectivities = pca.fit_transform(aggregated_selectivities)
            selectivities_aggregated_dict[legend] = aggregated_selectivities

    for k, legend in enumerate(legends):
        if not "shuffled=True" in legend:
            print(legend)
            Q = np.eye(selectivities_aggregated_dict[legend].shape[1])
            if "Dale=False" in legend:
                template_legend = re.sub(r'Dale=False', 'Dale=True', legend)
                Q, score = ICP_registration(points_source=selectivities_aggregated_dict[legend],
                                            points_target=selectivities_aggregated_dict[template_legend],
                                            max_iter=1000, tol=1e-10, n_tries=100, seed=seed)
            aggregated_selectivities = selectivities_aggregated_dict[legend] @ Q
            for axes in [[0, 1], [0, 2], [1, 2]]:
                path = os.path.join(img_save_folder, f"{taskname}_selectivitiesAgg_{legend}_{axes}.pdf")
                plot_selectivities(aggregated_selectivities,
                                   axes=axes,
                                   show=show,
                                   save=save,
                                   path=path,
                                   n_dim=2)

                # path = os.path.join(img_save_folder, f"{taskname}_selectivitiesAgg_{legend}_{axes}_3D.pdf")
                # plot_selectivities(aggregated_selectivities,
                #                    axes=axes,
                #                    show=show,
                #                    save=save,
                #                    path=path,
                #                    n_dim=3)
    return None

if __name__ == '__main__':
    plot_individual_selectivities()