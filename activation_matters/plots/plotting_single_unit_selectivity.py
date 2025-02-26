import numpy as np
import os
from trainRNNbrain.training.training_utils import prepare_task_arguments
from sklearn.manifold import MDS
from activation_matters.plots.ploting_utils import interpolate_color, plot_similarity_matrix, plot_embedding, plot_representations
import hydra
from omegaconf import OmegaConf
from style.style_setup import set_up_plotting_styles
import pickle
import ray
import re

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


@hydra.main(version_base="1.3", config_path=f"../../configs/", config_name=f'base')
def plot_tuning(cfg: OmegaConf):
    show = False
    save = True
    # q = 0.4
    set_up_plotting_styles(cfg.paths.style_path)
    taskname = cfg.task.taskname
    aux_datasets_folder = os.path.join(cfg.paths.auxilliary_datasets_path, taskname)
    img_save_folder = os.path.join(cfg.paths.img_folder, taskname)
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    if taskname == "CDDM":
        task.coherences = np.array([-1, -0.5, 0, 0.5, 1.0])
    if hasattr(task, 'random_window'):
        task.random_window = 0
    task.seed = 0 # for consistency
    inputs, targets, conditions = task.get_batch()

    data_dict = pickle.load(open(os.path.join(aux_datasets_folder, f"tunings_{dataSegment}{n_nets}.pkl"), "rb+"))
    legends = data_dict["legends"]
    inds_list = data_dict["inds_list"]
    RNN_tunings_list = data_dict["RNN_tunings_list"]

    # # plotting the trajectories
    # for k, legend in enumerate(legends):
    #     if "shuffle=False" in legend:
    #         for axes in [[0, 1, 2], [0, 2, 3], [1, 2, 3]]:
    #             neural_tunings = np.concatenate([RNN_tunings_list[inds_list[k][i]] for i in range(n_nets)], axis = 0)
    #
    #             path = os.path.join(img_folder, f"{taskname}_neural_representations_{legend}_{axes}.pdf")
    #             n_dim = 2
    #
    #             thr = np.quantile(neural_tunings, q)
    #             inds = np.where(np.abs(neural_tunings) > thr)[0]
    #             neural_tunings_reduced = neural_tunings[inds, :]
    #
    #             plot_representations(neural_tunings_reduced,
    #                                  axes=axes[:n_dim],
    #                                  labels=None,
    #                                  show=False,
    #                                  save=save, path=path,
    #                                  s=30, alpha=0.9, n_dim=n_dim)

    # VISUALIZE MDS EMBEDDING OF TRAJECTORIES
    Mat = pickle.load(open(os.path.join(aux_datasets_folder, f"tunings_similarity_matrix_{dataSegment}{n_nets}.pkl"), "rb"))
    path = os.path.join(img_save_folder, f"tunings_similarity_matrix_{dataSegment}{n_nets}.pdf")
    if show:
        plot_similarity_matrix(Mat, save=save, path=path)
    print("Computing the tuning embedding")
    colors = ["red", "red",
              "orange", "orange",
              "blue", "blue",
              "deepskyblue", "deepskyblue",
              "green", "green",
              "lightgreen", "lightgreen"]
    hatch = ["", "", "", "", "", "", "", "", "", "", "", ""]
    markers = ["o", "v", "o", "v", "o", "v", "o", "v", "o", "v", "o", "v"]

    np.fill_diagonal(Mat, 0)

    img_name = f"MDS_tunings_attempt=XXX_{dataSegment}{n_nets}.pdf"
    ray.init()
    # Launch tasks in parallel
    results = [
        run_mds_and_plot.remote(cfg, attempt, img_name, Mat, img_save_folder, inds_list,
                                legends, colors, hatch, markers, save, show)
        for attempt in range(3)]
    # Retrieve results (if needed)
    embeddings = ray.get(results)
    ray.shutdown()

    # for attempt in range(3):
    #     mds = MDS(n_components=2, dissimilarity='precomputed', n_init=101, eps=1e-6, max_iter=1000)
    #     mds.fit(Mat)
    #     embedding = mds.embedding_
    #     path = os.path.join(img_folder, f"MDS_tuning_attempt={attempt}_{dataSegment}{n_nets}.pdf")
    #     plot_embedding(embedding, inds_list, legends, colors, hatch, markers,
    #                    show_legends=False, save=save, path=path, show=show)


if __name__ == '__main__':
    plot_tuning()