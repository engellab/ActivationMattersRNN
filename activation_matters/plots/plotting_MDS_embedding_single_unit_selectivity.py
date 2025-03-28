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
def plot_selectivities(cfg):
    show = False
    save = True
    feature_type = "selectivities"
    set_up_plotting_styles(cfg.paths.style_path)
    taskname = cfg.task.taskname
    control_type = cfg.control_type
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

    data_dict = pickle.load(open(os.path.join(aux_datasets_folder, f"{feature_type}_{dataSegment}{n_nets}_{control_type}.pkl"), "rb+"))
    legends = data_dict["legends"]
    inds_list = data_dict["inds_list"]
    # RNN_selectivities_processed = data_dict["RNN_selectivities_processed"]

    # VISUALIZE MDS EMBEDDING OF SELECTIVITIES
    Mat = pickle.load(open(os.path.join(aux_datasets_folder, f"{feature_type}_similarity_matrix_{dataSegment}{n_nets}_{control_type}.pkl"), "rb"))
    path = os.path.join(img_save_folder, f"{feature_type}_similarity_matrix_{dataSegment}{n_nets}_{control_type}.pdf")
    if show:
        plot_similarity_matrix(Mat, save=save, path=path)
    print(f"Computing the {feature_type} embedding")
    colors = ["red", "red",
              "orange", "orange",
              "blue", "blue",
              "deepskyblue", "deepskyblue",
              "green", "green",
              "lightgreen", "lightgreen"]
    hatch = ["", "", "", "", "", "", "", "", "", "", "", ""]
    markers = ["o", "v", "o", "v", "o", "v", "o", "v", "o", "v", "o", "v"]

    np.fill_diagonal(Mat, 0)

    img_name = f"MDS_{feature_type}_attempt=XXX_{dataSegment}{n_nets}_{control_type}.pdf"
    ray.init()
    # Launch tasks in parallel
    results = [
        run_mds_and_plot.remote(cfg, attempt, img_name, Mat, img_save_folder, inds_list,
                                legends, colors, hatch, markers, save, show)
        for attempt in range(3)]
    # Retrieve results (if needed)
    embeddings = ray.get(results)
    ray.shutdown()


if __name__ == '__main__':
    plot_selectivities()