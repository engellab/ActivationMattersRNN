import os
from scipy.linalg import orthogonal_procrustes
from scipy.stats import ortho_group
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from copy import copy
import pickle
from activation_matters.analysis.fixed_point_analysis import ICP_registration, get_fp_data_dict
from activation_matters.plots.ploting_utils import normalize_color, plot_embedding, plot_aligned_FPs, plot_similarity_matrix, \
    plot_fixed_points
import numpy as np
import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import ray
import re
from activation_matters.plots.style.style_setup import set_up_plotting_styles


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
    os.environ["NUMEXPR_MAX_THREADS"] = "50"
    n_dim = 2
    show = False
    save = True
    taskname = cfg.task.taskname
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    img_save_folder = os.path.join(cfg.paths.img_folder, taskname)
    set_up_plotting_styles(cfg.paths.style_path)
    path_to_folder = os.path.join(cfg.paths.fixed_points_data_folder, taskname)
    aux_datasets_folder = os.path.join(cfg.paths.auxilliary_datasets_path, taskname)
    n_components = cfg.task.dynamical_topology_analysis.n_PCs
    control_type = cfg.control_type

    fp_dict_combined = pickle.load(open(os.path.join(aux_datasets_folder, f"FPs_{dataSegment}{n_nets}_{control_type}.pkl"), "rb+"))
    inds_list = fp_dict_combined["inds_list"]
    legends = fp_dict_combined["legends"]

    Mat = pickle.load(open(os.path.join(aux_datasets_folder, f"FPs_similarity_matrix_{dataSegment}{n_nets}_{control_type}.pkl"), "rb"))
    path = os.path.join(img_save_folder, f"fp_struct_similarity_matrix_{dataSegment}{n_nets}_{control_type}.pdf")
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

    img_name = f"MDS_fp_attempt=XXX_nPCs={n_components}_{dataSegment}{n_nets}_{control_type}.pdf"
    ray.init(ignore_reinit_error=True, address="auto")
    print(ray.available_resources())
    # Launch tasks in parallel
    results = [
        run_mds_and_plot.remote(cfg, attempt, img_name, Mat, img_save_folder,
                                inds_list, legends, colors, hatch, markers, save, show)
        for attempt in range(3)]
    embeddings = ray.get(results)
    ray.shutdown()
    # embeddings = run_mds_and_plot(cfg, 0, img_name, Mat, img_save_folder, inds_list, legends, colors, hatch, markers, save, show)
    return None


if __name__ == '__main__':
    plot_fixed_point_configurations()