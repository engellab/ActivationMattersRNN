import numpy as np
import os
from sklearn.manifold import MDS
from activation_matters.plots.ploting_utils import interpolate_color, plot_similarity_matrix, \
    plot_embedding
import hydra
from omegaconf import OmegaConf
from activation_matters.plots.style.style_setup import set_up_plotting_styles
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
    mds = MDS(n_components=2, dissimilarity='precomputed',
              n_init=101, eps=1e-6, max_iter=1000, random_state=attempt)
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
    taskname = cfg.task.taskname
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    set_up_plotting_styles(cfg.paths.style_path)
    aux_datasets_folder = os.path.join(cfg.paths.auxilliary_datasets_path, taskname)
    img_save_folder = os.path.join(cfg.paths.img_folder, taskname)
    if not cfg.paths.local:
        ray.init(ignore_reinit_error=True, address="auto")
    else:
        ray.init(ignore_reinit_error=True)

    for modality in ["trajectories", "selectivities", "trajectory_endpoints", "FPs"]:
        data_dict = pickle.load(open(os.path.join(aux_datasets_folder, f"{modality}_{dataSegment}{n_nets}_shuffled.pkl"), "rb+"))
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
        legends_sh, inds_list_sh, colors_sh, hatch_sh, markers_sh = zip(*[(l, i, c, h, m) for l, i, c, h, m in zip(legends, inds_list, colors, hatch, markers) if ("shuffled=True" in l) or ("control=True" in l)])
        inds_to_take = list(chain(*inds_list_sh))
        inds_list_new = [len(inds_list_sh[i]) for i in range(len(inds_list_sh))]
        inds_list_new = [np.arange(el) for el in inds_list_new]

        cnt = 0
        for i, inds in enumerate(inds_list_new):
            inds_list_new[i] += cnt
            cnt += len(inds_list_new[i])
        inds_list_sh = inds_list_new

        Mat = pickle.load(open(os.path.join(aux_datasets_folder, f"{modality}_similarity_matrix_{dataSegment}{n_nets}_shuffled.pkl"), "rb"))
        Mat_subsampled = Mat[np.ix_(inds_to_take, inds_to_take)]
        path = os.path.join(img_save_folder, f"{modality}_matrix_shuffled_connectivity_{dataSegment}{n_nets}_shuffled.pdf")
        if show:
            plot_similarity_matrix(Mat_subsampled, save=save, path=path)

        np.fill_diagonal(Mat_subsampled, 0)

        img_name = f"MDS_{modality}_attempt=XXX_{dataSegment}{n_nets}_shuffled_connectivity.pdf"
        print(ray.available_resources())
        # Launch tasks in parallel
        results = [
            run_mds_and_plot.remote(cfg, attempt, img_name, Mat_subsampled, img_save_folder, inds_list_sh,
                                    legends_sh, colors_sh, hatch_sh, markers_sh, save, show)
            for attempt in range(3)]
        # Retrieve results (if needed)
        embeddings = ray.get(results)
    ray.shutdown()
    return None

if __name__ == '__main__':
    plot_MDS_embedding_connectivity_shuffled_RNNs()