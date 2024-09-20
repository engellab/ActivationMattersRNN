import numpy as np
import os
from trainRNNbrain.training.training_utils import prepare_task_arguments
from sklearn.manifold import MDS
from activation_matters.plots.ploting_utils import interpolate_color, plot_similarity_matrix, plot_embedding, plot_representations
import hydra
from omegaconf import OmegaConf
from style.style_setup import set_up_plotting_styles
OmegaConf.register_new_resolver("eval", eval)
import pickle


taskname = "MemoryNumber"
show = False
save = True
n_nets = 5
q = 0.4

@hydra.main(version_base="1.3", config_path=f"../../configs/task/", config_name=f'{taskname}')
def visualize_neurreps(cfg: OmegaConf):
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

    data_dict = pickle.load(open(os.path.join(aux_datasets_folder, "neurrep_similarity.pkl"), "rb+"))
    legends = data_dict["legends"]
    inds_list = data_dict["inds_list"]
    RNN_neurreps_list = data_dict["RNN_neurreps_list"]

    # # plotting the trajectories
    # for k, legend in enumerate(legends):
    #     if "shuffle=False" in legend:
    #         for axes in [[0, 1, 2], [0, 2, 3], [1, 2, 3]]:
    #             neural_representations = np.concatenate([RNN_neurreps_list[inds_list[k][i]] for i in range(n_nets)], axis = 0)
    #
    #             path = os.path.join(img_folder, f"{taskname}_neural_representations_{legend}_{axes}.pdf")
    #             n_dim = 2
    #
    #             thr = np.quantile(neural_representations, q)
    #             inds = np.where(np.abs(neural_representations) > thr)[0]
    #             neural_representations_reduced = neural_representations[inds, :]
    #
    #             plot_representations(neural_representations_reduced,
    #                                  axes=axes[:n_dim],
    #                                  labels=None,
    #                                  show=False,
    #                                  save=save, path=path,
    #                                  s=30, alpha=0.9, n_dim=n_dim)

    # VISUALIZE MDS EMBEDDING OF TRAJECTORIES
    Mat = pickle.load(open(os.path.join(aux_datasets_folder, "neurreps_similarity_matrix.pkl"), "rb"))
    path = os.path.join(img_folder, f"neurreps_similarity_matrix.pdf")
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
        path = os.path.join(img_folder, f"MDS_neurreps_attempt={attempt}.pdf")
        plot_embedding(embedding, inds_list, legends, colors, hatch, markers,
                       show_legends=False, save=save, path=path, show=show)


if __name__ == '__main__':
    visualize_neurreps()