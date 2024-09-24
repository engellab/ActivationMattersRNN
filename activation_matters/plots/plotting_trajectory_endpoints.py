import numpy as np
import os
from trainRNNbrain.training.training_utils import prepare_task_arguments
from sklearn.manifold import MDS
from activation_matters.plots.ploting_utils import interpolate_color, plot_similarity_matrix, plot_embedding, \
    plot_stimuli_representations
import hydra
from omegaconf import OmegaConf
from style.style_setup import set_up_plotting_styles
OmegaConf.register_new_resolver("eval", eval)
import pickle

taskname = "CDDM"
show = True
save = False
n_nets = 10
n_dim = 2

@hydra.main(version_base="1.3", config_path=f"../../configs/task/", config_name=f'{taskname}')
def visualize_stimreps(cfg: OmegaConf):
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

    data_dict = pickle.load(open(os.path.join(aux_datasets_folder, "stimrep_similarity.pkl"), "rb+"))
    legends = data_dict["legends"]
    face_colors = data_dict["face_colors"]
    edge_colors = data_dict["edge_colors"]
    markers =  data_dict["markers"]
    inds_list = data_dict["inds_list"]
    RNN_stimreps_list = data_dict["PCA_stimuli_list"]
    # plotting the trajectories
    for k, legend in enumerate(legends):
        # if "shuffle=False" in legend:
        for axes in [[0, 1, 2], [0, 2, 3], [1, 2, 3]]:
            for i in range(n_nets):
                stimuli_representations = RNN_stimreps_list[inds_list[k][i]]
                path = os.path.join(img_folder, f"{taskname}_stimuli_representations_{legend}_{axes}_{i}.pdf")
                plot_stimuli_representations(PCA_stimuli=stimuli_representations,
                                             face_colors=face_colors,
                                             edge_colors=edge_colors,
                                             markers=markers,
                                             s=70,
                                             show=False,
                                             save=save,
                                             path=path, n_dim=n_dim)

    # VISUALIZE MDS EMBEDDING OF TRAJECTORIES
    Mat = pickle.load(open(os.path.join(aux_datasets_folder, "stimrep_similarity_matrix.pkl"), "rb"))
    path = os.path.join(img_folder, f"stimrep_similarity_matrix.pdf")
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
        path = os.path.join(img_folder, f"MDS_stimreps_attempt={attempt}.pdf")
        plot_embedding(embedding, inds_list, legends, colors, hatch, markers,
                       show_legends=False, save=save, path=path, show=show)


if __name__ == '__main__':
    visualize_stimreps()