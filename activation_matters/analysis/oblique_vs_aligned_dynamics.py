import numpy as np
np.set_printoptions(suppress=True)
from trainRNNbrain.training.training_utils import *
import hydra
from activation_matters.utils.trajectories_utils import *
from itertools import chain
import ray
from copy import deepcopy
from tqdm.auto import tqdm
# OmegaConf.register_new_resolver("eval", eval)

def subspace_cosine_similarity(W, P):
    """
    Compute the cosine similarity measure between the subspaces spanned by the columns of W and P.

    Parameters:
    W (numpy.ndarray): A tall skinny matrix (n x k).
    P (numpy.ndarray): A tall skinny matrix (n x m).

    Returns:
    float: The cosine similarity measure between the subspaces (ranges from 0 to 1).
    """
    # Ensure that W and P are tall skinny matrices
    assert W.shape[0] > W.shape[1], "W should be a tall skinny matrix."
    assert P.shape[0] > P.shape[1], "P should be a tall skinny matrix."

    # Compute the QR decomposition of W and P to get orthonormal bases
    Q_W, _ = np.linalg.qr(W)
    Q_P, _ = np.linalg.qr(P)

    # Compute the SVD of the product of the orthonormal bases
    _, S, _ = np.linalg.svd(Q_W.T @ Q_P)

    # The cosine similarity is the largest singular value
    cosine_similarity = np.max(S)

    return cosine_similarity

show = False
save = True
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def analyze_angles_between_output_and_activity_subspaces(cfg):
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    taskname = cfg.task.taskname
    dataset_path = os.path.join(f"{cfg.paths.RNN_dataset_path}", f"{taskname}_{dataSegment}{n_nets}.pkl")
    aux_datasets_folder = os.path.join(f"{cfg.paths.auxilliary_datasets_path}", taskname)
    dataset = pickle.load(open(dataset_path, "rb"))

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    if taskname == "CDDM":
        task.coherences = np.array(list(cfg.task.trajectory_analysis_params.coherences))
        decision_epoch_mask = np.arange(task_conf["dec_on"], task_conf["dec_off"])
    elif taskname == 'GoNoGo':
        decision_epoch_mask = np.arange(task_conf["cue_on"], task_conf["cue_off"])
    elif taskname == 'MemoryNumber':
        decision_epoch_mask = np.arange(task_conf["recall_on"], task_conf["recall_off"])

    if hasattr(task, 'random_window'):
        task.random_window = 0  # eliminating any source of randomness while analysing the trajectories
    task.seed = 0  # for consistency

    for activation in dataset.keys():
        for constraint in dataset[activation].keys():
            W_out_list = dataset[activation][constraint].W_out_RNN.tolist()
            W_out_list = [W_out.T for W_out in W_out_list]
            PCs_list = []
            activation_slope = dataset[activation][constraint]["activation_slope"].tolist()[0]
            trajectories = get_trajectories(dataset=dataset[activation][constraint],
                                    task=task,
                                    activation_name=activation,
                                    activation_slope=activation_slope,
                                    get_batch_args={})

            # get the relevant decision epoch
            print("Projecting trajectories")
            for trajectory in tqdm(trajectories):
                trajectory_decision_epoch = trajectory[:, decision_epoch_mask, :]
                trajectory_DE_flat = trajectory_decision_epoch.reshape(trajectory.shape[0], -1)

                # do the PCA on these trajectories
                pca = PCA(n_components=3)
                pca.fit(trajectory_DE_flat.T)
                PCs = pca.components_.T # has to be N dimensional
                PCs_list.append(deepcopy(PCs))

            print("Computing cosine similarity between the subspaces")
            cosines = []
            angles = []
            for W, P in tqdm(zip(W_out_list, PCs_list)):
                # calculate an angle between the subspaces
                cosines.append(subspace_cosine_similarity(W, P))
                angles.append(360 * np.arccos(cosines[-1]) / (2 * np.pi))
            m_a = np.round(np.mean(angles), 1)
            std_a = np.round(np.std(angles), 1)
            print(f"Activation: {activation}; constraint: {constraint}; Angle between subspaces: {m_a} \pm {std_a}")
    return None


if __name__ == '__main__':
    analyze_angles_between_output_and_activity_subspaces()