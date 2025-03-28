import numpy as np

np.set_printoptions(suppress=True)
import sys
sys.path.append("/")
from activation_matters.utils import cca_core as cca_core
from scipy.sparse.linalg import lsqr
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.rnns import RNN_torch
from scipy.linalg import orthogonal_procrustes
# from procrustes import generic
from activation_matters.utils.feautre_extraction_utils import *
from tqdm.auto import tqdm


def get_features(trajectory, ts, var_threshold=0.99):
    traj_subsampled_list = []
    for i in range(len(ts) - 1):
        traj_subsampled_list.append(np.mean(trajectory[:, ts[i]:ts[i+1], :], axis = 1))
    traj_subsampled = np.hstack(traj_subsampled_list)
    F = traj_subsampled.T
    pca = PCA(n_components=np.minimum(trajectory.shape[0], F.shape[0]))
    pca.fit(F)
    n = np.where(np.cumsum(pca.explained_variance_ratio_) > var_threshold)[0][0] + 1
    F_projected = pca.components_[:n, :] @ F.T
    return F_projected

def shuffle_connectivity(W_inp, W_rec, W_out):

    N = W_inp.shape[0]
    n_inputs = W_inp.shape[1]
    n_outputs = W_out.shape[0]

    W_inp_sh = np.copy(W_inp)
    W_rec_sh = np.copy(W_rec)
    W_out_sh = np.copy(W_out)

    for c in range(N): # go column by colum
        inds = np.concatenate([np.arange(c), np.arange(c + 1, N)])
        W_rec_sh[inds, c] = np.random.choice(W_rec[inds, c], size=N - 1, replace=False)
        W_inp_sh[c, :] = np.random.choice(W_inp[c, :], size=(n_inputs,), replace=False)
        W_out_sh = np.random.choice(W_out[:, c], size=(n_outputs,), replace=False)
    return W_inp_sh, W_rec_sh, W_out_sh

def get_trajectories(dataset, task,
                     activation_name,
                     activation_slope,
                     get_batch_args,
                     dt=1, tau=10,
                     shuffled=False,
                     random=False,
                     constrained=False):
    inputs, targets, conditions = task.get_batch(**get_batch_args)
    trajectories_list = []
    W_inp_list = dataset["W_inp_RNN"] if type(dataset["W_inp_RNN"]) == list else dataset["W_inp_RNN"].tolist()
    W_rec_list = dataset["W_rec_RNN"] if type(dataset["W_rec_RNN"]) == list else dataset["W_rec_RNN"].tolist()
    W_out_list = dataset["W_out_RNN"] if type(dataset["W_out_RNN"]) == list else dataset["W_out_RNN"].tolist()

    for i in (range(len(W_inp_list))):
        N = W_inp_list[i].shape[0]

        W_inp = W_inp_list[i]
        W_rec = W_rec_list[i]
        W_out = W_out_list[i]
        if shuffled:
            W_inp, W_rec, W_out = shuffle_connectivity(W_inp, W_rec, W_out)
        if random:
            if constrained:
                W_rec, W_inp, W_out, _, _, _, _ = RNN_torch.get_connectivity_Dale(N=N,
                                                                                  num_inputs=task.n_inputs,
                                                                                  num_outputs=task.n_outputs)
            else:
                W_rec, W_inp, W_out, _, _, _ = RNN_torch.get_connectivity(N=N,
                                                                          num_inputs=task.n_inputs,
                                                                          num_outputs=task.n_outputs)
            W_inp = W_inp.detach().numpy()
            W_rec = W_rec.detach().numpy()
            W_out = W_out.detach().numpy()
        net_params = {"N": N,
                      "dt": dt,
                      "tau": tau,
                      "activation_name": activation_name,
                      "activation_slope": activation_slope,
                      "W_inp": W_inp,
                      "W_rec": W_rec,
                      "W_out": W_out,
                      "bias_rec": None,
                      "y_init": np.zeros(N)}

        rnn = RNN_numpy(**net_params)
        rnn.y = np.zeros(N)
        rnn.run(inputs, sigma_rec=0, sigma_inp=0)
        trajectories = rnn.get_history()
        trajectories_list.append(np.copy(trajectories))
    return trajectories_list