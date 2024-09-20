import numpy as np
np.set_printoptions(suppress=True)
from matplotlib import pyplot as plt
import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from copy import deepcopy
from hdbscan import HDBSCAN
from sklearn.mixture import GaussianMixture as GM
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import silhouette_score
os.system('python ../../style/style_setup.py')

def mean_downsample(array, window):
    k = array.shape[1]
    n_chunks = int(k // window)
    downsampled_array = np.hstack([np.mean(array[:, i*window: np.minimum(k, (i+1) * window)], axis = 1).reshape(-1, 1) for i in range(n_chunks)])
    return downsampled_array

def permute_input_matrix(mat, order):
    new_mat = np.empty_like(mat)
    for i, r in enumerate(order):
        new_mat[i, :] = mat[r, :]
    return new_mat

def permute_output_matrix(mat, order):
    new_mat = np.empty_like(mat)
    for i, c in enumerate(order):
        new_mat[:, i] = mat[:, c]
    return new_mat

def permute_recurrent_matrix(mat, order):
    new_mat = permute_input_matrix(mat, order)
    new_mat = permute_output_matrix(new_mat, order)
    return new_mat

def bitstr(row: np.ndarray, th) -> str:
    return ''.join((row > th).astype(int).astype(str))

def get_ordering(W_inp, th=0.0):
    bitstrings = [(bitstr(W_inp[idx, :], th), idx) for idx in range(W_inp.shape[0])]
    bitstrings = sorted(bitstrings, key=lambda x: x[0], reverse=True)
    idxs = [idx for _, idx in bitstrings]
    return idxs

def gini(v, n_points = 1000):
    """Compute Gini coefficient of array of values"""
    v_abs =np.sort(np.abs(v))
    cumsum_v=np.cumsum(v_abs)
    n = len(v_abs)
    vals = np.concatenate([[0], cumsum_v/cumsum_v[-1]])
    dx = 1/n
    x = np.linspace(0, 1, n+1)
    f = interp1d(x=x, y=vals, kind='previous')
    xnew = np.linspace(0, 1, n_points+1)
    dx_new = 1/(n_points)
    vals_new = f(xnew)
    return 1 - 2 * np.trapz(y=vals_new, x=xnew, dx=dx_new)

def sparsity(M, method="gini"):
    a = []
    for i in range(M.shape[0]):
        a.append(eval(f"{method}")(np.abs(M[i, :])))
    return a

def find_n_clusters(data, n_reps, min_clusters, max_clusters, clusterer = 'hdbscan'):
    if clusterer == 'gm':
        res = np.zeros((max_clusters - min_clusters + 1, n_reps))
        label_dict = {}
        for n_clusters in tqdm(np.arange(min_clusters, max_clusters)):
            for i in np.arange(n_reps):
                cl = GM(n_components=n_clusters,
                        max_iter=1000,
                        tol=1e-12, init_params='k-means++')
                labels = cl.fit_predict(data)
                res[n_clusters - min_clusters, i] = silhouette_score(data, labels)
                label_dict[(n_clusters, i)] = deepcopy(labels)
        scores = np.mean(res, axis=1)

        n_clusters = np.argmax(scores) + min_clusters
        i = np.argmax(res[np.argmax(scores), :])
        best_labels = label_dict[(n_clusters, i)]
    if clusterer == 'hdbscan':
        cl = HDBSCAN(algorithm='best',cluster_selection_epsilon=0.1, alpha=1.0, approx_min_span_tree=True,
        gen_min_span_tree=True, leaf_size=40,
        metric='euclidean', min_cluster_size=20, min_samples=None, p=None)
        best_labels = cl.fit_predict(data)
        n_clusters = len(np.unique(best_labels))
    return n_clusters, best_labels
