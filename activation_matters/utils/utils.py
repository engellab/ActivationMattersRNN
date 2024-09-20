import os
import pickle
from copy import deepcopy
from scipy.interpolate import interp1d
from pathlib import Path
import numpy as np

def get_project_root():
    return Path(__file__).parent.parent

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
    # l = W_inp.shape[1]
    # N = W_inp.shape[0]
    # bitstrings = [(sum([W_inp[idx, i] * (W_inp[idx, i] > th) * 10**(l - i - 1) for i in range(l)]), idx) for idx in range(N)]
    bitstrings = [(bitstr(W_inp[idx, :], th), idx) for idx in range(W_inp.shape[0])]
    bitstrings = sorted(bitstrings, key=lambda x: x[0], reverse=True)
    idxs = [idx for _, idx in bitstrings]
    return idxs

# def get_ordering(W_inp, W_out, th=0.0):
#     l = W_inp.shape[1]
#     k = W_out.shape[1]
#     N = W_inp.shape[0]
#
#     bitstrings = [(
#         sum([W_inp[idx, i] * (W_inp[idx, i] > th) * 10**(l - i - 1) for i in range(l)]) +
#         sum([W_out[idx, i] * (W_out[idx, i] > th) * 5**(k - i - 1) for i in range(k)]),
#         idx
#     ) for idx in range(N)]
#     # bitstrings = [(bitstr(W_inp[idx, :], th), idx) for idx in range(W_inp.shape[0])]
#     bitstrings = sorted(bitstrings, key=lambda x: x[0], reverse=True)
#     idxs = [idx for _, idx in bitstrings]
#     return idxs

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


