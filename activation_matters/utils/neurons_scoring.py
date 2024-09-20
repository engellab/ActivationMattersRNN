import numpy as np
from matplotlib import pyplot as plt
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.analyzers.PerformanceAnalyzer import PerformanceAnalyzer
from copy import deepcopy

def score_rnn(RNN, task, scoring_function, mask=None):
    pa = PerformanceAnalyzer(rnn_numpy=RNN, task=task)
    input_batch, target_batch, conditions = task.get_batch()
    return pa.get_validation_score(scoring_function,
                                   input_batch,
                                   target_batch,
                                   mask=mask,
                                   sigma_rec=0, sigma_inp=0)


def score_neurons(RNN, task, scoring_function, mask=None):
    # establish a baseline score
    baseline_score = score_rnn(RNN=RNN, task=task, scoring_function=scoring_function, mask=mask)
    W_rec = np.copy(RNN.W_rec)
    W_inp = np.copy(RNN.W_inp)
    W_out = np.copy(RNN.W_out)

    N = RNN.W_rec.shape[0]
    nrn_scores = np.zeros(N)
    for i in range(N):
        W_inp_ablated = np.vstack([W_inp[:i, :],  W_inp[i+1:, :]])
        W_rec_ablated = np.vstack([W_rec[:i, :],  W_rec[i+1:, :]])
        W_rec_ablated = np.hstack([W_rec_ablated[:, :i],  W_rec_ablated[:, i+1:]])
        W_out_ablated = np.hstack([W_out[:, :i],  W_out[:, i+1:]])

        net_params = {"N": N - 1,
                      "dt": RNN.dt,
                      "tau": RNN.tau,
                      "activation_name": RNN.activation_name,
                      "activation_slope": RNN.activation_slope,
                      "W_inp": W_inp_ablated,
                      "W_rec": W_rec_ablated,
                      "W_out": W_out_ablated,
                      "bias_rec": None,
                      "y_init": np.zeros(N - 1)}
        RNN_ablated = RNN_numpy(**net_params)
        nrn_scores[i] = score_rnn(RNN_ablated, task, scoring_function, mask)
    return  (nrn_scores - baseline_score)

