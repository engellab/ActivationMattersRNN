import numpy as np
from matplotlib import pyplot as plt
import os

path = "/Users/tolmach/Documents/GitHub/latent_circuit_inference/data/inferred_LCs"
taskname = "CDDM"
score_dict = {}

for af in ["relu", "sigmoid", "tanh"]:
    score_dict[af] = {}
    for constrained in [True, False]:
        score_dict[af][f"constrained_{constrained}"] = []
        sfolder = f"{taskname}_{af}_constrained={constrained}"
        ssfolders = os.listdir(os.path.join(path, sfolder))
        for ssfolder in ssfolders:
            if ssfolder == ".DS_Store":
                pass
            else:
                sssfolders = os.listdir(os.path.join(path, sfolder, ssfolder))
                sssfolders = [sssfolder for sssfolder in sssfolders if sssfolder != ".DS_Store"]
                scores = [float(sssfolder.split("_")[0]) for sssfolder in sssfolders]
                score_dict[af][f"constrained_{constrained}"].append(max(scores))

for af in ["relu", "sigmoid", "tanh"]:
    for constrained in [True, False]:
        scores = score_dict[af][f"constrained_{constrained}"]
        print(f"{af}, constrained={constrained}, variance explained: {np.round(np.mean(scores),2)} +- {np.round(np.std(scores),2)}")
        # print([np.round(score, 2) for score in scores])