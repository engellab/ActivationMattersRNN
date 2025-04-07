## Analysing dynamics, trajectories, neural tuning, and behavior of continuous-time RNNs with different architectural choices

This repository contains the code for running analysis on how the seemingly inconsequential RNN design-choices affect the dynamics, trajectories, neural tuning and, finally, behavior.

### Setting up config files

You would require to set the paths in the `congigs/task` files.
Make sure to modify the following, setting *absolute* paths:

- dataset_path (contains the top 50 RNNs and bottom 50 RNNs for each task, located in `../data/RNN_datasets`)
- auxilliary_datasets_path (`../data/auxilliary_datasets)`
- img_folder (`../img/$task_name)`
- fixed_points_data_folder (`../fixed_points/$task_name)`

### Dependencies
This package relies on a [`trainRNNbrain`](https://github.com/engellab/trainRNNbrain) and [`latent_circiut_inference`](https://github.com/engellab/latent_circuit_inference) packages, make sure to download it and install it locally by running `python -m pip install -e ./` from the `trainRNNbrain`  and `latent_circuit_inference' folder.

| Package Name                 | Version     |
|------------------------------|-------------|
| **latent-circuit-inference** | **0.1**     |
| **rnn-analysis**             | **1.0**     |
| **trainrnnbrain**            | **1.0**     |
| aiosignal                    | 1.3.1       |
| antlr4-python3-runtime       | 4.9.3       |
| anyio                        | 4.4.0       |
| appnope                      | 0.1.4       |
| argon2-cffi                  | 23.1.0      |
| argon2-cffi-bindings         | 21.2.0      |
| arrow                        | 1.3.0       |
| asttokens                    | 2.4.1       |
| async-lru                    | 2.0.4       |
| attrs                        | 23.2.0      |
| babel                        | 2.15.0      |
| backcall                     | 0.2.0       |
| beautifulsoup4               | 4.12.3      |
| bleach                       | 6.1.0       |
| certifi                      | 2024.2.2    |
| cffi                         | 1.16.0      |
| charset-normalizer           | 3.3.2       |
| click                        | 8.1.7       |
| comm                         | 0.2.2       |
| contourpy                    | 1.2.1       |
| cycler                       | 0.12.1      |
| cython                       | 0.29.37     |
| debugpy                      | 1.8.2       |
| decorator                    | 5.1.1       |
| defusedxml                   | 0.7.1       |
| docopt                       | 0.6.2       |
| exceptiongroup               | 1.2.1       |
| executing                    | 2.0.1       |
| fastjsonschema               | 2.20.0      |
| filelock                     | 3.13.4      |
| fonttools                    | 4.51.0      |
| fqdn                         | 1.5.1       |
| frozenlist                   | 1.4.1       |
| fsspec                       | 2024.3.1    |
| h11                          | 0.14.0      |
| hdbscan                      | 0.8.33      |
| httpcore                     | 1.0.5       |
| httpx                        | 0.27.0      |
| hydra-core                   | 1.3.2       |
| idna                         | 3.7         |
| ipykernel                    | 6.29.4      |
| ipython                      | 8.12.3      |
| ipywidgets                   | 8.1.3       |
| isoduration                  | 20.11.0     |
| jedi                         | 0.19.1      |
| jinja2                       | 3.1.3       |
| joblib                       | 1.4.0       |
| json5                        | 0.9.25      |
| jsonpointer                  | 3.0.0       |
| jsonschema                   | 4.21.1      |
| jsonschema-specifications    | 2023.12.1   |
| jupyter                      | 1.0.0       |
| jupyter-client               | 8.6.2       |
| jupyter-console              | 6.6.3       |
| jupyter-core                 | 5.7.2       |
| jupyter-events               | 0.10.0      |
| jupyter-lsp                  | 2.2.5       |
| jupyter-server               | 2.14.1      |
| jupyter-server-terminals     | 0.5.3       |
| jupyterlab                   | 4.2.5       |
| jupyterlab-pygments          | 0.3.0       |
| jupyterlab-server            | 2.27.2      |
| jupyterlab-widgets           | 3.0.11      |
| kiwisolver                   | 1.4.5       |
| kneed                        | 0.8.5       |
| kooplearn                    | 1.1.3       |
| llvmlite                     | 0.42.0      |
| markupsafe                   | 2.1.5       |
| matplotlib                   | 3.4.3       |
| matplotlib-inline            | 0.1.7       |
| mistune                      | 3.0.2       |
| mpmath                       | 1.3.0       |
| msgpack                      | 1.0.8       |
| nbclient                     | 0.10.0      |
| nbconvert                    | 7.16.4      |
| nbformat                     | 5.10.4      |
| nest-asyncio                 | 1.6.0       |
| networkx                     | 3.3         |
| notebook                     | 7.2.1       |
| notebook-shim                | 0.2.4       |
| numba                        | 0.59.1      |
| numdifftools                 | 0.9.41      |
| numpy                        | 1.26.4      |
| omegaconf                    | 2.3.0       |
| overrides                    | 7.7.0       |
| packaging                    | 24.2        |
| pandas                       | 2.2.2       |
| pandocfilters                | 1.5.1       |
| parso                        | 0.8.4       |
| pexpect                      | 4.9.0       |
| pickleshare                  | 0.7.5       |
| pillow                       | 10.3.0      |
| pip                          | 25.0.1      |
| pipdeptree                   | 2.26.0      |
| pipreqs                      | 0.5.0       |
| platformdirs                 | 4.2.2       |
| pot                          | 0.9.3       |
| prometheus-client            | 0.20.0      |
| prompt-toolkit               | 3.0.47      |
| proplot                      | 0.9.7       |
| protobuf                     | 5.26.1      |
| psutil                       | 6.0.0       |
| ptyprocess                   | 0.7.0       |
| pure-eval                    | 0.2.2       |
| pycparser                    | 2.22        |
| pygments                     | 2.18.0      |
| pynndescent                  | 0.5.12      |
| pyparsing                    | 3.1.2       |
| python-dateutil              | 2.9.0.post0 |
| python-json-logger           | 2.0.7       |
| pytz                         | 2024.1      |
| pyyaml                       | 6.0.1       |
| pyzmq                        | 26.0.3      |
| qtconsole                    | 5.5.2       |
| qtpy                         | 2.4.1       |
| ray                          | 2.12.0      |
| referencing                  | 0.35.0      |
| requests                     | 2.31.0      |
| rfc3339-validator            | 0.1.4       |
| rfc3986-validator            | 0.1.1       |
| rpds-py                      | 0.18.0      |
| scikit-learn                 | 1.4.2       |
| scipy                        | 1.13.0      |
| seaborn                      | 0.13.2      |
| send2trash                   | 1.8.3       |
| six                          | 1.16.0      |
| sniffio                      | 1.3.1       |
| soupsieve                    | 2.5         |
| stack-data                   | 0.6.3       |
| sympy                        | 1.12        |
| terminado                    | 0.18.1      |
| threadpoolctl                | 3.4.0       |
| tinycss2                     | 1.3.0       |
| tomli                        | 2.0.1       |
| torch                        | 2.3.0       |
| tornado                      | 6.4.1       |
| tqdm                         | 4.66.2      |
| traitlets                    | 5.14.3      |
| types-python-dateutil        | 2.9.0.20240316 |
| typing-extensions            | 4.11.0      |
| tzdata                       | 2024.1      |
| umap                         | 0.1.1       |
| umap-learn                   | 0.5.6       |
| uri-template                 | 1.3.0       |
| urllib3                      | 2.2.1       |
| wand                         | 0.6.13      |
| wcwidth                      | 0.2.13      |
| webcolors                    | 24.6.0      |
| webencodings                 | 0.5.1       |
| websocket-client             | 1.8.0       |
| widgetsnbextension           | 4.0.11      |
| yarg                         | 0.1.9       |




### How to run the analysis
Modify the config variables in config/paths/path.yaml file, so that the path-variables point towards the correct folders.

The dataset containing top 50 networks for each of the architectural class are stored in `data/RNN_datasets`.

- To analyze the (dis)similarity of RNNs in terms of **trajectories**, run `analysis/trajectory_analysis.py`.
The code generates the trajectories from the networks (as well as the networks with shuffled connectivity, used as control), projects them with PCA, puts all trajectories to the same scale. Further, it computes the pairwise distances between the sets of trajectories and then produces MDS embedding. You would need to plot the trajectories and MDS embedding with `plots/plotting_MDS_embedding_trajectories.py`

- To analyze the (dis)similarity of RNNs in terms of their **single-unit selectivity** configurations, run `analysis/selectivity_analysis.py`.
The code generates the trajectories from the networks (as well as the networks with shuffled connectivity, used as control), reduces the dimensionality with PCA down to (N, n_PCs), where N - number of neurons in the RNN, puts all configurations to the same scale. It then computes the pairwise distances between the sets of single-unit selectivity and produces MDS embedding.
To plot single-unit selectivity, run `plots/plotting_MDS_embedding_single_unit_selectivity.py`

- For the differences in RNNs in terms of their **fixed point configurations** run `analysis/computing_fp.py` first (it computes the fixed points, and it takes awhile), followed by running `analysis/fixed_point_analysis.py`, which then compares the fixed point configurations, computing the pair-wise distances.
To plot MDS embedding of fixed points, run `plots/plotting_MDS_embedding_fixed_point_configurations.py`

- To analyze **trial endpoint configurations**, run `analysis/trajectory_endpoints_analysis.py`, and then plot it with `plots/plotting_MDS_embedding_trajectory_endpoints.py`

- Finally, to analyze the discrepancies in the behavior of RNNs with different activations functions on the context-dependent decision-making task, run `analysis/behavior_analysis_CDDM.py`

As a result the code produces multiple figures (trajectories, single-unit selectivity, fixed point configurations, and trial end point configurations for each RNN, and MDS embeddings for each aspect) in the corresponding `img/taskname` folder.

For convenience, if you want to run the whole analysis pipeline, run `analysis/run_entire_pipeline.py`.

Be advised, the computations take **some time**.
Computing the distance matrix for ensuing MDS embedding of fixed points, for instance, took about 10 hours on a M3 chip Mac 2024 laptop just for one task. 
I run the whole analysis pipeline on DELLA Princeton cluster, parallelizing the computations over 50 cores. 
Computations took about one day.
I am sure there exist a room for optimization.







 




