<div align="center">
    <a href="https://github.com/LukasMut/VICE/actions/workflows/tests.yml" rel="nofollow">
        <img src="https://github.com/LukasMut/VICE/actions/workflows/tests.yml/badge.svg" alt="Tests" />
    </a>
    <a href="https://codecov.io/gh/LukasMut/VICE" rel="nofollow">
        <img src="https://codecov.io/gh/LukasMut/VICE/branch/main/graph/badge.svg?token=gntaL1yrXI" alt="Code coverage" />
    </a>
    <a href="https://gist.github.com/cheerfulstoic/d107229326a01ff0f333a1d3476e068d" rel="nofollow">
        <img src="https://img.shields.io/badge/maintenance-yes-brightgreen.svg" alt="Maintenance" />
    </a>
    <a href="https://github.com/LukasMut/VICE/blob/main" rel="nofollow">
        <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg" alt="PyPI" />
    </a>
    <a href="https://github.com/LukasMut/VICE/blob/main/LICENSE" rel="nofollow">
        <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License" />
    </a>
    <a href="https://github.com/psf/black" rel="nofollow">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
    </a>
</div>

# VICE: Variational Interpretable Concept Embeddings


<!-- Citation -->
## :page_with_curl: Citation

If you use this GitHub repository (or any modules associated with it), we would appreciate to cite our [NeurIPS publication](https://proceedings.neurips.cc/papers/search?q=VICE) as follows:

```latex
@inproceedings{muttenthaler2022vice,
 author = {Muttenthaler, Lukas and Zheng, Charles Y and McClure, Patrick and Vandermeulen, Robert A and Hebart, Martin N and Pereira, Francisco},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {33661--33675},
 publisher = {Curran Associates, Inc.},
 title = {VICE: Variational Interpretable Concept Embeddings},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/da1a97b53eec1c763c6d06835538fe3e-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
```

<!-- Setting up your environment -->
### :computer: Setting up your environment

Before using `VICE`, we recommend to create a virtual environment (e.g., `vice`), including all dependencies, via `conda`

```bash
$ conda env create --prefix /path/to/conda/envs/vice --file envs/environment.yml
$ conda activate vice
```

or via `mamba` (this is a faster drop-in replacement for `conda`)

```bash
$ conda install mamba -n base -c conda-forge # install mamba into the base environment
$ mamba create -n vice # create an empty environment
$ mamba env update -n vice --file envs/environment.yml # update the empty environment with dependencies in environment.yml
$ conda activate vice
```


Alternatively, dependencies can be installed via `pip`

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Repository structure

```bash
root
├── envs
├── └── environment.yml
├── data
├── ├── __init__.py
├── ├── files/*tsv
├── └── triplet_dataset.py
├── optimization
├── ├── __init__.py
├── ├── priors.py
├── ├── triplet_loss.py
├── ├── trainer.py
├── └── Vice.py
├── embeddings
├── ├── things
├── ├── ├── final_embedding.npy
├── └── └── final_model.tar
├── .gitignore
├── DEMO.ipynb
├── get_embeddings.sh
├── create_things_splits.py
├── find_best_hypers.py
├── main_inference.py
├── main_optimization.py
├── main_robustness_eval.py
├── main_tripletize.py
├── partition_triplets.py
├── requirements.txt
├── utils.py
└── visualization.py
```

## VICE step-by-step

### VICE DEMO

We provide a `DEMO` Jupyter Notebook (`JN`) to guide users through each step of the `VICE` optimization. The `DEMO` file is meant to facilitate the process of using `VICE`. In the `DEMO.ipynb` one can easily examine whether `VICE` overfits the trainig data and behaves well with respect to the evolution of latent dimensions over time. Embeddings can be extracted and analyzed directly in the `JN`.

### VICE optimization

Explanation of arguments in `main_optimization.py`

```python
 
 main_optimization.py --task (str) \ # "odd-one-out" (3AFC; no anchor) or "target-matching" (2AFC; anchor) task
 --triplets_dir (str) \ # path/to/triplet/data
 --results_dir (str) \ # optional specification of results directory (if not provided will resort to ./results/modality/init_dim/optim/mixture/seed/spike/slab/pi)
 --plots_dir (str) \ # optional specification of directory for plots (if not provided will resort to ./plots/modality/init_dim/optim/mixture/seed/spike/slab/pi)
 --epochs (int) \ # maximum number of epochs to run VICE optimization
 --burnin (int) \ # minimum number of epochs to run VICE optimization (burnin period)
 --eta (float) \ # learning rate
 --init_dim (int) \ # initial dimensionality of the model's embedding space
 --batch_size (int) \ # mini-batch size
 --optim (str) \ # optimizer (e.g., 'adam', 'adamw', 'sgd')
 --mixture (str) \ # whether to use a mixture of Gaussians or Laplace distributions in the spike-and-slab prior (i.e., 'gaussian' or 'laplace')
 --mc_samples (int) \ # number of weight matrices used in Monte Carlo sampling (for computationaly efficiency, M is set to 1 during training but can be set to any number at inference time)
 --spike (float) \ # sigma of the spike distribution
 --slab (float) \ # sigma of the slab distribution
 --pi (float) \ # probability value that determines the relative weighting of the distributions; the closer this value is to 1, the higher the probability that weights are drawn from the spike distribution
 --k (int) \ # an embedding dimension is considered important (and won't be pruned) if the minimum number of objects with a non-zero weight is larger than k (we recommend to set this value to 5 or 10)
 --ws (int) \ # determines for how many epochs the number of latent dimensions (after pruning) is not allowed to vary (ws >> 100)
 --steps (int) \ # perform validation, save model parameters and create model and optimizer checkpoints every <steps> epochs
 --device (str) \ # cuda or cpu
 --num_threads (int) \ # number of threads used for intraop parallelism on CPU; use only if device is CPU (won't affect performance on GPU)
 --rnd_seed (int) \ # random seed for reproducibility
 --verbose (bool) \ # show print statements about model performance and evolution of latent dimensions during training (can be easily piped into log file)
 ```

#### Example call

```bash
$ python main_optimization.py --task odd-one-out \
--triplets_dir path/to/triplets \
--results_dir ./results \
--plots_dir ./plots \
--epochs 2000 \
--burnin 500 \
--eta 0.001  \
--init_dim 100  \ 
--batch_size 128  \
--k 5  \
--ws 200  \ 
--optim adam  \ 
--mixture gaussian \ 
--mc_samples 10 \
--spike 0.25 \
--slab 1.0 \
--pi 0.6  \
--steps 50  \
--device cpu  \
--num_threads 8 \
--rnd_seed 42 \
--verbose \
```

### NOTES:

1. Note that triplet data is expected to be in the format `N x 3`, where `N` = number of triplets (e.g., 100k) and 3 refers to the three objects in a triplet, where `col_0` = anchor, `col_1` = positive, `col_2` = odd-one-out/negative. Triplet data must be split into `train` and `test` splits, and named `train_90.txt` or `train_90.npy` and `test_10.txt` or `test_10.npy` respectively.

2. Every `--steps` epochs (i.e., `if (epoch + 1) % steps == 0`) a `model_epoch.tar` (including model and optimizer `state_dicts`) and a `results_epoch.json` (including train and validation cross-entropy errors) file are saved to disk. In addition, after convergence of VICE, a `pruned_params.npz` (compressed binary file) with keys `pruned_loc` and `pruned_scale`, including pruned VICE parameters, is saved to disk. Latent dimensions of the pruned parameter matrices are sorted according to their overall importance. See output folder structure below for where to find these files.</br>

```bash
root/results/modality/init_dim/optimizer/mixture/spike/slab/pi/seed
├── model
├── └── f'model_epoch{epoch+1:04d}.tar' if (epoch + 1) % steps == 0
├── 'parameters.npz'
├── 'pruned_params.npz'
└── f'results_{epoch+1:04d}.json' if (epoch + 1) % steps == 0
```

3. `train.py` (which is invoked by `main_optimization.py`) plots train and validation performances (to examine overfitting) against as well as negative log-likelihoods and KL-divergences (to evaluate contribution of the different loss terms) alongside each other. Evolution of (identified) latent dimensions over time is additionally plotted after convergence. See folder structure below for where to find plots after the optimization has finished.

```bash
root/plots/modality/init_dim/optimizer/mixture/spike/slab/pi/seed
├── 'single_model_performance_over_time.png'
├── 'llikelihood_and_complexity_over_time.png'
└── 'latent_dimensions_over_time.png'
```

### VICE evaluation

Explanation of arguments in `main_robustness_eval.py`

```python
 main_robustness_eval.py --task (str) \ # "odd-one-out" (3AFC; no anchor) or "target-matching" (2AFC; anchor) task
 --results_dir (str) \ # path/to/models
 --n_objects (int) \ # number of unique objects/items/stimuli in the dataset
 --init_dim (int) \  # latent space dimensionality with which VICE was initialized at run time
 --batch_size (int) \  # mini-batch size used during VICE training
 --thresh (float) \  # Pearson correlation value to threshold reproducibility of dimensions (e.g., 0.8)
 --optim (str) \ # optimizer that was used during training (e.g., 'adam', 'adamw', 'sgd')
 --mixture (str) \  # whether a Gaussian or Laplacian mixture was used in the spike-and-slab prior (i.e., 'gaussian' or 'laplace')
 --spike (float) \  # sigma of spike distribution
 --slab (float) \  # sigma of slab distribution
 --pi (float) \  # probability value that determines likelihood of samples from the spike
 --triplets_dir (str) \  # path/to/triplet/data
 --mc_samples (int) \ # number of weight matrices used in Monte Carlo sampling for evaluating models on validation set
 --device (str) \  # cpu or cuda
 --rnd_seed (int) \  # random seed
 ```

#### Example call

```bash
$ python main_robustness_eval.py --task odd-one-out \
--results_dir path/to/models \ 
--n_objects number/of/unique/objects (e.g., 1854) \
--init_dim 100 \
--batch_size 128 \
--thresh 0.8 \
--optim adam \
--mixture gaussian \
--spike 0.25 \
--slab 1.0 \
--pi 0.6 \
--triplets_dir path/to/triplets \
--mc_samples 5 \
--device cpu \
--rnd_seed 42
```

### VICE hyperparam. combination

Find the best hyperparameter combination via `find_best_hypers.py`

```python
 find_best_hypers.py --in_path (str) \ # path/to/models/and/evaluation/results (should all have the same root directory)
 --percentages (List[int]) \ # List of full dataset fractions used for VICE optimization
 ```

#### Example call

```python
$ python find_best_hypers.py --in_path path/to/models/and/evaluation/results \
--percentages 10 20 50 100
```

### NOTES:

After calling `find_best_hypers.py`, a `txt` file called `model_paths.txt` is saved to the data split subfolder in `path/to/models/and/evaluation/results` pointing towards the latest model snapshot (i.e., last epoch) for the best hyperparameter combination per data split and random seed.


## VICE embeddings

VICE embeddings for [THINGS](https://osf.io/jum2f/) can be found [here](https://github.com/LukasMut/VICE/tree/main/embeddings/things). The corresponding object concept names can be found on [OSF](https://osf.io/jum2f/) or [here](https://github.com/LukasMut/VICE/tree/main/data/files). 

If you want to download the embeddings and the `tsv` file containing the object names simultaneously, download [this file](https://github.com/LukasMut/VICE/blob/main/get_embeddings.sh) and execute it as follows

```bash
$ bash get_embedding.sh
```

This will download the THINGS object concept names to a subdirectory called `$(pwd)/data/things` and the VICE embeddings for the THINGS objects to a different subdirectory called `$(pwd)/embeddings/things/`.

## Tripletize any data

### Tripletizing representations

`VICE` can be used for any data. We provide a file called `main_tripletize.py` that converts (latent) representations from any domain (e.g., audio, fMRI or EEG recordings, Deep Neural Network features) corresponding to some set of stimuli (e.g., images, words) into an `N x 3` matrix of `N` triplets (see triplet format above). We do this by exploiting the similarity structure of the representations.

```python
 
 main_tripletize.py --in_path (str) \ # path/to/latent/representations
 --out_path (int) \ # path/to/triplets
 --n_samples (int) \  # number of triplet combinations to be sampled
 --rnd_seed (int) \ # random seed to ensure reproducibility of triplet sampling
 ```

#### Example call

```bash
$ python main_tripletize.py --in_path path/to/latent/representations \
--out_path path/to/triplets \
--n_samples 100000 \
--rnd_seed 42
```
