[![Unittests](https://github.com/LukasMut/VICE/actions/workflows/tests.yml/badge.svg)](https://github.com/LukasMut/VICE/actions/workflows/tests.yml)
![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)
[![codecov](https://codecov.io/gh/LukasMut/VICE/branch/main/graph/badge.svg?token=gntaL1yrXI)](https://codecov.io/gh/LukasMut/VICE)

# VICE: Variational Inference for Concept Embeddings

### Environment setup and dependencies

We recommend to create a virtual conda environment (e.g., `vice`) including all dependencies before running any code.

```bash
$ conda env create --prefix /path/to/conda/envs/vice --file envs/environment.yml
$ conda activate vice
```

Alternatively, dependencies can be installed via `pip`

```bash
$ pip install -r requirements.txt
```

## VICE folder / file structure

```bash
root
├── data
├── ├── sortindex
├── └── things_concepts.tsv
├── envs
├── └── environment.yml
├── models
├── ├── model.py
├── └── trainer.py
├── tests
├── ├── model
├── ├── ├── __init__.py
├── ├── └── test_model.py
├── ├── tripletizer
├── ├── ├── __init__.py
├── ├── └── test_tripletize.py
├── ├── __init__.py
├── ├── helper.py
├── ├── test_dataloader.py
├── └── test_utils.py
├── .gitignore
├── DEMO.ipynb
├── LICENSE
├── README.md
├── create_things_splits.py
├── dataloader.py
├── evaluate_robustness.py
├── find_best_hypers.py
├── inference.py
├── partition_food_data.py
├── requirements.txt
├── run.py
├── sampling.py
├── tripletize.py
├── utils.py
└── visualization.py
```

## VICE step-by-step

### VICE DEMO

We've provided a `DEMO` Jupyter Notebook to guide users through each step of the `VICE` optimization. The `DEMO` file is meant to facilitate the process of using `VICE`. In the `DEMO.ipynb` one can easily inspect whether `VICE` overfits the trainig data and behaves well with respect to the evolution of latent dimensions over training time. Embeddings can be extracted and analyzed directly in the `JN`.

### VICE optimization

Explanation of arguments in `run.py`.

```python
 
 run.py
  
 --task (str) / # odd-one-out (i.e., 3AFC) or similarity (i.e., 2AFC) task
 --modality (str) / # e.g., behavioral, text, visual, fMRI
 --triplets_dir (str) / # path/to/triplet/data
 --results_dir (str) / # optional specification of results directory (if not provided will resort to ./results/modality/latent_dim/optim/prior/seed/spike/slab/pi)
 --plots_dir (str) / # optional specification of directory for plots (if not provided will resort to ./plots/modality/latent_dim/optim/prior/seed/spike/slab/pi)
 --epochs (int) / # maximum number of epochs to run VICE optimization
 --burnin (int) / # minimum number of epochs to run VICE optimization (burnin period)
 --eta (float) / # learning rate
 --latent_dim (int) / # initial dimensionality of the model's latent space
 --batch_size (int) / # mini-batch size
 --optim (str) / # optimizer (e.g., 'adam', 'adamw', 'sgd')
 --prior (str) / # whether to use a mixture of Gaussians or Laplacians in the spike-and-slab prior (i.e., 'gaussian' or 'laplace')
 --mc_samples (int) / # number of weight matrices used in Monte Carlo sampling (for computationaly efficiency, M is set to 1 during training)
 --spike (float) / # sigma of the spike distribution
 --slab (float) / # sigma of the slab distribution
 --pi (float) / # probability value that determines the relative weighting of the distributions; the closer this value is to 1, the higher the probability that weights are drawn from the spike distribution
 --k (int) / # minimum number of items whose weights are non-zero for a latent dimension (according to importance scores)
 --ws (int) / # determines for how many epochs the number of latent causes (after pruning) is not allowed to vary (ws >> 100)
 --steps (int) / # perform validation, save model parameters and create model and optimizer checkpoints every <steps> epochs
 --device (str) / # cuda or cpu
 --num_threads (int) / # number of threads used for intraop parallelism on CPU; use only if device is CPU
 --rnd_seed (int) / # random seed
 --verbose (bool) / # show print statements about model performance and evolution of latent causes during training (can be piped into log file)
 ```

#### Example call

```python
$ python run.py --task odd_one_out --triplets_dir path/to/triplets --results_dir ./results --plots_dir ./plots --epochs 1000 --burnin 500 --eta 0.001 --latent_dim 100 --batch_size 128 --k 5 --ws 100 --optim adam --prior gaussian --mc_samples 10 --spike 0.25 --slab 1.0 --pi 0.6 --steps 50 --device cuda --rnd_seed 42 --verbose
```

### NOTES:

1. Note that triplet data is expected to be in the format `N x 3`, where N = number of trials (e.g., 100k) and 3 refers to the three objects per triplet, where `col_0` = anchor_1, `col_1` = anchor_2, `col_2` = odd one out. Triplet data must be split into train and test splits, and named `train_90.txt` or `train_90.npy` and `test_10.txt` or `test_10.npy` respectively.

2. Every `--steps` epochs (i.e., `if (epoch + 1) % steps == 0`) a `model_epoch.tar` (including model and optimizer `state_dicts`) and a `results_epoch.json` (including train and validation cross-entropy errors) file are saved to disk. In addition, after convergence of VICE, a `pruned_params.npz` (compressed binary file) with keys `pruned_loc` and `pruned_scale`, including pruned VICE parameters, is saved to disk. Latent dimensions of the pruned parameter matrices are sorted according to their overall importance. See output folder structure below for where to find these files.

3. Output folder / file structure:

```bash
root
├── results
├── ├── modality
├── ├── ├── init_dim
├── ├── ├── ├── optimizer
├── ├── ├── ├── ├── prior
├── ├── ├── ├── ├── ├── spike
├── ├── ├── ├── ├── ├── ├── slab
├── ├── ├── ├── ├── ├── ├── ├── pi
├── ├── ├── ├── ├── ├── ├── ├── ├── seed
├── ├── ├── ├── ├── ├── ├── ├── ├── ├── model
├── ├── ├── ├── ├── ├── ├── ├── ├── ├── └── f'model_epoch{epoch+1:04d}.tar' if (epoch + 1) % steps == 0
├── ├── ├── ├── ├── ├── ├── ├── ├── ├── parameters.npz
├── ├── ├── ├── ├── ├── ├── ├── ├── ├── pruned_params.npz
└── └── └── └── └── └── └── └── └── └── f'results_{epoch+1:04d}.json' if (epoch + 1) % steps == 0
```

4. If VICE was trained on triplets from the [THINGS](https://osf.io/jum2f/) database, make sure that you've saved a file called `sortindex` somewhere on disk (can be found in `data`). This is necessary to sort the `THINGS` objects in their correct order.

5. The script plots train and validation performances against (to examine overfitting) as well as negative log-likelihoods and KL-divergences alongside each other. Evolution of (selected) latent dimensions over time is also plotted after convergence. All plots can be found in `./plots/` after the optimization has finished (see `DEMO.ipynb` for more information).

### VICE evaluation

Explanation of arguments in `evaluate_robustness.py`.

```python
 
 evaluate_robustness.py
 
 --results_dir (str) / # path/to/models
 --task (str) / # odd-one-out (i.e., 3AFC) or similarity (i.e., 2AFC) task
 --modality (str) / # e.g., behavioral, fMRI, EEG, DNNs
 --n_items (int) / # number of unique items/stimuli/objects in the dataset
 --latent_dim (int) / # latent space dimensionality with which VICE was initialized at run time
 --batch_size (int) / # mini-batch size used during VICE training
 --thresh (float) / # Pearson correlation value to threshold reproducibility of dimensions (e.g., 0.8)
 --optim (str) / # optimizer that was used during training (e.g., 'adam', 'adamw', 'sgd')
 --prior (str) / # whether a Gaussian or Laplacian mixture was used in the spike-and-slab prior (i.e., 'gaussian' or 'laplace')
 --spike (float) / # sigma of spike distribution
 --slab (float) / # sigma of slab distribution
 --pi (float) / # probability value that determines likelihood of samples from the spike
 --triplets_dir (str) / # path/to/triplet/data
 --mc_samples (int) / # number of weight matrices used in Monte Carlo sampling
 --device (str) / # cpu or cuda
 --rnd_seed (int) / # random seed
 ```

#### Example call

```python
$ python evaluate_robustness.py --results_dir path/to/models --task odd_one_out --modality behavioral --n_items number/of/unique/stimuli (e.g., 1854) --latent_dim 100 --batch_size 128 --thresh 0.8 --optim adam --prior gaussian --spike 0.125 --slab 1.0 --pi 0.5 --triplets_dir path/to/triplets --mc_samples 10 --device cpu --rnd_seed 42
```

### VICE combination

Find best hyperparameter combination via `find_best_hypers.py`.

```python
 
 find_best_hypers.py
 
 --in_path (str) / # path/to/models/and/evaluation/results (should all have the same root directory)
 --percentages (List[int]) / # List of full dataset fractions used for VICE optimization
 ```

#### Example call

```python
$ python find_best_hypers.py --in_path path/to/models/and/evaluation/results --percentages 10 20 50 100
```

### NOTES:

After calling `find_best_hypers.py`, a `txt` file called `model_paths.txt` is saved to the data split subfolder in `path/to/models/and/evaluation/results` pointing towards the latest model checkpoint (i.e., last epoch) for the best hyperparameter combination per data split and random seed.

## Triplets

You can optimize `VICE` for any data. We provide a file called `tripletize.py` that converts latent representations from any domain (e.g., fMRI, EEG, DNNs) corresponding to some set of stimuli (e.g., images) into an `N x 3` matrix of triplets. We do this by exploiting the similarity structure of the representations.

```python
 
 tripletize.py
 
 --in_path (str) / # path/to/latent/representations
 --out_path (int) / # path/to/triplets
 --n_samples (int) / # number of triplet combinations to be sampled
 --rnd_seed (int) / # random seed to reproduce triplet sampling
 ```

#### Example call

```python
$ python tripletize.py --in_path path/to/latent/representations --out_path path/to/triplets --n_samples 100000 --rnd_seed 42
```

