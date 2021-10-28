[![Unittests](https://github.com/LukasMut/VICE/actions/workflows/python-package.yml/badge.svg)](https://github.com/LukasMut/VICE/actions/workflows/python-package.yml)
[![Code Coverage](https://codecov.io/gh/LukasMut/VICE/branch/main/graph/badge.svg?token=0RKlKIYtbd)](https://github.com/LukasMut/VICE/actions/workflows/coverage.yml)

# VICE: Variational Inference for Concept Embeddings

### Environment setup and dependencies

We recommend to create a virtual conda environment (e.g., `vice`) including all dependencies before running any code.

```bash
$ conda env create --prefix /path/to/conda/envs/vice --file envs/environment.yml
$ conda activate vice
```

Alternatively, dependencies can be installed via `pip` in the usual way.

```bash
$ pip install -r requirements.txt
```

### VICE optimization

Explanation of arguments in `run.py`.

```python
 
 run.py
  
 --task (str) / # odd-one-out (i.e., 3AFC) or similarity (i.e., 2AFC) task
 --modality (str) / # e.g., behavioral, text, visual
 --triplets_dir (str) / # path/to/triplets
 --results_dir (str) / # optional specification of results directory (if not provided will resort to ./results/modality/latent_dim/optim/prior/seed/spike/slab/pi)
 --plots_dir (str) / # optional specification of directory for plots (if not provided will resort to ./plots/modality/latent_dim/optim/prior/seed/spike/slab/pi)
 --epochs (int) / # maximum number of epochs to train VICE
 --eta (float) / # learning rate
 --latent_dim (int) / # initial dimensionality of the model's latent space
 --batch_size (int) / # mini-batch size
 --optim (str) / # optimizer (e.g., 'adam', 'adamw', 'sgd')
 --prior (str) / # whether to use a mixture of Gaussians or Laplacians in the spike-and-slab prior (i.e., 'gaussian' or 'laplace')
 --mc_samples (int) / # number of weight matrices to be sampled at inference time (for computationaly efficiency, M is set to 1 during training)
 --spike (float) / # sigma of the spike distribution
 --slab (float) / # sigma of the slab distribution
 --pi (float) / # probability value that determines the relative weighting of the distributions; the closer this value is to 1, the higher the probability that weights are drawn from the spike distribution
 --steps (int) / # perform validation, save model parameters and create model and optimizer checkpoints every <steps> epochs
 --device (str) / # cuda or cpu
 --rnd_seed (int) / # random seed
 --verbose (bool) / # show print statements about model performance during training (can be piped into log file)
 ```

#### Example call

```python
$ python run.py --task odd_one_out --triplets_dir path/to/triplets --results_dir ./results --plots_dir ./plots --epochs 1000 --eta 0.001 --latent_dim 100 --batch_size 128 --optim adam --prior gaussian --epochs 1000 --mc_samples 25 --spike 0.1 --slab 1.0 --pi 0.5 --steps 50 --device cuda --rnd_seed 42 --verbose
```

### NOTES:

1. Note that triplet data is expected to be in the format `N x 3`, where N = number of trials (e.g., 100k) and 3 refers to the three objects per triplet, where `col_0` = anchor_1, `col_1` = anchor_2, `col_2` = odd one out. Triplet data must be split into train and test splits, and named `train_90.txt` or `train_90.npy` and `test_10.txt` or `test_10.npy` respectively.


### VICE evaluation

Explanation of arguments in `evaluate_robustness.py`.

```python
 
 evaluate_robustness.py
 
 --results_dir (str) / # path/to/models
 --task (str) / # odd-one-out (i.e., 3AFC) or similarity (i.e., 2AFC) task
 --modality (str) / # e.g., behavioral, fMRI, EEG, DNNs
 --n_items (int) / # number of unique items/stimuli/objects in the dataset
 --latent_dim (int) / # latent space dimensionality with which VICE was initialized
 --batch_size (int) / # mini-batch size used during VICE training
 --thresh (float) / # Pearson correlation threshold (e.g., 0.8)
 --optim (str) / # optimizer that was used during training (e.g., 'adam', 'adamw', 'sgd')
 --prior (str) / # whether a mixture of Gaussians or Laplacians was used in the spike-and-slab prior (i.e., 'gaussian' or 'laplace')
 --spike (float) / # sigma of spike distribution
 --slab (float) / # sigma of slab distribution
 --pi (float) / # probability value with which to sample from the spike
 --triplets_dir (str) / # path/to/triplets
 --n_components (List[int]) / # number of modes to use in the Gaussian Mixture Model (GMM)
 --mc_samples (int) / # number of weight matrices to be sampled at inference time
 --things (bool) / # whether pruning pipeline should be applied to models that were trained on objects from the THINGS database
 --index_path (str) / # if objects from THINGS database are used, path/to/sortindex must be provided
 --device (str) / # cpu or cuda
 --rnd_seed (int) / # random seed
 ```

#### Example call

```python
$ python evaluate_robustness.py --results_dir path/to/models --task odd_one_out --modality behavioral --n_items number/of/unique/stimuli (e.g., 1854) --latent_dim 100 --batch_size 128 --thresh 0.8 --optim adam --prior gaussian --spike 0.125 --slab 1.0 --pi 0.5 --triplets_dir path/to/triplets --n_components 2 3 4 5 6 --mc_samples 30 --things --index_path ./data/sortindex --device cpu --rnd_seed 42
```

### NOTES:

1. If the pruning pipeline should be applied to models that were trained on triplets created from the [THINGS](https://osf.io/jum2f/) objects, make sure that you've saved a file called `sortindex` somewhere on disk. This is necessary to sort the THINGS objects in their correct order. 


### VICE combination

Find best hyperparameter combination via `find_best_hypers.py`.

```python
 
 find_best_hypers.py
 
 --in_path (str) / # path/to/models/and/evaluation/results (should all have the same root directory)
 --percentages (List[int]) / # List of full dataset fractions used for VICE optimization
 --thresh (float) / # reproducibility threshold used for VICE evaluation (e.g., 0.8)
 --seeds (List[int]) / # List of random seeds used to initialize VICE
 ```

#### Example call

```python
$ python find_best_hypers.py --in_path path/to/models/and/evaluation/results --percentages 10 20 50 100 --thresh 0.8 --seeds 3 10 19 30 42
```

### NOTES:

1. After correctly calling `find_best_hypers.py`, you will find a `json` file called `validation_results.json` in `path/to/models/and/evaluation/results` with keys `tuning_cross_entropies`, `pruning_cross_entropies`, `robustness`, and `best_comb`, summarizing both the validation performance and the reliability scores of the best hyperparameter combination for VICE per data split and random seed.

2. Additionally, for each data split, a `txt` file called `model_paths.txt` is saved to the data split subfolder in `path/to/models/and/evaluation/results` pointing towards the latest model checkpoint (i.e., last epoch) for the best hyperparameter combination per data split and random seed.

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

