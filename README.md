[![Unittests](https://github.com/ViCCo-Group/VSPoSE/actions/workflows/python-package.yml/badge.svg)](https://github.com/ViCCo-Group/VSPoSE/actions/workflows/python-package.yml)
[![Code Coverage](https://codecov.io/gh/ViCCo-Group/VSPoSE/branch/main/graph/badge.svg?token=0RKlKIYtbd)](https://github.com/ViCCo-Group/VSPoSE/actions/workflows/coverage.yml)

# BORING: Bayesian Object Representations Induced by Non-negative Gaussians

### Environment setup and dependencies

We recommend to create a virtual conda environment (e.g., `vspose`) including all dependencies before running any code.

```bash
$ conda env create --prefix /path/to/conda/envs/vspose --file envs/environment.yml
$ conda activate vspose
```

Alternatively, dependencies can be installed via `pip` in the usual way.

```bash
$ pip install -r requirements.txt
```

### BORING optimization

Explanation of arguments in `train.py`.

```python
 
 train.py
  
 --task (str) / # odd-one-out (i.e., 3AFC) or similarity (i.e., 2AFC) task
 --modality (str) / # e.g., behavioral, text, visual
 --triplets_dir (str) / # path/to/triplets
 --results_dir (str) / # optional specification of results directory (if not provided will resort to ./results/modality/version/dim/lambda/seed/)
 --plots_dir (str) / # optional specification of directory for plots (if not provided will resort to ./plots/modality/version/dim/lambda/seed/)
 --learning_rate (float) / # learning rate for Adam
 --embed_dim (int) / # initial dimensionality of the latent space
 --batch_size (int) / # mini-batch size
 --epochs (int) / # maximum number of epochs
 --mc_samples (int) / # number of weight samples used in Monte Carlo (MC) sampling at val time (for computationaly efficiency, M is set to 1 during trainig)
 --spike (float) / # sigma of the spike distribution
 --slab (float) / # sigma of the slab distribution
 --pi (float) / # probability value that determines the relative weighting of the distributions; the higher this value, the higher the probability that weights are drawn from the spike distribution
 --steps (int) / # perform validation, save model parameters and create model and optimizer checkpoints every <steps> epochs
 --device (str) / # cuda or cpu
 --rnd_seed (int) / # random seed
 ```

#### Example call

```python
$ python train.py --task odd_one_out --triplets_dir path/to/triplets --results_dir ./results --plots_dir ./plots --learning_rate 0.001 --embed_dim 100 --batch_size 128 --epochs 1000 --mc_samples 25 --spike 0.1 --slab 1.0 --pi 0.5 --steps 50 --device cuda --rnd_seed 42
```

### NOTES:

1. Note that triplet data is expected to be in the format `N x 3`, where N = number of trials (e.g., 100k) and 3 refers to the three objects per triplet, where `col_0` = anchor_1, `col_1` = anchor_2, `col_2` = odd one out. Triplet data must be split into train and test splits, and named `train_90.txt` or `train_90.npy` and `test_10.txt` or `test_10.npy` respectively.


### Reliability evaluation

Explanation of arguments in `evaluate_robustness.py`.

```python
 
 evaluate_robustness.py
 
 --results_dir (str) / # path/to/models
 --task (str) / # odd-one-out (i.e., 3AFC) or similarity (i.e., 2AFC) task
 --modality (str) / # e.g., behavioral, fMRI, EEG, DNNs
 --n_items (int) / # number of unique items/stimuli/objects in dataset
 --dim (int) / # initial latent space dimensionality of V-SPoSE params
 --thresh (float) / # reproducibility threshold (e.g., 0.8)
 --batch_size (int) / # batch size used for training V-SPoSE
 --spike (float) / # sigma of spike distribution
 --slab (float) / # sigma of slab distribution
 --pi (float) / # probability value with which to sample from the spike
 --triplets_dir (str) / # path/to/triplets
 --n_components (List[int]) / # number of modes in the Gaussian Mixture Model (GMM)
 --mc_samples (int) / # number of samples used in Monte Carlo (MC) sampling during validation
 --things (bool) / # whether pruning pipeline should be applied to models that were training on THINGS objects
 --index_path (str) / # if objects from THINGS database are used, path/to/sortindex must be provided
 --device (str) / # cuda or cpu
 --rnd_seed (int) / # random seed
 ```

#### Example call

```python
$ python evaluate_robustness.py --results_dir path/to/models --task odd_one_out --modality behavioral --n_items number/of/unique/stimuli --dim 100 --thresh 0.85 --batch_size 128 --spike 0.125 --slab 1.0 --pi 0.5 --triplets_dir path/to/triplets --n_components 2 3 4 5 6 --mc_samples 30 --things --index_path ./data/sortindex --device cpu --rnd_seed 42
```

### NOTES:

1. If the pruning pipeline should be applied to models that were trained on triplets created from the [THINGS](https://osf.io/jum2f/) objects, make sure that you've saved a file called `sortindex` somewhere on disk. This is necessary to sort the objects in the correct order. 
