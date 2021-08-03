[![Unittests](https://github.com/ViCCo-Group/VSPoSE/actions/workflows/python-package.yml/badge.svg)](https://github.com/ViCCo-Group/VSPoSE/actions/workflows/python-package.yml)
[![Code Coverage](https://codecov.io/gh/ViCCo-Group/VSPoSE/branch/main/graph/badge.svg?token=0RKlKIYtbd)](https://github.com/ViCCo-Group/VSPoSE/actions/workflows/coverage.yml)

# V-SPoSE

### Environment setup and dependencies

We recommend to create a virtual conda environment with all dependencies installed.

```bash
$ conda env create --prefix /Users/$(whoami)/anaconda3/envs/vspose --file envs/environment.yml
$ conda activate vspose
```

Alternatively, one can install dependencies via pip in the usual way.

```bash
$ pip install -r requirements.txt
```

### Latent space optimization

```python
 
 train.py
  
 --task (str) # odd-one-out (i.e., 3AFC) or similarity (i.e., 2AFC) task
 --modality (str) # e.g., behavioral, text, visual
 --triplets_dir (str) # path/to/triplets)
 --results_dir (str) # optional specification of results directory (if not provided will resort to ./results/modality/version/dim/lambda/seed/)
 --plots_dir (str) # optional specification of directory for plots (if not provided will resort to ./plots/modality/version/dim/lambda/seed/)
 --learning_rate (float) # learning rate for Adam
 --embed_dim (int) # initial dimensionality of the latent space
 --batch_size (int) # batch size
 --epochs (int) # maximum number of epochs
 --mc_samples (int) # number of samples to be used for monte-carlo sampling
 --spike (float) # scale of the spike distribution
 --slab (float) # scale of the slab distribution
 --pi (float) # probability value that determines the weighting of the distributions; the higher this value, the more values will be drawn from the spike distrbution (i.e., sparser solution)
 --steps (int) # perform validation, save model parameters and create checkpoints every <steps> epochs
 --device (str) # cuda or cpu
 --rnd_seed (int) # random seed
 ```

#### Example call

```python
$ python train.py --task odd_one_out --triplets_dir path/to/triplets --results_dir ./results --plots_dir ./plots --learning_rate 0.001 --embed_dim 100 --batch_size 128 --epochs 1000 --mc_samples 20 --spike 0.1 --slab 1.0 --pi 0.5 --steps 50 --device cuda --rnd_seed 42
```

