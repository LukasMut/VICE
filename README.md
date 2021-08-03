[![Unittests](https://github.com/ViCCo-Group/VSPoSE/actions/workflows/python-package.yml/badge.svg)](https://github.com/ViCCo-Group/VSPoSE/actions/workflows/python-package.yml)
[![Code Coverage](https://codecov.io/gh/ViCCo-Group/VSPoSE/branch/main/graph/badge.svg?token=0RKlKIYtbd)](https://github.com/ViCCo-Group/VSPoSE/actions/workflows/coverage.yml)

# V-SPoSE

### Setting up virtual environment and installing dependencies

We recommend to create a new virtual environment and install all dependencies via conda.

```bash
$ conda env create --prefix /Users/$(whoami)/anaconda3/envs/vspose --file envs/environment.yml
$ conda activate vspose
```

Alternatively, one can install all dependencies via pip in the usual way.

```bash
$ pip install -r requirements.txt
```

