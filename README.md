[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/flatironinstitute/nemos/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/flatironinstitute/nemos/graph/badge.svg?token=vvtrcTFNeu)](https://codecov.io/gh/flatironinstitute/nemos)
[![Documentation Status](https://readthedocs.org/projects/nemos/badge/?version=latest)](https://nemos.readthedocs.io/en/latest/?badge=latest)

# nemos
NEural MOdelS, a statistical modeling framework for neuroscience.

## Disclaimer
This is an alpha version, the code is in active development and the API is subject to change.

## Setup

To install, clone this repo and install using `pip`:

``` sh
git clone git@github.com:flatironinstitute/nemos.git
cd nemos/
pip install -e .
```

If you have a GPU, you may need to install jax separately to get the proper
build. The following has worked for me on a Flatiron Linux workstation: `conda
install jax cuda-nvcc -c conda-forge -c nvidia`. Note this should be done
without `jax` and `jaxlib` already installed, so either run this before the
earlier `pip install` command or uninstall them first (`pip uninstall jax
jaxlib`). See [jax docs](https://github.com/google/jax#conda-installation) for
details (the `pip` instructions did not work for me).

