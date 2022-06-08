# DeepPhysX.Torch

This python package is part of the [DeepPhysX](https://github.com/mimesis-inria/DeepPhysX) project.
It contains adaptations of some Core components that are compatible with the [PyTorch](https://pytorch.org/)
framework.

### Quick install

The package requires [DeepPhysX](https://github.com/mimesis-inria/DeepPhysX) and [PyTorch](https://pytorch.org/) to be 
installed.

The easiest way to install is using `pip`, but there are a several way to install and configure a **DeepPhysX**
environment (refer to the [**documentation**](https://deepphysx.readthedocs.io) for further instructions).

```bash
$ pip install DeepPhysX.Torch
```

If cloning sources, clone it in the same repository as other `DeepPhysX` packages.
It must be cloned in a directory with the corresponding name as shown below:

``` bash
$ mkdir DeepPhysX
$ cd DeepPhysX
$ git clone https://github.com/mimesis-inria/DeepPhysX.git Core             # Clone default package
$ git clone https://github.com/mimesis-inria/DeepPhysX.Torch.git Torch        # Clone AI package
$ ls
Core Torch
```
