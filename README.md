# AI4MOF
MultiScaleGNN is a self attention based multi-scale graph neural network for material chemistry applications. MultiScaleGNN obtains data in the form of atomic structure and its target characteristics, processes the data into multi-scale graphs, trains models, and provides predictions for the CO2 adsorption performance and CO2/N2 adsorption selectivity and structural characteristics of MOFs. This platform allows benchmark testing of different graph neural networks, including MultiScaleGNN, on different datasets extracted from CoRE MOFand hMOF databases. This software package uses Python Geometry（ https://github.com/rusty1s/pytorch_geometric ）The library provides powerful tools for GNN development and offers many easy-to-use pre built models.

## Table of contents
<ol>
	<li><a href="#installation">Installation</a></li>
	<li><a href="#usage">Usage</a></li>
	<li><a href="#acknowledgements">Acknowledgements</a></li>
</ol>

## Installation


### Prerequisites

Prerequisites are listed in requirements.txt. You will need two key packages, 1. Pytorch and 2. Pytorch-Geometric. You may want to create a virtual environment first, using Conda for example.

1. **Pytorch**: The package has been tested on Pytorch 1.9. To install, for example:
	```bash
	pip install torch==1.9.0
	```
2. **Pytorch-Geometric:**  The package has been tested on Pytorch-Geometric. 2.0.1 To install, [follow their instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), for example:
	```bash
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-geometric
	```	
    where where ${CUDA} and ${TORCH} should be replaced by your specific CUDA version (cpu, cu92, cu101, cu102, cu110, cu111) and PyTorch version (1.7.0, 1.8.0, 1.9.0), respectively.

3. **Remaining requirements:** The remainder may be installed by:
	```bash
    git clone https://github.com/vxfung/MOF_graph
    cd MatDeepLearn    
	pip install -r requirements.txt
	```

 
