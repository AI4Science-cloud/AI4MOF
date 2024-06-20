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
    git clone https://github.com/AI4Science-cloud/AI4MOF
    cd MOF_graph    
	pip install -r requirements.txt
	```
## Usage

### Running your first calculation

This example provides instructions for adsorption capacity prediction. Procedure below:

1. Go to MOF_graph/data/MOF_data and type
	```bash
	tar -xvf MOF_database.tar.gz 
	```
	to unpack the dataset.
	
2.	Go to MOF_graph, type
	```bash
	python main.py --data_path=data/MOF_data/MOF_data
	```
	where default settings will be used and configurations will be read from the provided config.yml.
	
3. The program will begin training; As default, the program will provide two outputs: (1) "XXX_model.pth" which is a saved model which can be used for predictions on new structures, (2) "XXX_job_train_job_YYY_outputs.csv" where YYY are train, val and test; these contain structure ids, targets and the predicted values from the last epoch of training and validation, and for the test set.

 
