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

### The configuration file

The configuration file is provided in .yml format and encodes all the settings used. By default it should be in the same directory as main.py or specified in a separate location by --config_path in the command line. 

There are four categories or sections: 1. Job, 2. Processing, 3. Training, 4. Models

1. **Job:** This section encodes the settings specific to the type of job to run. Current supported are: Training, Predict, Repeat, CV, Hyperparameter, Ensemble, Analysis. The program will only read the section for the current job, which is selected by --run_mode in the command line, e.g. --run_mode=Training. Some other settings which can be changed in the command line are: --job_name, --model, --seed, --parallel.

2. **Processing:** This section encodes the settings specific to the processing of structures to graphs or other features. Primary settings are the "graph_max_radius", "graph_max_neighbors" and "graph_edge_length" which controls radius cutoff for edges, maximum number of edges, and length of edges from a basis expansion, respectively. Prior to this, the directory path containing the structure files must be specified by "data_path" in the file or --data_path in the command line.

3. **Training:** This section encodes the settings specific to the training. Primary settings are the "loss", "train_ratio" and "val_ratio" and "test_ratio". This can also be specified in the command line by --train_ratio, --val_ratio, --test_ratio.

4. **Models:** This section encodes the settings specific to the model used, aka hyperparameters. Example hyperparameters are provided in the example config.yml. Only the settings for the model selected in the Job section will be used. Model settings which can be changed in the command line are: --epochs, --batch_size, and --lr.

### The configuration file

The configuration file is provided in .yml format and encodes all the settings used. By default it should be in the same directory as main.py or specified in a separate location by --config_path in the command line. 

There are four categories or sections: 1. Job, 2. Processing, 3. Training, 4. Models

1. **Job:** This section encodes the settings specific to the type of job to run. Current supported are: Training, Predict, Repeat, CV, Hyperparameter, Ensemble, Analysis. The program will only read the section for the current job, which is selected by --run_mode in the command line, e.g. --run_mode=Training. Some other settings which can be changed in the command line are: --job_name, --model, --seed, --parallel.

2. **Processing:** This section encodes the settings specific to the processing of structures to graphs or other features. Primary settings are the "graph_max_radius", "graph_max_neighbors" and "graph_edge_length" which controls radius cutoff for edges, maximum number of edges, and length of edges from a basis expansion, respectively. Prior to this, the directory path containing the structure files must be specified by "data_path" in the file or --data_path in the command line.

3. **Training:** This section encodes the settings specific to the training. Primary settings are the "loss", "train_ratio" and "val_ratio" and "test_ratio". This can also be specified in the command line by --train_ratio, --val_ratio, --test_ratio.

4. **Models:** This section encodes the settings specific to the model used, aka hyperparameters. Example hyperparameters are provided in the example config.yml. Only the settings for the model selected in the Job section will be used. Model settings which can be changed in the command line are: --epochs, --batch_size, and --lr.


### Training and prediction on an unseen dataset

This example provides instructions for a conventional ML task of training on an existing dataset, and using a trained model to provide predictions on an unseen dataset for screening. This assumes the model used is already sufficiently good at the task to be performed (with suitable model hyperparameters, etc.). The default hyperparameters can do a reasonably good job for testing purposes; for hyperparameter optimization refer to the next section.

1. To run, MatDeepLearn requires: 
	- A configuration file, config.yml, as described in the previous section. 
	- A dataset directory containing structure files, a csv file containing structure ids and target properties (default: targets.csv), and optionally a json file containing elemental properties (default: atom_dict.json). Five example datasets are provided with all requisite files needed. Structure files can take any format supported by the Atomic Simulation Environment [(ASE)](https://wiki.fysik.dtu.dk/ase/) such as .cif, .xyz, POSCAR, and ASE's own .json format.

2. It is then necessary to first train the ML model an on existing dataset with available target properties. A general example for training is:

	```bash
	python main.py --data_path='XXX' --job_name="my_training_job" --run_mode='Training' --model='CGCNN_demo' --save_model='True' --model_path='my_trained_model.pth'
	```		
	where "data_path" points to the path of the training dataset, "model" selects the model to use, and "run_mode" specifies training. Once finished, a "my_trained_model.pth" should be saved. 

3. Run the prediction on an unseen dataset by:

	```bash
	python main.py --data_path='YYY' --job_name="my_prediction_job" --run_mode='Predict' --model_path='my_trained_model.pth'
	```		
	where the "data_path" and "run_mode" are now updated, and the model path is specified. The predictions will then be saved to my_prediction_job_predicted_outputs.csv for analysis.
	

### Repeat trials

Sometimes it is desirable to obtain performance averaged over many trials. Specify repeat_trials in the config.yml for how many trials to run.

```bash
python main.py --data_path=data/test_data --run_mode=Repeat
```		

### Cross validation

Specify cv_folds in the config.yml for how many folds in the CV.

```bash
python main.py --data_path=data/test_data --run_mode=CV
```		

### Analysis

This mode allows the visualization of graph-wide features with t-SNE.

```bash
python main.py --data_path=data/test_data --run_mode=Analysis --model_path=XXX
```		
## Acknowledgements

Contributors: Lujun Li
## Contact

Code is maintained by:

[Lujun Li]
