import os
import re
import sys
# import time
import math
import csv
import json
# import warnings
import numpy as np
#import ase
import glob
# from ase import io
from scipy.stats import rankdata
# from scipy import interpolate
from matdeeplearn.process.atoms import Atoms
##torch imports
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
# import torch_geometric.transforms as T
# from torch_geometric.utils import degree
import random
import networkx as nx
# import math
from copy import deepcopy
from jarvis.core.lattice import Lattice, lattice_coords_transformer
# from jarvis.core.composition import Composition
from jarvis.core.specie import chem_data, get_node_attributes
from collections import defaultdict

from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski

import pandas as pd
################################################################################
# Data splitting
################################################################################

##basic train, val, test split
def split_data(
    dataset,
    train_ratio,
    val_ratio,
    test_ratio,
    seed=np.random.randint(1, 1e6),
    save=False,
):
    dataset_size = len(dataset)
    if (train_ratio + val_ratio + test_ratio) <= 1:
        train_length = int(dataset_size * train_ratio)
        val_length = int(dataset_size * val_ratio)
        test_length = int(dataset_size * test_ratio)
        unused_length = dataset_size - train_length - val_length - test_length
        (
            train_dataset,
            val_dataset,
            test_dataset,
            unused_dataset,
        ) = torch.utils.data.random_split(
            dataset,
            [train_length, val_length, test_length, unused_length],
            generator=torch.Generator().manual_seed(seed),
        )
        print(
            "train length:",
            train_length,
            "val length:",
            val_length,
            "test length:",
            test_length,
            "unused length:",
            unused_length,
            "seed :",
            seed,
        )
        #print(train_dataset[0].x)
        return train_dataset, val_dataset, test_dataset
    else:
        print("invalid ratios")


##Basic CV split
def split_data_CV(dataset, num_folds=5, seed=np.random.randint(1, 1e6), save=False):
    dataset_size = len(dataset)
    fold_length = int(dataset_size / num_folds)
    unused_length = dataset_size - fold_length * num_folds
    folds = [fold_length for i in range(num_folds)]
    folds.append(unused_length)
    cv_dataset = torch.utils.data.random_split(
        dataset, folds, generator=torch.Generator().manual_seed(seed)
    )
    print("fold length :", fold_length, "unused length:", unused_length, "seed", seed)
    return cv_dataset[0:num_folds]


################################################################################
# Pytorch datasets
################################################################################

##Fetch dataset; processes the raw data if specified _previous
def get_dataset(data_path, target_index, reprocess="False", processing_args=None):
    if processing_args == None:
        processed_path = "processed_Multi_93_15_adsorption"    
        # processed_Multi_93_15_selectivity_adsorption
    else:
        processed_path = processing_args.get("processed_path", "processed_Multi_93_15_adsorption")

    transforms = GetY(index=target_index)

    if os.path.exists(data_path) == False:
        print("Data not found in:", data_path)
        sys.exit()

    if reprocess == "True":
        os.system("rm -rf " + os.path.join(data_path, processed_path))
        process_data(data_path, processed_path, processing_args)

    if os.path.exists(os.path.join(data_path, processed_path, "data.pt")) == True:
        dataset = StructureDataset(
            data_path,
            processed_path,
            transforms,
        )
    elif os.path.exists(os.path.join(data_path, processed_path, "data_0.pt")) == True:
        dataset = StructureDataset_large(
            data_path,
            processed_path,
            transforms,
        )
    else:
        process_data(data_path, processed_path, processing_args)
        if os.path.exists(os.path.join(data_path, processed_path, "data.pt")) == True:
            dataset = StructureDataset(
                data_path,
                processed_path,
                transforms,
            )
        elif os.path.exists(os.path.join(data_path, processed_path, "data_0.pt")) == True:
            dataset = StructureDataset_large(
                data_path,
                processed_path,
                transforms,
            )        
    return dataset


##Dataset class from pytorch/pytorch geometric; inmemory case
class StructureDataset(InMemoryDataset):
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        file_names = ["data.pt"]
        return file_names


##Dataset class from pytorch/pytorch geometric
class StructureDataset_large(Dataset):
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset_large, self).__init__(
            data_path, transform, pre_transform
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        # file_names = ["data.pt"]
        file_names = []
        for file_name in glob.glob(self.processed_dir + "/data*.pt"):
            file_names.append(os.path.basename(file_name))
        # print(file_names)
        return file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, "data_{}.pt".format(idx)))
        return data


################################################################################
#  Processing
################################################################################


def process_data(data_path, processed_path, processing_args):

    #data_path: "data/MOF_data/MOF_data" processed_path = "processed2"
    ##Begin processing data
    print("Processing data to: " + os.path.join(data_path, processed_path))
    assert os.path.exists(data_path), "Data path not found in " + data_path

    ##Load dictionary
    #E:\MOF_design\MOF_design\MOF_graph\matdeeplearn\process\dictionary_default.json
    if processing_args["dictionary_source"] != "generated":#default
        if processing_args["dictionary_source"] == "default":
            print("Using default dictionary.")
            atom_dictionary = get_dictionary(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "charges_default.json",
                )
            )
        elif processing_args["dictionary_source"] == "blank":
            print(
                "Using blank dictionary. Warning: only do this if you know what you are doing"
            )
            atom_dictionary = get_dictionary(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "dictionary_blank.json"
                )
            )
        else:
            dictionary_file_path = os.path.join(
                data_path, processing_args["dictionary_path"]
            )
            if os.path.exists(dictionary_file_path) == False:
                print("Atom dictionary not found, exiting program...")
                sys.exit()
            else:
                print("Loading atom dictionary from file.")
                atom_dictionary = get_dictionary(dictionary_file_path)

    ##Load targets
    #target_property_file=data/MOF_data/MOF_data/targets1.csv
    target_property_file = os.path.join(data_path, processing_args["target_path"])
    assert os.path.exists(target_property_file), (
        "targets not found in " + target_property_file
    )
    # with open(target_property_file) as f:
    #     reader = csv.reader(f)
    #     target_data = [row for row in reader]
    
    columns_to_extract = ['MOFname','working_capacity_vacuum_swing [mmol/g]','CO2/N2_selectivity', 'volume [A^3]', 'void_fraction', 'largest_free_sphere_diameter(PLD)[A]', 'largest_included_sphere_along_free_sphere_path_diameter(LCD)[A]','CO2_uptake_P0.15bar_T298K [mmol/g]','CO2_uptake_P0.10bar_T363K [mmol/g]','CO2_uptake_P0.70bar_T413K [mmol/g]','CO2_binary_uptake_P0.15bar_T298K [mmol/g]','N2_binary_uptake_P0.85bar_T298K [mmol/g]']
   # columns_to_extract = ['MOFname','CO2_uptake_P0.15bar_T298K [mol/kg]','CO2_uptake_P0.10bar_T298K [mol/kg]','CO2_uptake_P0.70bar_T298K [mol/kg]','CO2_binary_uptake_P0.15bar_T298K [mol/kg]','N2_binary_uptake_P0.85bar_T298K [mol/kg]', 'CO2_binary_uptake_P0.25bar_T298K [mol/kg]','N2_binary_uptake_P0.75bar_T298K [mol/kg]', 'CO2_binary_uptake_P0.50bar_T298K [mol/kg]','N2_binary_uptake_P0.50bar_T298K [mol/kg]']
    df = pd.read_csv(target_property_file)
    target_data = df[columns_to_extract]
    

    ##Process structure files and create structure graphs
    use_lat = processing_args["use_lattice"]
    
    data_list = []
    atom_features="cgcnn"
    if_adsorption=processing_args["if_adsorption"]
    
    for index in range(0, len(target_data)):
        #len(target_data)=20375

        structure_id = target_data.loc[index,'MOFname']
        data = Data()
        
        #Read in structure file 
        if processing_args["data_format"] == "cif":
            cif_structure = Atoms.from_mofcif(data_path+'/'+structure_id+'.cif')
            mol = AllChem.MolFromPDBFile(data_path+'/'+structure_id+'.pdb', sanitize=True,removeHs=False,flavor=0,proximityBonding=False)
        print(structure_id)
        # print('E:\MOF_design\MOF_design\MOF_graph\data\MOF_data\MOF_data'+'\\'+structure_id+'.pdb')
        # N = mol.GetNumAtoms()
        # print(N)
        edges = nearest_neighbor_edges_submit(
            atoms=cif_structure,
            cutoff=processing_args["graph_max_radius"],
            max_neighbors=processing_args["graph_max_neighbors"],
            use_canonize=True,
            use_lattice=True,
            use_angle=False,
        )
        # print(edges.shape)
        #print(edges)
        u, v, r = build_undirected_edgedata(cif_structure, edges)
        
        #print(u)
        #print(v)
        #print(r)
        #print(u.shape)
        #print(v.shape)
        #print(r.shape)
        u_1 = []
        v_1 = []
        r_1 = torch.zeros(1,3)
        u_2 = []
        v_2 = []
        r_2 = torch.zeros(1,3)
        u_3 = []
        v_3 = []
        r_3 = torch.zeros(1,3)
        # print(u_1)
        # print(v_1)
        # print(r_1)
        for idatom in range(u.shape[0]):
            r_distance = math.sqrt(r[idatom][0]**2+r[idatom][1]**2+r[idatom][2]**2)
            if r_distance<=3:
                if len(u_1) == 0:
                    r_1=r[idatom]
                else:
                    r_1 = torch.cat([r_1,r[idatom]],0)
                u_1.append(u[idatom])
                v_1.append(v[idatom])
                continue
            elif r_distance>3 and r_distance<=8:
                
                if len(u_2) == 0:
                    r_2=r[idatom]
                    u_2.append(u[idatom])
                    v_2.append(v[idatom])
                else:
                    r_2 = torch.cat([r_2,r[idatom]],0)
                    u_2.append(u[idatom])
                    v_2.append(v[idatom])
                
                continue    
            elif r_distance>8:      
                if len(u_3) == 0:
                    r_3=r[idatom]
                else:
                    r_3 = torch.cat([r_3,r[idatom]],0)
                u_3.append(u[idatom])
                v_3.append(v[idatom]) 
                continue  
        u_1 = torch.tensor(u_1)
        v_1 = torch.tensor(v_1)
        r_1 = torch.reshape(r_1,(-1,3))
        u_2 = torch.tensor(u_2)
        v_2 = torch.tensor(v_2)
        r_2 = torch.reshape(r_2,(-1,3))
        u_3 = torch.tensor(u_3)
        v_3 = torch.tensor(v_3)
        r_3 = torch.reshape(r_3,(-1,3))
        #print(u_1.shape)
        #print(v_1.shape)
        #print(r_1.shape)
        #print(u_2.shape)
        #print(v_2.shape)
        #print(r_2.shape)
        print(u_3)
        print(v_3)
        print(r_3)
        # print(r_1)
        if u_1.shape[0] != v_1.shape[0]:
            print("edge_index_1 error!")
        elif u_1.shape[0] != r_1.shape[0]:
            print("edge_attr_1 error!")
        if u_2.shape[0] != v_2.shape[0]:
            print("edge_index_2 error!")
        elif u_2.shape[0] != r_2.shape[0]:
            print("edge_attr_2 error!")
        if u_3.shape[0] != v_3.shape[0]:
            print("edge_index_3 error!")
        elif u_3.shape[0] != r_3.shape[0]:
            print("edge_attr_3 error!")
        edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
        edge_index_1 = torch.cat((u_1.unsqueeze(0), v_1.unsqueeze(0)), dim=0).long()
        edge_index_2 = torch.cat((u_2.unsqueeze(0), v_2.unsqueeze(0)), dim=0).long()
        edge_index_3 = torch.cat((u_3.unsqueeze(0), v_3.unsqueeze(0)), dim=0).long()

        sps_features = []
        for ii, s in enumerate(cif_structure.elements):
            # node_LJ = get_node_LJ(s)
            feat = list(get_node_attributes(s, atom_features=atom_features))
            # feats = feat + node_LJ
            # print(np.size(feats))
            # print(feats)
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(torch.get_default_dtype())
        target = target_data.iloc[index,1:]
        
        y = torch.Tensor(np.array([target], dtype=np.float32))
        #'CO2/N2_selectivity', 'volume [A^3]', 'void_fraction', 'largest_free_sphere_diameter(PLD)[A]', 'largest_included_sphere_along_free_sphere_path_diameter(LCD)[A]'
        # y[0][1] = y[0][1]/25000
        # y[0][2] = y[0][2]/20000
        # y[0][3] = y[0][3]/5500
        # y[0][6] = y[0][6]/15
        data.y = y
        data.x = node_features
        if use_lat == "True":
            print("use_lattice is True")
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2
            data.edge_index_3 = edge_index_3
            data.edge_attr_1 = r_1
            data.edge_attr_2 = r_2
            data.edge_attr_3 = r_3
        else:
            print("use_lattice is False")
            data.edge_index = edge_index
            data.edge_attr = r
           #print(edge_index)
           #print(r)
           #print(edge_index.shape)
           #print(r.shape)
        data.structure_id = structure_id
   
        ###placeholder for state feature
        u = np.zeros((3))
        u = torch.Tensor(u[np.newaxis, ...])
        # print(u)
        data.u = u
        
        z = torch.LongTensor(cif_structure.num_atoms)
        data.z = z
        # AtomCharges(data, data_path, cat=True)
        position = AtomPositions(data, data_path)
        # data = OneHotDegree(data, max_degree = 25)
        # print(data.edge_index.shape[0],data.edge_index.shape[1])
        # print(data.x.shape[0],data.x.shape[1])
        #data = OneHotHAcceptor(data, mol)
        # data = OneHotChirality(data, mol)
        #data = OneHotRing(data, mol)
        #data = OneHotMetal(data, mol)
        #g = Data(x=node_features, edge_index=edge_index, edge_attr=r)
       #print(data)
        # print(data.num_edge_features)
        # print(data.num_features)
       #print(if_adsorption=="True")
        if if_adsorption=="True":
            print("is adsorption")
            data_1=Data()
            data_1.x = node_features
            if use_lat=="True":
                print("data1 use lattice.")
                data_1.edge_index_1 = edge_index_1
                data_1.edge_index_2 = edge_index_2
                data_1.edge_index_3 = edge_index_3
                data_1.edge_attr_1 = r_1
                data_1.edge_attr_2 = r_2
                data_1.edge_attr_3 = r_3
            else:
                print("data1 use no lattice.")
                data_1.edge_attr = r
                data_1.edge_index= edge_index
            data_1.structure_id = structure_id
            data_1.u = u
            data_1.z = z
            data_1.parameters = torch.Tensor(np.array([0.15, 298], dtype=np.float32))
            data_1.y = target_data.loc[index,'CO2_uptake_P0.15bar_T298K [mmol/g]']
            data_1.position = position
             
            data_2=Data()
            data_2.x = node_features
            if use_lat=="True":
                print("data2 use lattice.")
                data_2.edge_index_1 = edge_index_1
                data_2.edge_index_2 = edge_index_2
                data_2.edge_index_3 = edge_index_3
                data_2.edge_attr_1 = r_1
                data_2.edge_attr_2 = r_2
                data_2.edge_attr_3 = r_3
            else:
                print("data2 use no lattice.")
                data_2.edge_attr = r
                data_2.edge_index= edge_index
            data_2.structure_id = structure_id
            data_2.u = u
            data_2.z = z
            data_2.parameters = torch.Tensor(np.array([0.10, 363], dtype=np.float32))
            data_2.y = target_data.loc[index,'CO2_uptake_P0.10bar_T363K [mmol/g]']
            data_2.position = position
            
            data_3=Data()
            data_3.x = node_features
            if use_lat=="True":
                print("data3 use lattice.")
                data_3.edge_index_1 = edge_index_1
                data_3.edge_index_2 = edge_index_2
                data_3.edge_index_3 = edge_index_3
                data_3.edge_attr_1 = r_1
                data_3.edge_attr_2 = r_2
                data_3.edge_attr_3 = r_3
            else:
                print("data3 use no lattice.")
                data_3.edge_attr = r
                data_3.edge_index= edge_index
            data_3.structure_id = structure_id
            data_3.u = u
            data_3.z = z
            data_3.parameters = torch.Tensor(np.array([0.70, 413], dtype=np.float32))
            data_3.y = target_data.loc[index,'CO2_uptake_P0.70bar_T413K [mmol/g]']
            data_3.position = position
           #print(data_1.parameters)
           #print(data_2.parameters)
           #print(data_3.parameters)
           #print(data_1.edge_attr)
           #print(data_1.edge_attr.shape)
           #print(data_1.edge_index)
           #print(data_1.edge_index.shape)
            data_list.append(data_1)
            data_list.append(data_2)
            data_list.append(data_3)
           #print("adsorption!")
        else:
            data.parameters = torch.Tensor(np.array([0.15, 0.85, 298, 0.33, 0.364], dtype=np.float32))
            #print(data)
            data_list.append(data)
        #print(data_list[-3])
        #print(data_list[-2])
        #print(data_list[-1])
             #print(data)
        #print(data.x.shape[0],data.x.shape[1])
        # print(data.num_features)
        # print(data.edge_attr_1.size(1))
        #print(data.batch)
    
    # for index in range(0,len(data_list)):
    #     atom_fea = np.vstack(
    #         [
    #             atom_dictionary[str(data_list[index].ase.get_atomic_numbers()[i])]
    #             for i in range(len(data_list[index].ase))
    #         ]
    #     ).astype(float)
        
        
    if os.path.isdir(os.path.join(data_path, processed_path)) == False:
        os.mkdir(os.path.join(data_path, processed_path))

    ##Save processed dataset to file
    if processing_args["dataset_type"] == "inmemory":
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), os.path.join(data_path, processed_path, "data.pt"))

    elif processing_args["dataset_type"] == "large":
        for i in range(0, len(data_list)):
            torch.save(
                data_list[i],
                os.path.join(
                    os.path.join(data_path, processed_path), "data_{}.pt".format(i)
                ),
            )




################################################################################
#  Processing sub-functions
################################################################################

##Selects edges with distance threshold and limited number of neighbors
def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    #print(mask)
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    # print(distance_matrix_trimmed)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )#每一行从小到大标，同样大先出现的先标
        # print(distance_matrix_trimmed)
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    # print(distance_matrix_trimmed)
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0


    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        #print(distance_matrix_trimmed)
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr

def nearest_neighbor_edges_submit(
    atoms=None,
    cutoff=15,
    max_neighbors=12,
    #id=None,
    use_canonize=True,
    use_lattice=True,
    use_angle=False,
):
   #print(use_lattice)
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    # print(cutoff)
    # # for neighborlist in all_neighbors:
    # #     print(len(neighborlist))
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)
    # #print(min_nbrs)
    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize = use_canonize,
            cutoff=r_cut,
            use_lattice = use_lattice,
            max_neighbors=max_neighbors,
            #id=id,
        )
    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        # max_dist = distances[-1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 1, 0])))
            #print('use_lattice')
        # else:
        #     print('no_lattice')
    return edges

def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image

def build_undirected_edgedata(
    atoms=None,
    edges={},
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            # if np.linalg.norm(d)!=0:
            # print ('jv',dst_image,d)
            # add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                # print(dd)
                u.append(uu)
                v.append(vv)
                r.append(dd)
                # if uu==vv:
                #     print(uu)
                #     print(vv)
                #     print(dd)
    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r).type(torch.get_default_dtype())

    return u, v, r
    
##Slightly edited version from pytorch geometric to create edge from gaussian basis
#0, 1, processing_args["graph_edge_length"] (50), 0.2
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):

        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)

        return torch.exp(self.coeff * torch.pow(dist, 2))


##Obtain node degree in one-hot representation
def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    # print(idx)
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    # print(deg)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)
    # print(deg)
    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        #print(data.x)
    else:
        data.x = deg

    return data

def OneHotHAcceptor(data, mol, cat=True):
    acceptors = Lipinski._HAcceptors(mol)
    N = data.num_nodes
    IsHAcceptor = torch.zeros(N,1)
    for ids in acceptors:
        if len(ids) == 1:
            atom_id = int(ids[0])
            IsHAcceptor[atom_id] = 1
            #print(int(ids[0]))
        else:
            for i in range(len(ids)):
                print(data.structure_id)
                # print(len(ids))
    x = data.x
    # deg = degree(idx, data.num_nodes, dtype=torch.long)
    # deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, IsHAcceptor.to(x.dtype)], dim=-1)
        
    else:
        data.x = IsHAcceptor
    # print(IsHAcceptor)
    return data

def OneHotRing(data, mol, cat=True):
    x = data.x
    N = data.num_nodes
    IsRing = torch.zeros(N,1)
    for i in range(N):
        atom = mol.GetAtomWithIdx(i)
        isring = atom.IsInRing()
        if isring:
            IsRing[i] = 1

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, IsRing.to(x.dtype)], dim=-1)
        #print(data.x)
    else:
        data.x = IsRing
    # print(IsRing)
    return data

def OneHotMetal(data, mol, cat=True):
    x = data.x
    N = data.num_nodes
    IsMetal = torch.zeros(N,1)
    # Li:3, Be:4, Na:11, Mg:12, Al:13, K:19, Ca:20, Sc:21, Ti:22, V:23, Cr:24, Mn:25, Fe:26, Co:27, Ni:28, 
    # Cu:29, Zn:30, Ga:31, Ge:32, Rb:37, Sr:38, Y:39, Zr:40, Nb:41, Mo:42, Tc:43, Ru:44, Rh:45, Pd:46, Ag:47,
    # Cd:48, In:49, Sn:50, Sb:51, Cs:55, Ba:56, La:57, Ce:58, Pr:59, Nd:60, Pm:61, Sm:62, Eu:63, Gd:64, Tb:65,
    # Dy:66, Ho:67, Er:68, Tm:69, Yb:70, Lu:71, Hf:72, Ta:73, W:74, Re:75, Os:76, Ir:77, Pt:78, Au:79, Hg:80,
    # Tl:81, Pb:82, Bi:83, Po:84, Fr:87, Ra:88, Ac:89, Th:90, Pa:91, U:92, Np:93, Pu:94, Am:95, Cm:96, Bk:97, 
    # Cf:98, Es:99, Fm:100, Md:101, No:102, Lr:103,  Rf:104, Db:105, Sg:106, Bh:107, Hs:108, Mt:109, Ds:110,
    # Rg:111, 
    metal_list = [3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 37, 38, 39, 40, 41, 
                  42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 
                  69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 87, 88, 89, 90, 91, 92, 93, 94, 
                  95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
    for i in range(N):
        atom = mol.GetAtomWithIdx(i)
        atomnum = atom.GetAtomicNum()
        if atomnum in metal_list:
            IsMetal[i] = 1

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, IsMetal.to(x.dtype)], dim=-1)
        #print(data.x)
    else:
        data.x = IsMetal
    # print(IsMetal)
    return data

# def OneHotChirality(data, mol, cat=True):
#     x = data.x
#     N = data.num_nodes
#     for i in range(N):
#         atom = mol.GetAtomWithIdx(i)
#         handedness = atom.GetChiralTag()
        
#     #idx, x = data.edge_index[1 if in_degree else 0], data.x
#     deg = degree(idx, data.num_nodes, dtype=torch.long)
#     deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

#     if x is not None and cat:
#         x = x.view(-1, 1) if x.dim() == 1 else x
#         data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
#         #print(data.x)
#     else:
#         data.x = deg

#     return data

def get_node_LJ(s):
    with open(r'E:\MOF_design\MOF_design\MOF_graph\matdeeplearn\process\LJ.json','r') as file:
        data_LJ = file.read()
        LJ_dict = json.loads(data_LJ)
        LJ_params = LJ_dict[s]
        # print(LJ_params)
        # print(s)
    return (LJ_params)

def AtomPositions(data,data_path):
    x = data.x
    structureid = str(data.structure_id)
    #position = torch.from_numpy(data.ase.get_scaled_positions().astype(np.float32))
    cif_name = os.path.join(data_path, structureid + ".cif")
    # print(cif_name)
    positions=[]
    with open(cif_name, 'r', encoding='utf-8') as fr:
        for line in fr:
            if '_atom_type_partial_charge' in line:
                line_ = fr.readline()
                
                while 'loop_' not in line_:
                    line = line_
                    line_ = fr.readline()
                    line_data_position = re.sub(' +', " ", line)
                    position = list(map(float,line_data_position.split(" ")[-4:-1]))
                    # print(position)
                    if position == []:
                        continue
                    else:
                    # position = [float(line_data_position.split(" ")[-4].rstrip()),float(line_data_position.split(" ")[-3].rstrip()),float(line_data_position.split(" ")[-2].rstrip())]
                    
                    # print(position)
                        if len(position)!=0:
                        # position = float(position)
                            positions.append(position)
                # print(positions)
    positions = torch.tensor(positions)
    if (np.array(positions).shape[0] == np.array(data.x).shape[0]):
        
        #data.x = torch.cat([x, positions], dim=-1)
        #data.position = positions
        return positions
    else:
        print(cif_name)
    # data.x = torch.cat([x, positions], dim=-1)
    #return data

def AtomCharges(data, data_path, cat=True):
    x = data.x
    structureid = str(data.structure_id)
    #new_structureid = structureid[3:-3]
    #data_path = 'E:\MOF_design\MOF_design\MOF_graph\data\MOF_data\MOF_data'
    cif_name = os.path.join(data_path, structureid + ".cif")

    # print(cif_name)
    charges=[]
    with open(cif_name, 'r', encoding='utf-8') as fr:
        for line in fr:
            if '_atom_type_partial_charge' in line:
                line_ = fr.readline()
                
                while 'loop_' not in line_:
                    line = line_
                    line_ = fr.readline()
                    line_data_charge = re.sub(' +', " ", line)
                    charge = line_data_charge.split(" ")[-1].rstrip()
                    
                    # print(type(charge))
                    if len(charge)!=0:
                        charge = float(charge)
                        charges.append(charge)
    
    # print(type(charges))
    charges = torch.tensor(charges).view(-1,1)
    # print(charges)
    # print(data.x.size())
    if (np.array(charges).shape[0] == np.array(data.x).shape[0]):
        
        data.x = torch.cat([x, charges], dim=-1)
    else:
        print(cif_name,"wrong")
    return data

##Obtain dictionary file for elemental features
def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary


##Deletes unnecessary data due to slow dataloader
def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass


##Get min/max ranges for normalized edges
def GetRanges(dataset, descriptor_label):
    mean = 0.0
    std = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
            if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()

    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max


##Normalizes edges
def NormalizeEdge(dataset, descriptor_label):
    mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)

    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
            data.edge_descriptor[descriptor_label] - feature_min
        ) / (feature_max - feature_min)


# WIP
def SM_Edge(dataset):
    from dscribe.descriptors import (
        CoulombMatrix,
        SOAP,
        MBTR,
        EwaldSumMatrix,
        SineMatrix,
    )

    count = 0
    for data in dataset:
        n_atoms_max = len(data.ase)
        make_feature_SM = SineMatrix(
            n_atoms_max=n_atoms_max,
            permutation="none",
            sparse=False,
            flatten=False,
        )
        features_SM = make_feature_SM.create(data.ase)
        features_SM_trimmed = np.where(data.mask == 0, data.mask, features_SM)
        features_SM_trimmed = torch.Tensor(features_SM_trimmed)
        out = dense_to_sparse(features_SM_trimmed)
        edge_index = out[0]
        edge_weight = out[1]
        data.edge_descriptor["SM"] = edge_weight

        if count % 500 == 0:
            print("SM data processed: ", count)
        count = count + 1

    return dataset


################################################################################
#  Transforms
################################################################################

##Get specified y index from data.y
class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1 and self.index != 3:
            #print(data.y[0][self.index])
            #print(data.structure_id)
            #print(data.parameters)
            #print(data.y)
            data.y = data.y[0][self.index]
        if self.index == 3:
            data.y = data.y
        #if self.index == 1:
        #    max_y = torch.tensor(456.2986475, dtype=torch.float32)
        #    min_y = torch.tensor(14.20076702, dtype=torch.float32)
        #    eps = torch.tensor(1e-15, dtype=torch.float32)
        #    data.y = (torch.log10(data.y+eps)-torch.log10(min_y))/(torch.log10(max_y)-torch.log10(min_y))
        if self.index == 0:
            max_y = torch.tensor(5.115154, dtype=torch.float32)
            min_y = torch.tensor(1.500153, dtype=torch.float32)
            eps = torch.tensor(1e-15, dtype=torch.float32)
            data.y = (torch.log(data.y+eps)-torch.log(min_y))/(torch.log(max_y)-torch.log(min_y))
            #data.y = torch.cat([torch.tensor([data.y[0][self.index]]), data.y[0][3:]])
            #print(data.y)
        #data.x = data.x[0][:92]
        return data
