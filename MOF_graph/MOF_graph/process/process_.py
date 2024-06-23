import os
import re
import sys
import time
import csv
import json
import warnings
import numpy as np
import ase
import glob
from ase import io
from scipy.stats import rankdata
from scipy import interpolate
from matdeeplearn.process.atoms import Atoms
##torch imports
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import random
import networkx as nx
import math
from copy import deepcopy
from jarvis.core.lattice import Lattice, lattice_coords_transformer
from jarvis.core.composition import Composition
from collections import defaultdict
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

##Fetch dataset; processes the raw data if specified
def get_dataset(data_path, target_index, reprocess="False", processing_args=None):
    if processing_args == None:
        processed_path = "processed2"
    else:
        processed_path = processing_args.get("processed_path", "processed2")

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
                    "dictionary_default.json",
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
    with open(target_property_file) as f:
        reader = csv.reader(f)
        target_data = [row for row in reader]

    ##Read db file if specified
    ase_crystal_list = []
    if processing_args["data_format"] == "db":#cif
        db = ase.db.connect(os.path.join(data_path, "data.db"))
        row_count = 0
        # target_data=[]
        for row in db.select():
            # target_data.append([str(row_count), row.get('target')])
            ase_temp = row.toatoms()
            ase_crystal_list.append(ase_temp)
            row_count = row_count + 1
            if row_count % 500 == 0:
                print("db processed: ", row_count)

    ##Process structure files and create structure graphs
    data_list = []
    for index in range(0, len(target_data)):
        #len(target_data)=20375

        structure_id = target_data[index][0]
        data = Data()
        
        
        #Read in structure file using ase
        if processing_args["data_format"] != "db":
            ase_crystal = ase.io.read(
                os.path.join(
                    data_path, structure_id + "." + processing_args["data_format"]
                )
            )
          
            data.ase = ase_crystal
            # print(data.ase)
        else:
            ase_crystal = ase_crystal_list[index]
            data.ase = ase_crystal

        ##Compile structure sizes (# of atoms) and elemental compositions
        if index == 0:
            length = [len(ase_crystal)]
            #print(length)
            elements = [list(set(ase_crystal.get_chemical_symbols()))]
            #print(elements)
              
        else:
            length.append(len(ase_crystal))
            elements.append(list(set(ase_crystal.get_chemical_symbols())))
        
   
        distance_matrix = ase_crystal.get_all_distances(mic=True)
        # print(distance_matrix)
        
        distance_matrix_trimmed = threshold_sort(
            distance_matrix,
            processing_args["graph_max_radius"],
            processing_args["graph_max_neighbors"],
            adj=False,
        )

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        # print(distance_matrix_trimmed)
        edge_index = out[0]
        # print(edge_index)
        edge_weight = out[1]
        # print(edge_weight)

        self_loops = True
        if self_loops == True:
            #print(len(edge_index), len(edge_weight))
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0
            )

            data.edge_index = edge_index
            data.edge_weight = edge_weight
            # print(edge_index)
            # print(edge_weight)
            distance_matrix_mask = (
                distance_matrix_trimmed.fill_diagonal_(1) != 0
            ).int()
            # print(distance_matrix_mask)
        elif self_loops == False:
            data.edge_index = edge_index
            data.edge_weight = edge_weight

            distance_matrix_mask = (distance_matrix_trimmed != 0).int()

        data.edge_descriptor = {}
        data.edge_descriptor["distance"] = edge_weight
        data.edge_descriptor["mask"] = distance_matrix_mask

        target = target_data[index][1:]
        #print(target)
        y = torch.Tensor(np.array([target], dtype=np.float32))
        data.y = y

        z = torch.LongTensor(ase_crystal.get_atomic_numbers())
        data.z = z
        # print(z)

        ###placeholder for state feature
        u = np.zeros((3))
        u = torch.Tensor(u[np.newaxis, ...])
        # print(u)
        data.u = u

        data.structure_id = [[structure_id] * len(data.y)]
        print(data.structure_id)

        if processing_args["verbose"] == "True" and (
            (index + 1) % 500 == 0 or (index + 1) == len(target_data)
        ):
            print("Data processed: ", index + 1, "out of", len(target_data))


        data_list.append(data)

    ##
    n_atoms_max = max(length)
    species = list(set(sum(elements, [])))
    #print(sum(elements, []))
    species.sort()
    num_species = len(species)
    if processing_args["verbose"] == "True":
        print(
            "Max structure size: ",
            n_atoms_max,
            "Max number of elements: ",
            num_species,
        )
        print("Unique species:", species)
    crystal_length = len(ase_crystal)
    #print(len(ase_crystal))
    data.length = torch.LongTensor([crystal_length])

    ##Generate node features
    if processing_args["dictionary_source"] != "generated":
        ##Atom features(node features) from atom dictionary file
        for index in range(0, len(data_list)):
            atom_fea = np.vstack(
                [
                    atom_dictionary[str(data_list[index].ase.get_atomic_numbers()[i])]
                    for i in range(len(data_list[index].ase))
                ]
            ).astype(float)
            data_list[index].x = torch.Tensor(atom_fea)

    elif processing_args["dictionary_source"] == "generated":
        ##Generates one-hot node features rather than using dict file
        from sklearn.preprocessing import LabelBinarizer

        lb = LabelBinarizer()
        lb.fit(species)
        for index in range(0, len(data_list)):
            data_list[index].x = torch.Tensor(
                lb.transform(data_list[index].ase.get_chemical_symbols())
            )






    ##Adds node degree to node features (appears to improve performance)
    for index in range(0, len(data_list)):
        data_list[index] = OneHotDegree(
            data_list[index], processing_args["graph_max_neighbors"] + 1
        )
        #print(data_list[index].ase.get_positions(wrap=True))
        data_list[index] = AtomPositions(data_list[index])
        data_list[index] = AtomCharges(data_list[index],data_path)

    processing_args["voronoi"] = "False"
 
    ##makes SOAP and SM features from dscribe
    if processing_args["SOAP_descriptor"] == "True":
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False

        from dscribe.descriptors import SOAP
            
        make_feature_SOAP = SOAP(
            species=species,
            rcut=processing_args["SOAP_rcut"],
            nmax=processing_args["SOAP_nmax"],
            lmax=processing_args["SOAP_lmax"],
            sigma=processing_args["SOAP_sigma"],
            periodic=periodicity,
            sparse=False,
            average="inner",
            rbf="gto",
            crossover=False,
        )
        for index in range(0, len(data_list)):
            features_SOAP = make_feature_SOAP.create(data_list[index].ase)
            data_list[index].extra_features_SOAP = torch.Tensor(features_SOAP)
            if processing_args["verbose"] == "True" and index % 500 == 0:
                if index == 0:
                    print(
                        "SOAP length: ",
                        features_SOAP.shape,
                    )
                print("SOAP descriptor processed: ", index)

    elif processing_args["SM_descriptor"] == "True":
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False

        from dscribe.descriptors import SineMatrix, CoulombMatrix
        
        if periodicity == True:
            make_feature_SM = SineMatrix(
                n_atoms_max=n_atoms_max,
                permutation="eigenspectrum",
                sparse=False,
                flatten=True,
            )
        else:
            make_feature_SM = CoulombMatrix(
                n_atoms_max=n_atoms_max,
                permutation="eigenspectrum",
                sparse=False,
                flatten=True,
            )
            
        for index in range(0, len(data_list)):
            features_SM = make_feature_SM.create(data_list[index].ase)
            data_list[index].extra_features_SM = torch.Tensor(features_SM)
            if processing_args["verbose"] == "True" and index % 500 == 0:
                if index == 0:
                    print(
                        "SM length: ",
                        features_SM.shape,
                    )
                print("SM descriptor processed: ", index)

   

    
##Generate edge features
    if processing_args["edge_features"] == "True":

        ##Distance descriptor using a Gaussian basis
        distance_gaussian = GaussianSmearing(
            0, 1, processing_args["graph_edge_length"], 0.2
        )
        # print(GetRanges(data_list, 'distance'))
        NormalizeEdge(data_list, "distance")
        # print(GetRanges(data_list, 'distance'))
        for index in range(0, len(data_list)):
            data_list[index].edge_attr = distance_gaussian(
                data_list[index].edge_descriptor["distance"]
            )
            if processing_args["verbose"] == "True" and (
                (index + 1) % 500 == 0 or (index + 1) == len(target_data)
            ):
                print("Edge processed: ", index + 1, "out of", len(target_data))


    data_list_ = []
    for index in range(0, len(data_list)):

        data_i = Augmention(data_list[index])
        #data_i, data_j, data_k = Augmention(data_list[index])
        data_list_.append(data_i)
        # data_list_.append(data_j)
        # data_list_.append(data_k)

    Cleanup(data_list_, ["ase", "edge_descriptor"])
    #for index in range(0, len(data_list)):
        #print(data_list[index])
    if os.path.isdir(os.path.join(data_path, processed_path)) == False:
        os.mkdir(os.path.join(data_path, processed_path))

    ##Save processed dataset to file
    if processing_args["dataset_type"] == "inmemory":
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), os.path.join(data_path, processed_path, "data.pt"))

    elif processing_args["dataset_type"] == "large":
        for i in range(0, len(data_list_)):
            torch.save(
                data_list_[i],
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
    cutoff=8,
    max_neighbors=12,
    #id=None,
    use_canonize=False,
    use_lattice=False,
    use_angle=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    # for neighborlist in all_neighbors:
    #     print(len(neighborlist))
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)
    #print(min_nbrs)
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
            use_canonize=use_canonize,
            cutoff=r_cut,
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
                u.append(uu)
                v.append(vv)
                r.append(dd)

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
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        #print(data.x)
    else:
        data.x = deg

    return data

def AtomPositions(data):
    x = data.x
    position = torch.from_numpy(data.ase.get_scaled_positions().astype(np.float32))
    position = position.view(-1,3)
    data.x = torch.cat([x, position], dim=-1)
    return data

def AtomCharges(data, data_path, cat=True):
    x = data.x
    structureid = str(data.structure_id)
    new_structureid = structureid[3:-3]
    #data_path = 'E:\MOF_design\MOF_design\MOF_graph\data\MOF_data\MOF_data'
    cif_name = os.path.join(data_path, new_structureid + ".cif")

    #print(cif_name)
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
        print(cif_name)
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

def remove_subgraph(Graph, center, percent=0.2):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes)*percent))
    removed = []
    temp = [center]
    
    while len(removed) < num:
        neighbors = []
        if len(temp) < 1:
            break

        for n in temp:
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])      
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break

        temp = list(set(neighbors))
    return G, removed

def Augmention(dataset):
    N = dataset.ase.get_global_number_of_atoms()

    M = len(dataset.edge_weight)

    


    start_i, start_j, start_k = random.sample(list(range(N)), 3)
    percent_i = random.uniform(0, 0.2)

    edges_i = []
    edge_index_i = deepcopy(dataset.edge_index.numpy())
    edge_attr_i = deepcopy(dataset.edge_attr.numpy())
    edge_weight_i = deepcopy(dataset.edge_weight.numpy())
    mask_i = deepcopy(dataset.edge_descriptor['mask'])
    distance_i = deepcopy(dataset.edge_descriptor['distance'].numpy())
    data_y_i = deepcopy(dataset.y)
    data_u_i = deepcopy(dataset.u)
    data_structure_id_i = deepcopy(dataset.structure_id)
    x_i = deepcopy(dataset.x)
    z_i = deepcopy(dataset.z)
    
    # mask_j = deepcopy(dataset.edge_descriptor['mask'])
    # x_j = deepcopy(dataset.x)
    # z_j = deepcopy(dataset.z)
    # data_y_j = deepcopy(dataset.y)
    # data_u_j = deepcopy(dataset.u)
    # data_structure_id_j = deepcopy(dataset.structure_id)
    # data_edge_index_j = deepcopy(dataset.edge_index)
    # data_edge_weight_j = deepcopy(dataset.edge_weight)
    # data_edge_attr_j =  deepcopy(dataset.edge_attr)
    # distance_j = deepcopy(dataset.edge_descriptor['distance'])
    
    # edge_index_k = deepcopy(dataset.edge_index.numpy())
    # edge_attr_k = deepcopy(dataset.edge_attr.numpy())
    # edge_weight_k = deepcopy(dataset.edge_weight.numpy())
    # distance_k = deepcopy(dataset.edge_descriptor['distance'].numpy())
    # mask_k = deepcopy(dataset.edge_descriptor['mask'])
    # data_x_k = deepcopy(dataset.x)
    # data_y_k = deepcopy(dataset.y)
    # data_u_k = deepcopy(dataset.u)
    # data_structure_id_k = deepcopy(dataset.structure_id)
    # data_z_k = deepcopy(dataset.z)
    
    for i in range(0,len(edge_index_i[0])):
        edges_i.append([edge_index_i[0][i],edge_index_i[1][i]])

    moleGraph = nx.Graph(edges_i)
    G_i, removed_i = remove_subgraph(moleGraph, start_i, percent=percent_i)
    # print(removed_i)
    atom_remain_indices_i = [i for i in range(N) if i not in removed_i]
    data_edge_index_i_0 = []
    data_edge_index_i_1 = []
    data_edge_weight_i = []
    data_edge_attr_i=[]
    data_edge_discriptor_distance_i = []
    
    for i in range(M):
        if edge_index_i[0][i] in atom_remain_indices_i and edge_index_i[1][i] in atom_remain_indices_i:
            data_edge_index_i_0.append(edge_index_i[0][i])
            data_edge_index_i_1.append(edge_index_i[1][i])
            data_edge_weight_i.append(edge_weight_i[i])
            data_edge_attr_i.append(edge_attr_i[i])
            data_edge_discriptor_distance_i.append(distance_i[i])
            
    data_edge_index_i = torch.tensor([data_edge_index_i_0,data_edge_index_i_1])
    data_edge_weight_i = torch.tensor(np.array(data_edge_weight_i))
    data_edge_attr_i =   torch.tensor(np.array(data_edge_attr_i))
    data_edge_descriptor_i = {}
    data_edge_descriptor_i['distance'] = torch.tensor(data_edge_discriptor_distance_i)

    for atom_idx in range(N):
        if atom_idx in removed_i:
            mask_i[atom_idx,:]=1
            mask_i[:,atom_idx]=1
            x_i[atom_idx,:] = 0
            z_i[atom_idx] = 0

    data_edge_descriptor_i['mask'] = mask_i
    data_x_i = x_i
    data_z_i = z_i
     
    data_i = Data(x = data_x_i, y = data_y_i, z = data_z_i, u = data_u_i, structure_id = data_structure_id_i, edge_index = data_edge_index_i, edge_weight = data_edge_weight_i, edge_attr = data_edge_attr_i, edge_descriptor = data_edge_descriptor_i)
    
    # num_mask_nodes_j= max([0, math.floor(0.25*N)])
    # mask_nodes_j = random.sample(list(range(N)), num_mask_nodes_j)

    # for atom_idj in range(N):
    #     if atom_idj in mask_nodes_j:
    #         mask_j[atom_idj,:]=1
    #         mask_j[:,atom_idj]=1
    #         x_j[atom_idj,:] = 0
    #         z_j[atom_idj] = 0
            
    # data_edge_descriptor_j = {}
    # data_edge_descriptor_j['mask'] = mask_j
    # data_x_j = x_j
    # data_z_j = z_j
    # data_edge_descriptor_j['distance'] = distance_j

    # data_j = Data(x = data_x_j, y = data_y_j, z = data_z_j, u = data_u_j, structure_id = data_structure_id_j, edge_index = data_edge_index_j, edge_weight = data_edge_weight_j, edge_attr = data_edge_attr_j, edge_descriptor = data_edge_descriptor_j)
    
    # data_edge_index_k_0=[]
    # data_edge_index_k_1=[]
    # data_edge_weight_k=[]
    # data_edge_attr_k=[]
    # data_edge_discriptor_distance_k=[]
  
    # num_mask_edges_k = max([0, math.floor(0.25*M)])
    # mask_edge_k = random.sample(list(range(M)), num_mask_edges_k)
    # for edge_idk in range(M):
    #     if edge_idk not in mask_edge_k:
    #         data_edge_index_k_0.append(edge_index_k[0][edge_idk])
    #         data_edge_index_k_1.append(edge_index_k[1][edge_idk])
    #         data_edge_weight_k.append(edge_weight_k[edge_idk])
    #         data_edge_attr_k.append(edge_attr_k[edge_idk])
    #         data_edge_discriptor_distance_k.append(distance_k[edge_idk])
    # data_edge_index_k = torch.tensor([data_edge_index_k_0,data_edge_index_k_1])
    # data_edge_weight_k = torch.tensor(data_edge_weight_k)
    # data_edge_attr_k = torch.tensor(np.array(data_edge_attr_k))
    # data_edge_descriptor_k = {}
    # data_edge_descriptor_k['distance'] = torch.tensor(data_edge_discriptor_distance_k)
    # data_edge_descriptor_k['mask'] = mask_k

    # data_k = Data(x = data_x_k, y = data_y_k, z = data_z_k, u = data_u_k, structure_id = data_structure_id_k, edge_index = data_edge_index_k, edge_weight = data_edge_weight_k, edge_attr = data_edge_attr_k, edge_descriptor = data_edge_descriptor_k)
    
  
    
    return data_i#, data_j, data_k

################################################################################
#  Transforms
################################################################################

##Get specified y index from data.y
class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data
