#=============================================
# Multi-State Surface Learning Package - MSLP
#
# Author: Yinan Shu
# University of Minnesota, Minneapolis, Minnesota, United States
#
# MLS Package involves following methods:
#     Diabatization by Deep Neural Network (DDNN) methods: DDNN, PR-DDNN, PM-DDNN
#     Companion Matrix Neural Network (CMNN) method 
#     Compatabilization by Deep Neural Network (CDNN) methods: CDNN, PR-CDNN, PM-CDNN
#
#
# Feb. 1, 2025: Created by Yinan Shu
#
# The dataset module
#=============================================

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import permutations, product, combinations
from collections import defaultdict
from torch_geometric.data import Data

from .utilities import generate_pips, compute_z_matrix, compute_distances, compute_bond_distances_and_pips
from .atomic_data import ATOMIC_NUMBER, ATOMIC_MASS, ELECTRON_NEGATIVITY, COVALENT_RADIUS, \
                        VALENCE_ELECTRONS, IONIZATION_ENERGY, ELECTRON_AFFINITY


# ======================================
# ======================================
# Wrapper used to create dataset in main.py
# ======================================
# ======================================
def create_dataset(data, method, permutation_override=None, representation_override=None, target_key_override=None):
    """Creates dataset based on the method parameters."""
    database_style = method["database_style"]  # Get the database style directly from method
    permutation = permutation_override if permutation_override is not None else method["permutation"]
    representation = representation_override if representation_override is not None else method["mol_rep"]
    target_key = target_key_override if target_key_override is not None else method["training_target"]

    if database_style in ["xyz", "canonical_xyz"]:
        if representation == "graph":
            return XYZDataset(
                data,
                representation=representation,
                target_key=target_key,
                permutation=permutation,
                permutation_order=method["permutation_order"],
                permutation_pairs=method["permutation_pairs"],
                edge_type=method["graph_rep"]["edge_type"],
                node_features=method["graph_rep"]["node_features"],
                edge_features=method["graph_rep"]["edge_features"],
                cutoff_radius=method["graph_rep"]["cutoff_radius"]
            )
        else:
            return XYZDataset(
                data,
                representation=representation,
                target_key=target_key,
                permutation=permutation,
                permutation_order=method["permutation_order"],
                permutation_pairs=method["permutation_pairs"]
            )
    elif database_style == "general":
        features, targets = data  # Correctly unpacking the tuple
        return GeneralDataset(
            features=features,
            targets=targets,
            permutation=permutation,
            permutation_order=method["permutation_order"],
            permutation_pairs=method["permutation_pairs"]
        )
    else:
        raise ValueError(f"Unknown database_style: {database_style}")



# ======================================
# ======================================
# Combined dataset
# ======================================
# ======================================
class CombinedDataset(Dataset):
    def __init__(self, graph_dataset, R_indicator, base_potential):
        self.graph_dataset = graph_dataset
        self.R_indicator = R_indicator
        self.base_potential = base_potential

    def __getitem__(self, idx):
        return self.graph_dataset[idx], self.R_indicator[idx], self.base_potential[idx]

    def __len__(self):
        return len(self.graph_dataset)

# ======================================
# ======================================
# creating a dataset from general format
# ======================================
# ======================================
# Used to generate a dataset for general input. 
# when permutation="constrain", it generates PIPs, according to permutation_order
# when permutation="restrain", it generates permuted dataset with identity removed
# both generations are according to permutation_pairs: pairs of considered identity nuclei


class GeneralDataset(Dataset):
    def __init__(self, features, targets, permutation=False, permutation_order=1, permutation_pairs=None):
        """
        Args:
            features (numpy.ndarray or torch.Tensor): Input feature matrix of shape (N_samples, N_features)
            targets (numpy.ndarray or torch.Tensor): Target labels of shape (N_samples, N_targets) or (N_samples,)
        """
        features = np.array(features)  # Ensure it's a NumPy array
        if isinstance(permutation_pairs, list) and all(isinstance(p, int) for p in permutation_pairs):
            permutation_pairs = [permutation_pairs]  # Wrap single list in another list

        # the input indicies start from 1
        if permutation_pairs is not None and permutation_pairs != "all":
            permutation_pairs = [[idx - 1 for idx in group] for group in permutation_pairs]  # Convert to zero-based index

        if permutation == "constrain" and permutation_pairs is not None:
            if permutation_pairs == "all":
                features = self.apply_permutation_all(features, permutation_order)
            else:
                features = self.apply_permutation(features, permutation_pairs, permutation_order)
        elif permutation == "restrain" and permutation_pairs is not None:
            if permutation_pairs == "all":
                features, targets = self.generate_permuted_database_all(features, targets)
            else:
                features, targets = self.generate_permuted_database(features, targets, permutation_pairs)

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a single sample as (feature, target) tuple."""
        x = self.features[idx]  # Features (node features)
        y = self.targets[idx]   # Target property
        edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges (empty graph)
        edge_attr = torch.empty((0,))

        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        #return self.features[idx], self.targets[idx]


    def apply_permutation(self, features, permutation_pairs, max_order):
        """Applies permutation-invariant polynomial transformation to the specified feature groups."""
        new_features = []
        
        for row in features:
            transformed_row = []
            processed_indices = set()
            
            for group in permutation_pairs:
                group_values = [row[idx] for idx in group]
                transformed_values = generate_pips(group_values, max_order)
                transformed_row.extend(transformed_values)
                processed_indices.update(group)
            
            # Include features that are not part of any permutation group
            remaining_features = [row[i] for i in range(len(row)) if i not in processed_indices]
            transformed_row.extend(remaining_features)
            
            new_features.append(transformed_row)
        
        return np.array(new_features)

    def apply_permutation_all(self, features, max_order):
        """Applies permutation-invariant polynomial transformation to all features."""
        new_features = []

        for row in features:
            transformed_row = generate_pips(row, max_order)
            new_features.append(transformed_row)

        return np.array(new_features)

    def generate_permuted_database(self, features, targets, permutation_pairs):
        """Generates all possible feature permutations according to permutation_pairs and duplicates targets."""
        new_features = []
        new_targets = []
        
        for row, target in zip(features, targets):
            base_row = list(row)
            permuted_rows = set()
            
            # Generate all permutations for each group
            group_permutations = [list(permutations([base_row[idx] for idx in group])) for group in permutation_pairs]
            
            # Compute Cartesian product of all group permutations
            for permuted_values in product(*group_permutations):
                new_row = base_row[:]
                for group, perm in zip(permutation_pairs, permuted_values):
                    for i, idx in enumerate(group):
                        new_row[idx] = perm[i]
                permuted_rows.add(tuple(new_row))

            permuted_rows.discard(tuple(base_row))  # Remove the identity permutation
            
            for perm_row in permuted_rows:
                new_features.append(list(perm_row))
                new_targets.append(target)
        
        return np.array(new_features), np.array(new_targets)

    def generate_permuted_database_all(self, features, targets):
        """Generates all possible permutations of all features and removes the identity permutation."""
        new_features = []
        new_targets = []
        
        for row, target in zip(features, targets):
            base_row = list(row)
            permuted_rows = set(permutations(base_row))
            permuted_rows.discard(tuple(base_row))  # Remove the identity permutation
            
            for perm_row in permuted_rows:
                new_features.append(list(perm_row))
                new_targets.append(target)
        
        return np.array(new_features), np.array(new_targets)



# ======================================
# ======================================
# creating a dataset from XYZ format
# ======================================
# ======================================
# Used to generate a dataset for XYZ input.  
# when permutation="constrain", it generates PIPs, according to permutation_order
# when permutation="restrain", it generates permuted dataset with identity removed
# both generations are according to permutation_pairs: pairs of considered identity nuclei


class XYZDataset(Dataset):
    def __init__(self, database, representation="xyz", target_key="energy", 
            permutation=False, permutation_order=1, permutation_pairs=None, 
            edge_type="full", node_features=["atomic_number"], edge_features=["distance"], 
            cutoff_radius=5.0, custom_node_feature_fn=None):
        """
        Args:
            database (list of dicts): Parsed molecular data where each dictionary contains:
                - "coordinates" (N_atoms, 3): Atomic positions.
                - "metadata": Dictionary with target values under `target_key`.
            representation (str): Type of molecular representation ("xyz", "zmatrix", "distance").
            target_key (str): Key in metadata to extract target value (default: "energy").
            permutation (bool or str): False (no perm), "restrain" (apply perm), "constrain" (restricted perm).
            edge_type (str): Graph connectivity ("full", "cutoff", "bond").
            node_features (list): List of atomic features to use as node attributes.
            edge_features (list): List of edge features to use (e.g., ["distance", "angle", "dihedral_angle", "coulomb"]).
            cutoff_radius (float): Radius for "cutoff" edge_type.
            custom_node_feature_fn (Callable): Function for custom node features if node_features="custom".
        """
        self.features = []
        self.targets = []
        self.is_graph = (representation == "graph")
        self.node_features = node_features  # List of features
        self.edge_features = edge_features  # List of edge features

        # Compute transformed permutation pairs once based on atomic composition
        if permutation != False:
            transformed_permutation_pairs = None
            num_atoms = database[0]["num_atoms"]
            atoms = database[0]["atoms"]
            # the following transformed_permutation_pairs works for distances 
            if representation == "xyz" or representation == "zmatrix":
                transformed_permutation_pairs = XYZDataset.get_transformed_permutation_pairs(atoms, permutation_pairs)
            elif representation == "distance" or representation == "pip":
                # if we provide pairs, the transformed_permutation_pairs is computed 
                # according to new labels, otherwise, use original chemical label 
                if permutation_pairs != "all" and permutation_pairs is not None:
                    atoms_relabel = list(atoms)
                    label_map = {}
                    new_atomic_label=0
                    for pairs in permutation_pairs:
                        new_atomic_label=new_atomic_label+1
                        for elements in pairs:
                            atoms_relabel[elements-1]=f"X{new_atomic_label}"
                    for idx in label_map:
                        atoms_relabel[idx] = label_map[idx]
                    atoms = atoms_relabel
                
                bond_index=1
                bond_collector = {}
                for (idx1, idx2) in combinations(range(num_atoms), 2):
                    atom1, atom2 = atoms[idx1], atoms[idx2]
                    bond_type = tuple(sorted([atom1, atom2]))
                    if bond_type not in bond_collector:
                        bond_collector[bond_type] = []
                    bond_collector[bond_type].append(bond_index)
                    bond_index += 1
                transformed_permutation_pairs = [group for group in bond_collector.values() if len(group) > 1]
        else:
            transformed_permutation_pairs=None


        for entry in database:
            coordinates = entry["coordinates"]
            num_atoms = entry["num_atoms"]
            metadata = entry["metadata"]
            atomic_symbols = entry["atoms"]

            # Extract target values (do this once, before conditionals)
            if target_key in metadata:
                target_values = [v if isinstance(v, (int, float)) else v[0] for v in metadata[target_key].values()]
                #target_values = list(metadata[target_key].values())  # Convert to list format
            else:
                raise KeyError(f"Target key '{target_key}' not found in metadata.")

            if representation == "xyz":
                if permutation == False:
                    self.features.append(coordinates.flatten())
                    self.targets.append(target_values)
                elif permutation == "restrain":
                    permuted_coordinate_sets = XYZDataset.generate_permuted_coordinates(coordinates, transformed_permutation_pairs)
                    for permuted_coords in permuted_coordinate_sets:
                        self.features.append(permuted_coords.flatten())
                        self.targets.append(target_values)
                elif permutation == "constrain":
                    raise KeyError("for xyz representation, permutation can not be constrain")

            elif representation == "zmatrix":
                if permutation == False:
                    self.features.append(compute_z_matrix(num_atoms, coordinates))
                    self.targets.append(target_values)
                elif permutation == "restrain":
                    permuted_coordinate_sets = XYZDataset.generate_permuted_coordinates(coordinates, transformed_permutation_pairs)
                    for permuted_coords in permuted_coordinate_sets:
                        self.features.append(compute_z_matrix(num_atoms, permuted_coords))
                        self.targets.append(target_values)
                elif permutation == "constrain":
                    raise KeyError("for zmatrix representation, permutation can not be constrain")

            elif representation in ["distance", "pip"]:
                dataset = GeneralDataset(
                    features=[compute_distances(coordinates)],
                    targets=[target_values],
                    permutation=permutation,
                    permutation_order=permutation_order,
                    permutation_pairs=transformed_permutation_pairs
                )
                # Extend features and targets from GeneralDataset
                self.features.extend(dataset.features.tolist())
                self.targets.extend(dataset.targets.tolist())

            elif representation == "graph":
                self.process_graph_representation(
                        coordinates, num_atoms, atomic_symbols, target_values, 
                        edge_type, cutoff_radius, custom_node_feature_fn
                )

            else:
                raise ValueError(f"Unsupported representation '{representation}'")
                   
        # Convert to tensors
        if not self.is_graph:
            self.features = torch.tensor(np.array(self.features), dtype=torch.float32)
            self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a single sample as (feature, target) tuple."""
        if self.is_graph:
            return self.features[idx]
        # If not a graph, create a PyG Data object
        x = self.features[idx]  # Features (node features)
        y = self.targets[idx]   # Target property
        edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges (empty graph)
        edge_attr = torch.empty((0,))

        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    def process_graph_representation(self, coordinates, num_atoms, atomic_symbols, target_values, edge_type, cutoff_radius, custom_node_feature_fn):
        """Handles Graph representation with configurable connectivity, node features, and edge features."""
        
        # Collect node features based on user-selected properties
        node_feature_list = []
        for feature in self.node_features:
            if feature == "atomic_number":
                node_feature_list.append([ATOMIC_NUMBER[atom] for atom in atomic_symbols])
            elif feature == "atomic_mass":
                node_feature_list.append([ATOMIC_MASS[atom] for atom in atomic_symbols])
            elif feature == "electron_negativity":
                node_feature_list.append([ELECTRON_NEGATIVITY[atom] for atom in atomic_symbols])
            elif feature == "covalent_radius":
                node_feature_list.append([COVALENT_RADIUS[atom] for atom in atomic_symbols])
            elif feature == "valence_electrons":
                node_feature_list.append([VALENCE_ELECTRONS[atom] for atom in atomic_symbols])
            elif feature == "ionization_energy":
                node_feature_list.append([IONIZATION_ENERGY[atom] for atom in atomic_symbols])
            elif feature == "electron_affinity":
                node_feature_list.append([ELECTRON_AFFINITY[atom] for atom in atomic_symbols])
            else:
                raise ValueError(f"Unsupported node feature '{feature}'")

        # Stack features into a single tensor
        node_features = torch.tensor(np.array(node_feature_list).T, dtype=torch.float)

        # Compute edges and edge features
        edge_index, edge_attr = self.compute_graph_edges(coordinates, num_atoms, atomic_symbols, edge_type, cutoff_radius)

        # Construct PyTorch Geometric Data object
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(target_values, dtype=torch.float))
        self.features.append(graph)

    def compute_graph_edges(self, coordinates, num_atoms, atomic_symbols, edge_type, cutoff_radius):
        """Computes edges and their features for the graph representation."""
        edge_index = []
        edge_attr_list = []

        for i, j in combinations(range(num_atoms), 2):
            distance = torch.norm(torch.tensor(coordinates[i]) - torch.tensor(coordinates[j]))


            edge_type = edge_type[0]
            if edge_type == "full" or (edge_type=="cutoff" and distance < cutoff_radius):
                edge_index.append([i, j])
                edge_index.append([j, i])

                # Compute edge features based on user selection
                edge_feature_values = []
                for feature in self.edge_features:
                    if feature == "distance":
                        edge_feature_values.append(distance)
                    elif feature == "angle":
                        edge_feature_values.append(self.compute_angle(coordinates, i, j))
                    elif feature == "dihedral_angle":
                        edge_feature_values.append(self.compute_dihedral_angle(coordinates, i, j))
                    elif feature == "coulomb":
                        edge_feature_values.append(self.compute_coulomb_interaction(coordinates, atomic_symbols, i, j))
                    else:
                        raise ValueError(f"Unsupported edge feature '{feature}'")

                edge_attr_list.append(edge_feature_values)
                edge_attr_list.append(edge_feature_values)  # Symmetric edges

        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float)

        return edge_index, edge_attr

    def compute_angle(self, coordinates, i, j):
        """Computes bond angle using nearest third atom."""
        neighbors = [k for k in range(len(coordinates)) if k not in (i, j)]
        if not neighbors:
            return 0.0  # No valid angle

        k = min(neighbors, key=lambda n: np.linalg.norm(coordinates[j] - coordinates[n]))  # Closest atom to j

        v1 = coordinates[i] - coordinates[j]
        v2 = coordinates[k] - coordinates[j]

        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Ensure within valid range

        return np.degrees(angle)

    def compute_dihedral_angle(self, coordinates, i, j):
        """Computes dihedral angle using nearest two atoms."""
        neighbors = [k for k in range(len(coordinates)) if k not in (i, j)]
        if len(neighbors) < 2:
            return 0.0  # No valid dihedral

        k, l = sorted(neighbors[:2], key=lambda n: np.linalg.norm(coordinates[j] - coordinates[n]))  # Two closest atoms

        b1 = coordinates[k] - coordinates[i]
        b2 = coordinates[j] - coordinates[k]
        b3 = coordinates[l] - coordinates[j]

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)

        m1 = np.cross(n1, n2)

        x = np.dot(n1, n2)
        y = np.dot(m1, b2 / np.linalg.norm(b2))

        dihedral = np.arctan2(y, x)

        return np.degrees(dihedral)

    @staticmethod
    def generate_permuted_coordinates(coordinates, permutation_pairs):
        """
        Generate all possible coordinate sets by permuting identical nuclei, excluding the identity permutation.
    
        Parameters:
            coordinates (numpy.ndarray): Original molecular coordinates (shape: N x 3).
            permutation_pairs (list of lists): List of groups of identical nuclei (1-based indexing).
    
        Returns:
            list of numpy.ndarray: List of permuted coordinate sets.
        """
        num_atoms = len(coordinates)
        permuted_coordinates_list = []

        # Generate all possible non-identity permutations for each group
        permutation_options = []
        for group in permutation_pairs:
            original_order = tuple(group)  # Convert to tuple for comparison
            permutations_list = [perm for perm in permutations(group) if perm != original_order]
            permutation_options.append(permutations_list)

        # Generate Cartesian product of all non-identity permutations across different groups
        for permuted_groups in product(*permutation_options):
            permuted_coords = np.copy(coordinates)  # Start with original coordinates
        
            for group, permuted_group in zip(permutation_pairs, permuted_groups):
                # Apply permutation (convert 1-based index to 0-based)
                for original_idx, new_idx in zip(group, permuted_group):
                    permuted_coords[original_idx - 1] = coordinates[new_idx - 1]

            permuted_coordinates_list.append(permuted_coords)

        return permuted_coordinates_list

    def compute_coulomb_interaction(self, coordinates, atomic_symbols, i, j):
        """Approximates Coulomb interaction using electronegativity-based charge estimates."""
        r_ij = np.linalg.norm(coordinates[i] - coordinates[j])
        
        # compute the average electron negativity over a whole molecule
        chi_i = self.compute_weighted_average_electronegativity(coordinates, atomic_symbols, i)
        chi_j = self.compute_weighted_average_electronegativity(coordinates, atomic_symbols, j)

        q_i = ELECTRON_NEGATIVITY[atomic_symbols[i]] - chi_i
        q_j = ELECTRON_NEGATIVITY[atomic_symbols[j]] - chi_j

        return q_i * q_j / r_ij if r_ij > 0 else 0.0

    def compute_weighted_average_electronegativity(self, coordinates, atomic_symbols, i):
        """Computes a spatially weighted electronegativity for atom i."""
        chi_values = np.array([ELECTRON_NEGATIVITY[atom] for atom in atomic_symbols])
        distances = np.array([np.linalg.norm(coordinates[i] - coordinates[j]) for j in range(len(coordinates)) if j != i])

        # Define weight as inverse distance
        weights = 1 / (distances + 1e-6)  # Avoid division by zero
        weights /= np.sum(weights)  # Normalize weights

        return np.sum(weights * chi_values[np.arange(len(chi_values)) != i])


    @staticmethod
    def get_transformed_permutation_pairs(atomic_labels, permutation_pairs):
        """
        Transforms permutation_pairs to group identical atoms if 'all' is specified.

        Parameters:
            atomic_labels (list): List of atomic labels (e.g., ['C', 'H', 'H']).
            permutation_pairs (str or list of lists): 'all' or manually specified permutation groups.

        Returns:
            list of lists: Groups of identical atoms (1-based indexing).
        """
        if permutation_pairs == "all":
            # Group atoms by label
            atom_groups = defaultdict(list)
            for idx, label in enumerate(atomic_labels):
                atom_groups[label].append(idx + 1)  # Convert 0-based index to 1-based
        
            # Remove groups with only one atom (no permutation needed)
            transformed_permutation_pairs = [group for group in atom_groups.values() if len(group) > 1]
        else:
            transformed_permutation_pairs = permutation_pairs
    
        return transformed_permutation_pairs

