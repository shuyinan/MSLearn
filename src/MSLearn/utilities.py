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
# The utility module
#=============================================

import numpy as np
import re
from itertools import combinations, product, permutations
from collections import defaultdict


def parse_general_file(file_path):
    """
    Parses a general file where each data point consists of two lines:
    - First line contains features (space-separated values).
    - Second line contains the corresponding target value(s).

    The function reads the file and returns two NumPy arrays:
    - `features`: Shape (N_samples, N_features)
    - `targets`: Shape (N_samples, N_targets)

    Args:
        file_path (str): Path to the data file.

    Returns:
        tuple: (features, targets)
        - `features`: NumPy array of shape (N_samples, N_features)
        - `targets`: NumPy array of shape (N_samples, N_targets)
    """
    features = []
    targets = []
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) % 2 != 0:
        raise ValueError("File format error: Each feature line must have a corresponding target line.")
    
    for i in range(0, len(lines), 2):
        feature_values = np.array(list(map(float, lines[i].split())))
        target_values = np.array(list(map(float, lines[i+1].split())))
        
        features.append(feature_values)
        targets.append(target_values)
    
    return np.array(features), np.array(targets)


def parse_convenient_multistate_xyz(xyz_file):
    """
    Parses a multi-state XYZ database containing atomic structures and electronic state information.

    This function:
    - Reads an XYZ file with atomic positions and optional gradients.
    - Extracts the number of atoms and electronic states.
    - Stores energy values for each electronic state.
    - Stores atomic symbols and coordinates.
    - If gradients are provided, they are stored per electronic state.
    - Returns two versions of the database:
      - `database`: Contains only the provided gradient data.
      - `database_filled`: Fills missing gradients with `np.nan` for uniformity.

    Example of the convenient xyz file:
    3 3
    -100.32 -98.23 -95.67
    C   0.000   0.000   0.000
    H   0.000   0.000   1.000
    O   1.000   0.000   0.000

    Args:
        filename (str): Path to the XYZ file.

    Returns:
        tuple: (database, database_filled)
        - **database**: List of dictionaries, each containing:
          - "num_atoms": Number of atoms.
          - "num_states": Number of electronic states.
          - "energies": List of energies for each state.
          - "atoms": List of atomic symbols.
          - "coordinates": NumPy array of shape (N_atoms, 3) for XYZ coordinates.
          - "gradients": Dictionary mapping state indices to (N_atoms, 3) NumPy arrays of gradients (if available).

        - **database_filled**: Same structure as `database`, but with missing gradients replaced by `np.nan`.
    """

    database = []
    database_filled = []
    
    with open(xyz_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    i = 0
    while i < len(lines):
        num_atoms, num_states = map(int, lines[i].split())
        i += 1
        
        energies = list(map(float, lines[i].split()))
        i += 1
        
        atoms = []
        coordinates = np.zeros((num_atoms, 3))
        gradients = {}  # Store gradients only if present
        gradients_filled = {state: np.full((num_atoms, 3), np.nan) for state in range(num_states)}
        
        for j in range(num_atoms):
            parts = lines[i].split()
            if len(parts) < 4:
                raise ValueError(f"Insufficient data in atomic line {j+1}.")
            
            atoms.append(parts[0])
            coordinates[j] = np.array(parts[1:4], dtype=float)
            
            if len(parts) > 4:
                grad_data = np.array(parts[4:], dtype=float)
                if len(grad_data) == num_states * 3:
                    if not gradients:
                        gradients = {state: np.empty((num_atoms, 3)) for state in range(num_states)}
                    for state in range(num_states):
                        gradients[state][j] = grad_data[state*3:(state+1)*3]
                        gradients_filled[state][j] = grad_data[state*3:(state+1)*3]
            
            i += 1
        
        database.append({
            "num_atoms": num_atoms,
            "num_states": num_states,
            "metadata": {
                "energy": {i + 1: [energies[i]] for i in range(len(energies))}
            },
            "atoms": atoms,
            "coordinates": coordinates,
            "gradients": gradients if gradients else {}
        })
        
        database_filled.append({
            "num_atoms": num_atoms,
            "num_states": num_states,
            "metadata": {
                "energy": {i + 1: [energies[i]] for i in range(len(energies))}    
            },
            "atoms": atoms,
            "coordinates": coordinates,
            "gradients": gradients_filled
        })
    
    return database, database_filled


def parse_canonical_multistate_xyz(xyz_file):
    """
    Parses a multi-state XYZ database in canonical-style format with labeled metadata, gradients & NACs.
    
    This function:
    - Reads the XYZ file containing atomic structures and electronic state information.
    - Extracts metadata (e.g., energy, dipole, transition dipole).
    - Identifies whether metadata is **state-wise** (e.g., energy, dipole) or **state-pair-wise** (e.g., transition dipole, NAC).
    - Constructs `database` with available values.
    - Constructs `database_filled` where all missing metadata values are set to `np.nan`, ensuring a uniform data structure.

    Example of xyz_file content:
    3 3
    Metadata=energy:R:3(1,2,3):dipole:R:9(1,2,3):transition_dipole:R:9(1-2,1-3,2-3)
    Properties=species:S:1:pos:R:3:forces:R:6(1,3):nac:R:6(1-3,2-3)
    energy=-100.32 -98.23 -95.67 dipole=1.23 0.12 0.78  2.34 1.23 3.45  0.56 0.78 0.90 transition_dipole=0.01 0.02 0.03  0.04 0.05 0.06  0.07 0.08 0.09
    C   0.000   0.000   0.000   0.001  0.002  0.003   0.007  0.008  0.009   0.04 0.05 0.06   0.07 0.08 0.09
    H   0.000   0.000   1.000   0.005  -0.002  -0.001   0.004  0.005  0.006   0.05 0.06 0.07   0.08 0.09 0.10
    O   1.000   0.000   0.000   -0.002  0.003  -0.004   0.006  0.007  0.008   0.06 0.07 0.08   0.09 0.10 0.11

    first line: number of atoms, and number of states
    second line and third line: information for metadata and atomic data
    third line: metadata
    from fourth line: atomic data (element, coordinate, gradient, NAC) 

    !!!!WHAT HAS NOT BEEN DONE:
    current metadata does not work for node features. That said, the metadata are properties that related to electronic states
    it can be state-wise properties or state-pairwise properties. 
    
    Args:
        xyz_file (str): Path to the multi-geometry XYZ file.

    Returns:
        tuple: (database, database_filled)
        - **database**: List of dictionaries, each containing:
          - `"num_atoms"`: Number of atoms.
          - `"num_states"`: Number of electronic states.
          - `"atoms"`: List of atomic symbols.
          - `"coordinates"`: NumPy array of shape (N_atoms, 3) for XYZ coordinates.
          - `"metadata"`: Dictionary of extracted metadata (energies, dipoles, etc.).
          - `"metadata_labels"`: Dictionary of metadata labels for states and state-pairs.
          - `"gradients"`: Dictionary mapping state indices to (N_atoms, 3) NumPy arrays of gradients.
          - `"gradient_labels"`: List of states with available gradients.
          - `"nac"`: Dictionary mapping state-pair tuples to (N_atoms, 3) NumPy arrays of nonadiabatic couplings.
          - `"nac_labels"`: List of state-pairs with available NACs.

        - **database_filled**: Same structure as `database`, but with missing metadata values filled with `np.nan`.
    """

    database = []
    database_filled = []

    with open(xyz_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        num_atoms, num_states = map(int, lines[i].split())
        i += 1  # Move to Metadata= line

        metadata_properties_line = lines[i].replace("Metadata=", "").strip()
        i += 1  # Move to Properties= line

        properties_line = lines[i].replace("Properties=", "").strip()
        i += 1  # Move to actual metadata values

        # Parse metadata fields dynamically
        metadata_labels = {}
        metadata_statewise = {}  # True if state-wise, False if state-pair-wise
        expected_metadata_count = 0
        metadata_size_per_entry = {}
 
        metadata_matches = re.findall(r'(\w+):R:(\d+)\((.*?)\)', metadata_properties_line)
        for key, size, state_str in metadata_matches:
            size = int(size)
            indices = [
                tuple(map(int, x.split("-"))) if "-" in x else int(x)
                for x in state_str.split(",")
            ]
            num_entries = len(indices)
            metadata_labels[key] = indices
            metadata_statewise[key] = all(isinstance(x, int) for x in indices)  # True if all are single numbers
            expected_metadata_count += size
            if size % num_entries != 0:
                raise ValueError(f"Error: Metadata size {size} is not evenly divisible by the number of entries {num_entries} for key {key}.")
            metadata_size_per_entry[key] = size // num_entries

        # Parse Properties= line for gradient/NAC structure
        gradient_labels = []
        nac_labels = []
        expected_atom_value_count = 3  # Start with XYZ coordinates

        properties_matches = re.findall(r'(\w+):R:(\d+)\((.*?)\)', properties_line)
        for key, size, state_str in properties_matches:
            size = int(size)
            indices = [
                tuple(map(int, x.split("-"))) if "-" in x else int(x)
                for x in state_str.split(",")
            ]
            expected_atom_value_count += size

            if key == "forces":
                gradient_labels = indices
            elif key == "nac":
                nac_labels = indices

        # Identify all possible states and state-pairs
        metadata_states = set(range(1, num_states + 1))
        metadata_state_pairs = [pair for key in metadata_labels if not metadata_statewise[key] for pair in metadata_labels[key]]
        #metadata_state_pairs = {(s1, s2) for s1 in metadata_states for s2 in metadata_states if s1 < s2}
 
        # Fill metadata_labels in `database_filled` with all possible states/state-pairs
        metadata_labels_filled = {
            key: (list(metadata_states) if metadata_statewise[key] else list(metadata_state_pairs))
            for key in metadata_labels
        }

        # Read metadata values
        metadata = {key: {} for key in metadata_labels}
        metadata_filled = {
            key: (
                {state: np.nan for state in metadata_states} if metadata_statewise[key]
                else {pair: np.full(3, np.nan) for pair in metadata_state_pairs}
            )
            for key in metadata_labels
        }
        metadata_line = lines[i]
        i += 1  # Move to atomic data

        # Extract metadata values
        metadata_values = re.findall(r'(\S+)=([\d.\- ]+)', metadata_line)
        extracted_metadata_values = [float(v) for _, values in metadata_values for v in values.split()]

        # Error check: Ensure correct number of metadata values
        if len(extracted_metadata_values) != expected_metadata_count:
            raise ValueError(
                f"Error: Expected {expected_metadata_count} metadata values, but found {len(extracted_metadata_values)}.\n"
                f"Metadata line: {metadata_line}"
            )

        # Store metadata correctly
        index = 0
        for key in metadata_labels:
            indices = metadata_labels[key]
            num_values = metadata_size_per_entry[key]
            if metadata_statewise[key]:  # State property
                for label in indices:
                    metadata[key][label] = extracted_metadata_values[index:index + num_values]
                    metadata_filled[key][label] = extracted_metadata_values[index:index + num_values]  # Store in filled
                    index += num_values
            else:  # State-pair property
                for pair in indices:
                    metadata[key][pair] = np.array(extracted_metadata_values[index:index + num_values])
                    metadata_filled[key][pair] = np.array(extracted_metadata_values[index:index + num_values])  # Store in filled
                    index += num_values

        # Read atomic data
        atoms = []
        coordinates = np.zeros((num_atoms, 3))
        gradients = {state: np.zeros((num_atoms, 3)) for state in gradient_labels}
        gradients_filled = {state: np.full((num_atoms, 3), np.nan) for state in metadata_states}

        nac = {pair: np.zeros((num_atoms, 3)) for pair in nac_labels}
        nac_filled = {pair: np.full((num_atoms, 3), np.nan) for pair in metadata_state_pairs}

        for j in range(num_atoms):
            parts = lines[i].split()
            if len(parts) != expected_atom_value_count + 1:
                raise ValueError(
                    f"Error: Expected {expected_atom_value_count} numerical values in atomic row {j+1}, but found {len(parts)-1}.\n"
                    f"Line: {lines[i]}"
                )

            atoms.append(parts[0])
            coordinates[j] = np.array(parts[1:4], dtype=float)

            grad_start = 4
            for state in gradient_labels:
                gradients[state][j] = np.array(parts[grad_start:grad_start+3], dtype=float)
                gradients_filled[state][j] = gradients[state][j]
                grad_start += 3

            nac_start = grad_start
            for pair in nac_labels:
                nac[pair][j] = np.array(parts[nac_start:nac_start+3], dtype=float)
                nac_filled[pair][j] = nac[pair][j]
                nac_start += 3

            i += 1

        database.append({
            "num_atoms": num_atoms,
            "num_states": num_states,
            "atoms": atoms,
            "coordinates": coordinates,
            "metadata": metadata,
            "metadata_labels": metadata_labels,
            "gradients": gradients,
            "gradient_labels": gradient_labels,
            "nac": nac,
            "nac_labels": nac_labels,
        })

        database_filled.append({
            "num_atoms": num_atoms,
            "num_states": num_states,
            "atoms": atoms,
            "coordinates": coordinates,
            "metadata": metadata_filled,
            "metadata_labels": metadata_labels_filled,
            "gradients": gradients_filled,
            "gradient_labels": list(metadata_states),
            "nac": nac_filled,
            "nac_labels": list(nac_filled.keys()),
        })

    return database, database_filled


def compute_z_matrix(num_atoms, coordinates):
    """
    Computes the full Z-matrix representation from atomic coordinates.
    """
    z_matrix = []
    if num_atoms < 2:
        return np.array(z_matrix)

    # First atom: no information
    z_matrix.append([])

    # Second atom: distance to the first atom
    z_matrix.append([np.linalg.norm(coordinates[1] - coordinates[0])])

    # Third atom: distance to second, angle with 1-2
    if num_atoms > 2:
        v1 = coordinates[1] - coordinates[0]
        v2 = coordinates[2] - coordinates[1]
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 > 0 and norm_v2 > 0:
            angle = np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0))
        else:
            angle = 0.0  # Assign 0 if invalid
        z_matrix.append([np.linalg.norm(v2), np.degrees(angle)])

    # Fourth atom and beyond: distance, angle, and dihedral
    for i in range(3, num_atoms):
        v1 = coordinates[i-2] - coordinates[i-3]
        v2 = coordinates[i-1] - coordinates[i-2]
        v3 = coordinates[i] - coordinates[i-1]

        bond_length = np.linalg.norm(v3)
        norm_v2 = np.linalg.norm(v2)
        norm_v3 = np.linalg.norm(v3)
        if norm_v2 > 0 and norm_v3 > 0:
            angle = np.arccos(np.clip(np.dot(v2, v3) / (norm_v2 * norm_v3), -1.0, 1.0))
        else:
            angle = 0.0

        # Compute dihedral angle
        normal1 = np.cross(v1, v2)
        normal2 = np.cross(v2, v3)
        norm_n1 = np.linalg.norm(normal1)
        norm_n2 = np.linalg.norm(normal2)
        if norm_n1 > 0 and norm_n2 > 0:
            normal1 /= norm_n1
            normal2 /= norm_n2
            dihedral = np.arctan2(np.dot(np.cross(normal1, normal2), v2 / norm_v2), np.dot(normal1, normal2))
        else:
            dihedral = 0.0

        z_matrix.append([bond_length, np.degrees(angle), np.degrees(dihedral)])

    return np.array([item for sublist in z_matrix for item in sublist])


def compute_distances(coordinates):
    """Computes all pairwise distances from atomic coordinates."""
    num_atoms = coordinates.shape[0]
    distances = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distances.append(np.linalg.norm(coordinates[i] - coordinates[j]))
    return np.array(distances)


def compute_bond_distances_and_pips(database, pip_order):
    """
    Computes bond distances and generates permutationally invariant polynomials (PIPs) from the parsed database.

    Args:
        database (list of dicts): Parsed XYZ database.
        pip_order orders of permutationally invariant polynomials

    Returns:
        results (list of dicts): Each dict contains:
            - "bond_distances": Dictionary mapping bond types to lists of distances.
            - "pip_features": Dictionary of permutationally invariant polynomial features.
    """
    results = []
    for entry in database:
        num_atoms = entry["num_atoms"]
        atoms = entry["atoms"]
        coordinates = entry["coordinates"]

        bond_distances = {}
        for (idx1, idx2) in combinations(range(num_atoms), 2):
            atom1, atom2 = atoms[idx1], atoms[idx2]
            bond_type = tuple(sorted([atom1, atom2]))  # Ensure consistent ordering
            distance = np.linalg.norm(coordinates[idx1] - coordinates[idx2])
            if bond_type not in bond_distances:
                bond_distances[bond_type] = []
            bond_distances[bond_type].append(distance)

        pip_features = {}
        for bond_type, distances in bond_distances.items():
            distances = np.array(distances)
            pips = generate_pips(distances, pip_order)
            pip_features[bond_type] = pips

        results.append({
            "num_atoms": num_atoms,
            "atoms": atoms,
            "bond_distances": bond_distances,
            "pip_features": pip_features
        })

    return results



def generate_pips(values, max_order):
    """
    Generate all Permutationally Invariant Polynomials (PIPs) given a list of values and a required order.

    Parameters:
    values (list): List of numerical values.
    max_order (int): Maximum total order of the polynomials.

    Returns:
    list: List of unique PIP terms.
    """
    num_distances = len(values)
    pips = []

    # Iterate over all possible total orders
    for order in range(1, max_order + 1):
        monomial_dict = defaultdict(list)

        # Generate all possible exponent assignments
        for partition in product(range(order + 1), repeat=num_distances):
            if sum(partition) == order:
                # Sort partition to ensure same terms are grouped
                sorted_partition = tuple(sorted(partition, reverse=True))
                monomial_value = 1
                for i, exp in enumerate(partition):
                    monomial_value *= values[i] ** exp

                monomial_dict[sorted_partition].append(monomial_value)

        # Symmetrize by averaging over permutations
        symmetric_pips = [sum(vals) / len(vals) for vals in monomial_dict.values()]

        pips.extend(symmetric_pips)

    return pips


