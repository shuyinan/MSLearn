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
# The main module
#=============================================
import yaml
import json
import torch
import random
import numpy as np
import os
import ast
import copy
import argparse
import datetime
import csv

from .utilities import parse_canonical_multistate_xyz
from .utilities import parse_convenient_multistate_xyz
from .utilities import parse_general_file

from .dataset import GeneralDataset
from .dataset import XYZDataset
from .dataset import create_dataset

from .models import MSLP
from .train import MSLPtrain

# Set a fixed seed for reproducibility
def set_seed(seed=1257394):
    """Set the random seed for full reproducibility across all frameworks."""
    torch.manual_seed(seed)                      # Set seed for PyTorch
    torch.cuda.manual_seed_all(seed)             # Set seed for CUDA (if using GPUs)
    np.random.seed(seed)                         # Set seed for NumPy
    random.seed(seed)                            # Set seed for Python's random module

    # Ensures deterministic behavior in cuDNN (useful for GPUs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disables auto-tuning for convolution layers

    # Additional stability for newer PyTorch versions (optional)
    torch.use_deterministic_algorithms(True, warn_only=True)


def deep_update_dict(defaults, user_input):
    for key, value in user_input.items():
        if isinstance(value, dict) and key in defaults and isinstance(defaults[key], dict):
            deep_update_dict(defaults[key], value)
        else:
            defaults[key] = value  


# Convert human-friendly user input to structured config
def parse_user_input(input_file):
    """ Reads user input from a plain text file and converts it into a structured dictionary. """

    VALID_OPTIONS = {
        "database_style": {"general", "xyz", "canonical_xyz"},
        "architecture": {"nn", "mpnn"},
        "optimizer_type": {"adam", "sgd"},
        "graph_rep": {
            "edge_type": {"full", "cutoff"},
            "node_features": {"atomic_number", "atomic_mass", "electron_negativity",
                          "covalent_radius", "valence_electrons", "ionization_energy",
                          "electron_affinity"},
            "edge_features": {"distance", "angle", "dihedral_angle", "coulomb"},
        },
        "loss_function": {"mse", "mae"},
        "scheduler_type": {"step", "cosine"},
        "activation": {"relu", "gelu", "sigmoid", "tanh", "leakyrelu"},
        "mpnn_structure": {"linear", "nonlinear"},
        "base_potential_type": {"atom+diatom"},
    }

    #============================
    #============================
    # METHOD DEFAULT
    #============================
    #============================

    method = {
	# general
	"database_style": "general",
	"method": "CDNN",
	"permutation": False,
        "permutation_order": 3,
	"permutation_pairs": None,
	"mol_rep": None,
        "graph_rep": {
            "edge_type": "full",
            "node_features": ["atomic_number"],
            "edge_features": ["distance"],
            "cutoff_radius": 5.0,
        },
        "architecture": "nn",
        "random_seed": 12379,

        # actual databases, not names, but real databases
        "train_dataset": None,
        "diabatic_dataset": None,
        "permutation_dataset": None,

	# training
	"database": None,
        "training_target": "energy",
	"training_percent": 80,
	"batch_size": 32,
	"diabatic_restrain": False,
	"diabatic_database": None,
	"diabatic_weight": 1.0,
	"permutation_restrain": False,
        "permutation_database": None,
        "permutation_weight": 1.0,
        "topology_attention": False,
        "topology_attention_weight": None,
        "regularization": True,
        "regularization_weight": 1e-7,
        "gradient": False,
        "gradient_weight": 1.0,
        "nac": False,
        "nac_weight": 1e-3, 
        "num_epochs": 10000,
        "learning_rate": 1e-3,
        "optimizer_type": "adam",
        "scheduler_type": "cosine",
        "scheduler_frequency": 30,
        "scheduler_scale": 0.5, 
        "loss_threshold": 0.002,
        "early_stopping_patience": 10,
        "early_stopping_delta": None,
        "loss_function": "mse",
        "checkpoint_path": "checkpoint.pth",
        "print_interval": 100,
        "save_interval": 10,

        #network detail
        "input_dim": None,
        "output_dim": None,
        "hidden_layers": [256, 128, 64],
        "activation": "gelu",
        "matrix_type": None,

        # MPNN
        "mpnn_structure": "linear",
        "message_passing_network": [128, 64],
        "update_network": [64, 32],
        "edge_attr_dim": None,

        # parametrically managed activation functon
        "parametric_mode": False,
        "PM_config": {
            "base_potential_type": None,
            # atom+diatom pm
            "R_indicator": None,
            "R_indicator_train_dataset": None,
            "R_indicator_permutation_dataset": None,
            "pm": None,
            "base_potential": None,
            "base_potential_train_dataset": None,
            "base_potential_permutation_dataset": None,
        }
    }

    #============================
    #============================
    # READ INPUT
    #============================
    #============================

    print("=====================================================")
    print(" Start parsing input")
    print("=====================================================")

    file_extension = os.path.splitext(input_file)[-1].lower()

    if file_extension in {".yaml", ".yml"}:
        # ============================
        # parse yaml file
        # ============================
        try:
            with open(input_file, "r") as file:
                user_input = yaml.safe_load(file) or {}
            deep_update_dict(method, user_input)
            print(f"Successfully loaded YAML file: {input_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    else:
        # ============================
        # parse plain text 
        # ============================
        try:
            with open(input_file, "r") as file:
                lines = [line.strip() for line in file if line.strip() and not line.startswith("#")]

            for line in lines:
                key, value = map(str.strip, line.split(":", 1))

                #============================
                # general considerations:
                #============================

                # what kind of file is reading:
                # canonical_xyz, xyz (which is convenient xyz), or general
                if key == "database_style":
                    method["database_style"] = value
                # the method will use: 
                # DDNN, CMNN, or CDNN
                elif key == "method":
                    method["method"] = value
                # how to deal with permutation:
                # no, constrian (which uses PIP for regular NN, or a graph), restrain (involves a cost function term)
                elif key == "permutation":
                    method["permutation"] = value
                elif key == "permutation_order":
                    method["permutation_order"] = value
                elif key == "permutation_pairs":
                    value = value.strip()
                    if value.lower() == "all":  # Case-insensitive check for "all"
                        method["permutation_pairs"] = "all"  # Placeholder; actual elements should be provided elsewhere
                    else:
                        try:
                            parsed_value = eval(value)
                            if isinstance(parsed_value, list) and all(isinstance(item, (list, tuple)) for item in parsed_value):
                                method["permutation_pairs"] = [list(item) for item in parsed_value]  # Convert tuples to lists
                            else:
                                raise ValueError
                        except Exception:
                            raise ValueError(f"Invalid format for 'permutation_pairs': {value}")
                # how to represent the molecule: 
                # graph, distance, pip, zmatrix, xyz, general
                elif key == "mol_rep" or key == "molecular_representation":
                    method["mol_rep"] = value

                # graph representation keywords
                elif key == "edge_type":
                    method["graph_rep"]["edge_type"] = value
                elif key == "node_features":
                    if isinstance(value, str):
                        method["graph_rep"]["node_features"] = [feature.strip() for feature in value.split(",")]
                    elif isinstance(value, list):
                        method["graph_rep"]["node_features"] = value
                elif key == "edge_features":
                    if isinstance(value, str):
                        method["graph_rep"]["edge_features"] = [feature.strip() for feature in value.split(",")]
                    elif isinstance(value, list):
                        method["graph_rep"]["edge_features"] = value
                elif key == "cutoff_radius":
                    method["graph_rep"]["cutoff_radius"] = value

                elif key == "architecture":
                    method["architecture"] = value.lower()
        
                elif key == "random_seed":
                    method["random_seed"] = value

                #============================
                # training
                #============================
                # file of main database
                elif key == "database": 
                    method["database"] = value
                elif key == "training_target":
                    method["training_target"] = value
                # percentage of training set, rest will be validation
                elif key == "training_percent":
                    method["training_percent"] = value
                # batch size
                elif key == "batch_size":
                    method["batch_size"] = value
                # ====
                # diabatic restraint
                elif key == "diabatic_restrain":
                    method["diabatic_restrain"] = value
                # diabatic database if DDNN
                elif key == "diabatic_database":
                    method["diabatic_database"] = value
                # diabatic restraint weight
                elif key == "diabatic_weight":
                    method["diabatic_weight"] = value
                # ====
                # permutation restrain
                elif key == "permutation_restrain":
                    method["permutation_restrain"] = value
                elif key == "permutation_database":
                    method["permutation_database"] = value
                    print("""Warning: We do not recommend provide permutation_database manually, 
                             it should be automatically generated according to permutation_order,
                             and permutation_pairs. 
                    """)
                # permutation restraint weight
                elif key == "permutation_weight":
                    method["permutation_weight"] = value
                # ====
                # topology attention
                elif key == "topology_attention":
                    method["topology_attention"] = value
                elif key == "topology_attention_weight":
                    method["topology_attention_weight"] = value
                # ====
                # regularization term
                elif key == "regularization":
                    method["regularization"] = value
                elif key == "regularization_weight":
                    method["regularization_weight"] = value

                # ===
                elif key == "gradient":
                    method["gradient"] = value
                elif key == "gradient_weight":
                    method["gradient_weight"] = value
                elif key == "nac":
                    method["nac"] = value
                elif key == "nac_weight":
                    method["nac_weight"] = value

                # ====
                # optimizer hyper parameters
                elif key == "num_epochs":
                    method["num_epochs"] = value
                elif key == "learning_rate":
                    method["learning_rate"] = value
                elif key == "optimizer_type":
                    method["optimizer_type"] = value.lower()
                elif key == "scheduler_type":
                    method["scheduler_type"] = value.lower()
                elif key == "scheduler_frequency":
                    method["scheduler_frequency"] = value
                elif key == "scheduler_scale":
                    method["scheduler_scale"] = value
                elif key == "loss_threshold":
                    method["loss_threshold"] = value
                elif key == "early_stopping_patience":
                    method["early_stopping_patience"] = value
                elif key == "early_stopping_delta":
                    method["early_stopping_delta"] = value
                elif key == "loss_function":
                    method["loss_function"] = value.lower()
                elif key == "checkpoint_path":
                    method["checkpoint_path"] = value
                elif key == "print_interval":
                    method["print_interval"] = value
                elif key == "save_interval":
                    method["save_interval"] = value
                # ====
                # network detail
                elif key == "input_dim":
                    method["input_dim"] = value
                elif key == "output_dim":
                    method["output_dim"] = value
                elif key == "hidden_layers":
                    method["hidden_layers"] = parse_list(value, key, dtype=int)
                elif key == "activation":
                    method["activation"] = value.lower()
                elif key == "matrix_type":
                    method["matrix_type"] = value

                # MPNN related
                elif key == "mpnn_structure":
                    method["mpnn_structure"] = value
                elif key in {"message_passing_network", "msg_nn"}:
                    method["message_passing_network"] = parse_list(value, key, dtype=int)
                elif key in {"update_network", "update_nn"}:
                    method["update_network"] = parse_list(value, key, dtype=int)
                elif key == "edge_attr_dim":
                    method["edge_attr_dim"] = value

                # parametrically managed activation function 
                elif key == "parametric_mode":
                    method["parametric_mode"] = value
                elif key == "base_potential_type":
                    method["PM_config"]["base_potential_type"] = value
                elif key == "R_indicator":
                    method["PM_config"]["R_indicator"] = value
                elif key == "pm":
                    method["PM_config"]["pm"] = parse_list(value, key, dtype=float)
                elif key == "base_potential":
                    method["PM_config"]["base_potential"] = value

                # actual databases should not be an input
                elif key == "train_dataset":
                    raise ValueError(f"Error, train_dataset should not be an input")
                elif key == "diabatic_dataset":
                    raise ValueError(f"Error, diabatic_dataset should not be an input")
                elif key == "permutation_dataset":
                    raise ValueError(f"Error, permutation_dataset should not be an input")
                else:
                    print(f"Warning: Unrecognized parameter '{key}'")
        except Exception as e:
            raise ValueError(f"Error parsing plain text file: {e}")



    #============================
    #============================
    # VALIDATE KEYS and CHECK CONSISTENTCY
    #============================
    #============================

    print("Start validating keywords and checking potential conflict")

    # check valid keys
    validate_method_config(method, VALID_OPTIONS)

    # ================================
    # yaml does not support scientifc numbers
    # we here make sure they have the correct type
    # ================================

    method["graph_rep"]["cutoff_radius"] = float(method["graph_rep"]["cutoff_radius"])
    method["learning_rate"] = float(method["learning_rate"])
    method["diabatic_weight"] = float(method["diabatic_weight"])
    method["permutation_weight"] = float(method["permutation_weight"])
    if method["topology_attention_weight"] is not None:
        method["topology_attention_weight"] = float(method["topology_attention_weight"])
    method["regularization_weight"] = float(method["regularization_weight"])
    method["gradient_weight"] = float(method["gradient_weight"])
    method["nac_weight"] = float(method["nac_weight"])
    method["learning_rate"] = float(method["learning_rate"])
    method["loss_threshold"] = float(method["loss_threshold"])
    if method["early_stopping_delta"] is not None:
        method["early_stopping_delta"] = float(method["early_stopping_delta"])


    # check, and set conditional changes
    # ================================
    # for "permutation_pairs"
    # ================================
    if method["permutation"] != False and method["permutation_pairs"] == None:
        method["permutation_pairs"] = "all"


    # ================================
    # restore graph_rep in case it was accidentally removed.
    # ================================
    if method["graph_rep"] is None or not isinstance(method["graph_rep"], dict):
        print("Warning: Resetting 'graph_rep' to default.")
        method["graph_rep"] = {
            "edge_type": "full",
            "node_features": ["atomic_number"],
            "edge_features": ["distance"],
            "cutoff_radius": 5.0
        }


    # ================================
    # for "database_style"
    # ================================
    if method["gradient"] == True:
        if method["database_style"] != "canonical_xyz":
            raise ValueError(f"Error: gradient cost term requires the database style to be canonical_xyz")

    if method["nac"] == True:
        if method["database_style"] != "canonical_xyz":
            raise ValueError(f"Error: gradient cost term requires the database style to be canonical_xyz")

    # ================================
    # for "mol_rep"
    # ================================
    # print out warnings if mol_rep has been set in input
    if method["database_style"] == "general" and method["mol_rep"] != "general":
        print("""Warning: The provided database is in a general style, set the molecular
                 representation to general.""")
        method["mol_rep"] = "general"
    if method["mol_rep"] == "xyz":
        print("""Warning: using xyz as a molecular representation to learn properties may
                 lack of translational, rotational, and permutational invariance.""")
    if method["mol_rep"] == "zmatrix":
        print("""Warning: using Z-matrix as a molecular representation to learn properties may
                 lack of permutational invariance.""")
    if method["mol_rep"] == "general":
        print("Using User defined general representation.")

    if method["architecture"] == "mpnn" and method["mol_rep"] != "graph":
        print("""Warning: when using message-passing neural network, the molecular representation
                 has to be graph, change mol_rep to "graph".""")
        method["mol_rep"] = "graph"

    # set default for mol_rep if it was not set in input
    if method["mol_rep"] == None:
        if method["architecture"] == "nn":
            if method["database_style"] == "general":
                method["mol_rep"] = "general"
            elif method["database_style"] == "xyz" and method["permutation"] == "constrain":
                method["mol_rep"] = "pip"
            elif method["database_style"] == "canonical_xyz" and method["permutation"] == "constrain":
                method["mol_rep"] = "pip"
            else:
                method["mol_rep"] = "distance"
        elif method["architecture"] == "mpnn":
            method["mol_rep"] = "graph"

    if method["permutation"] == False or method["permutation"] == "restrain":
        if method["mol_rep"] == "pip" or method["mol_rep"] == "graph":
            print("""Warning: use permutation invariant polynomials (PIPs) as molecular
                     representation, automatically set permutation to "constrain".""")
            method["permutation"] = "constrain"

    if method["permutation"] == "constrain":
        if method["mol_rep"] != "pip" or method["mol_rep"] != "graph":
            print("""Warning: permutation is required to be constrained,
                     molecular representation can currently ONLY be pip or graph,
                     change molecular representation to permutation invariant polynomials
                     of distances.""")
            method["repsentation"] = "pip"


    # ================================
    # for "architecture"
    # ================================

    if method["mol_rep"] == "graph" and method["architecture"] != "mpnn":
        print("""Warning: when using graph representation, the architecture has to be 
                 message-passing neural network, change architecture to "mpnn".""")
        method["architecture"] = "mpnn"


    # ================================
    # Up to this point, conflicst between fundamental commends check are DONE
    # The following are strict rules:
    # 1. If one uses "database_style" = "general", molecular representation has to be "general"
    # 2. If one uses "architecture" = "mpnn", molecular representation has to be "graph"
    # 3. If one sets ""permutation" to "constrain", molecular representation has to be "pip" or "graph"
    #    and vice versa.
    # ================================

    # ================================
    # for "database"
    # ================================
    if method["database"] == None:
        raise ValueError("Error: main training database must be provided")

    # ================================
    # for diabatic cost term
    # ================================
    if method["method"] == "DDNN" and method["diabatic_restrain"] == False:
        print("""Warning: using diabatization by deep neural network,
                 set diabatic_restrain to True.""")
        method["diabatic_restrain"] = True
    if method["diabatic_restrain"] == True  and method["diabatic_database"] == None:
        raise ValueError("Error: 'diabatic_database' must be specified when using diabatic_restrain term.")


    # ================================
    # for permutation cost term
    # ================================
    if method["permutation"] == "restrain" and method["permutation_restrain"] == False:
        print("Warning: using restrained permutation, set permutation_restrain to True.")
        method["permutation_restrain"] = True
    if method["permutation"] == "constrain" and method["permutation_restrain"] == True:
        print("""Warning: permutation invariance is set to be constrain, i.e. use
                 permutation invariant polynomials (PIPs) as molecular representation,
                 meanwhile, one requires using permutation restraint term in cost function,
                 this is a duplication. We will keep molecular representation as PIPs,
                 the permutation restraint term will be ignored!""")
        method["permutation_restrain"] = False
    if method["permutation"] == "restrain":
        print("permutation database will be generated according to permutation_pairs")
    if method["permutation_restrain"] == True and method["permutation"] == False:
        raise ValueError("""Error: in fundamental comment, "permutation" is set to false,
                            but "permutation_restrain" is set to true. This is a big conflict.
                            please check if permutation should be considered. If yes, please set
                            "permutation" to "restrain" or "constrain".""")

    # ================================
    # for topology attention
    # ================================
    if method["topology_attention_weight"] == None:
        method["topology_attention_weight"] = 0.2*method["loss_threshold"]

    # ================================
    # for optimizer
    # ================================
    if method["early_stopping_delta"] == None:
        method["early_stopping_delta"] = max(1e-4, 0.1*method["loss_threshold"])

    # ================================
    # Now deal with databases
    # ================================
    # ================================
    # 1. Read main database
    # ================================
    # xyz: convenient xyz; canonical_xyz: canonical xyz; general: general
    # now the following is a little bit confusing:
    # in either XYZDataset or GeneralDataset functions, it will generate a database
    # if permutation=method["permutation"] is "constrain", i.e. uses PIPs,
    #   the generated database uses PIP as features
    # if permutation=method["permutation"] is False, obviously, nothing more is to be done
    # if permutation=method["permutation"] is "restrain", the generated database is in fact
    #   the permutation database, i.e. permuted features with duplicated targets.
    # THEREFORE: here we are doing main databases, and we need to use the following if statement.

    training_set = method["database"]

    # checking the database_style
    parsed_data = {}
    autoparsing_happened = False

    try:
        database, database_filled = parse_convenient_multistate_xyz(training_set)
        parsed_data["xyz"] = database_filled
    except Exception as e:
        parsed_data["xyz"] = None

    try:
        database, database_filled = parse_canonical_multistate_xyz(training_set)
        parsed_data["canonical_xyz"] = database_filled
    except Exception as e:
        parsed_data["canonical_xyz"] = None

    try:
        features, targets = parse_general_file(training_set)
        parsed_data["general"] = (features, targets)
    except Exception as e:
        parsed_data["general"] = None

    # Identify the correct parsing method
    if method["database_style"] in parsed_data and parsed_data[method["database_style"]] is not None:
        dataset_input = parsed_data[method["database_style"]]
    else:
        # Warn and auto-correct if possible
        for style, data in parsed_data.items():
            if data is not None:
                print(f"Warning: Provided 'database_style' ({method['database_style']}) is incorrect. "
                      f"!!!Automatically using '{style}' instead.")
                dataset_input = data
                method["database_style"] = style
                autoparsing_happened = True
                break
        else:
            raise ValueError("Error: Could not parse the training set with any available method.")

    """
    the following code has been replaced with the automatic detection:

    if method["database_style"] == "xyz":
        database, database_filled = parse_convenient_multistate_xyz(training_set)
        dataset_input = database_filled
    elif method["database_style"] == "canonical_xyz":
        database, database_filled = parse_canonical_multistate_xyz(training_set)
        dataset_input = database_filled
    elif method["database_style"] == "general":
        features, targets = parse_general_file(training_set)
        dataset_input = (features, targets)
    """


    # Create main train dataset
    train_dataset = create_dataset(
        dataset_input,
        method,
        permutation_override=(False if method["permutation"] == "restrain" else None)
    )

    method["train_dataset"] = copy.deepcopy(train_dataset)


    # if gradient and/or nacs are included, check if they exist
    if method["gradient"] == True:
        if not database_filled[0].get("gradients"):
            raise ValueError(f"Error: there is no gradient information provided, but gradient cost function is required")
    if method["nac"] == True:
        if not database_filled[0].get("nac"):
            raise ValueError(f"Error: there is no NAC information provided, but nac cost function is required")

    # ================================
    # 2. Read diabatic database
    # ================================

    if method["diabatic_restrain"] == True:
        diabatic_set = method["diabatic_database"]

        # Attempt all parsing methods and validate consistency
        diabatic_parsed_data = {}

        try:
            diabatic_database, diabatic_database_filled = parse_convenient_multistate_xyz(diabatic_set)
            diabatic_parsed_data["xyz"] = diabatic_database_filled
        except Exception as e:
            diabatic_parsed_data["xyz"] = None

        try:
            diabatic_database, diabatic_database_filled = parse_canonical_multistate_xyz(diabatic_set)
            diabatic_parsed_data["canonical_xyz"] = diabatic_database_filled
        except Exception as e:
            diabatic_parsed_data["canonical_xyz"] = None

        try:
            diabatic_features, diabatic_targets = parse_general_file(diabatic_set)
            diabatic_parsed_data["general"] = (diabatic_features, diabatic_targets)
        except Exception as e:
            diabatic_parsed_data["general"] = None

        # Identify the correct parsing method
        if method["database_style"] in diabatic_parsed_data and diabatic_parsed_data[method["database_style"]] is not None:
            diabatic_input = diabatic_parsed_data[method["database_style"]]
        else:
            if autoparsing_happened is True:
                raise ValueError(
                    "Error: The 'database_style' was already corrected once for the main database. "
                    "Now it is also inconsistent for the diabatic database. Please manually check "
                    "and ensure the correct style is used for each database."
                )

            # Warn and auto-correct if possible
            for style, data in diabatic_parsed_data.items():
                if data is not None:
                    print(f"Warning: Provided 'database_style' ({method['database_style']}) is incorrect for diabatic database. "
                          f"Automatically using '{style}' instead.")
                    diabatic_input = data
                    method["database_style"] = style
                    break
            else:
                raise ValueError("Error: Could not parse the diabatic training set with any available method.")

        """
        the following code has been replaced with the automatic detection:


        if method["database_style"] == "xyz":
            diabatic_database, diabatic_database_filled = parse_convenient_multistate_xyz(diabatic_set)
            diabatic_input = diabatic_database_filled
        elif method["database_style"] == "canonical_xyz":
            diabatic_database, diabatic_database_filled = parse_canonical_multistate_xyz(diabatic_set)
            diabatic_input = diabatic_database_filled
        elif method["database_style"] == "general":
            diabatic_features, diabatic_targets = parse_general_file(diabatic_set)
            diabatic_input = (diabatic_features, diabatic_targets)  # Explicitly passing as tuple

        """

        # for diabatic database, we do not want to provide permuted ones.
        diabatic_dataset = create_dataset(
            diabatic_input, 
            method,
            permutation_override=(False if method["permutation"] == "restrain" else None)
        )
        method["diabatic_dataset"] = copy.deepcopy(diabatic_dataset)

    # ================================
    # 3. Generating permutation databases
    # ================================
    if method["permutation_restrain"] == True:
        if method["permutation"] != "restrain":
            raise ValueError("""Error: when permutation_restrain is true,
                                permutation must be "restrain".""")
        permutation_dataset = create_dataset(dataset_input, method)
        method["permutation_dataset"] = copy.deepcopy(permutation_dataset)
        print(f"permutationa database has been generated")
        print(f"original database has", len(method["train_dataset"]), "datapoints")
        print(f"permuted database has", len(method["permutation_dataset"]), "datapoints")

    # ================================
    # Network details
    # ================================

    # input dimension
    actual_input_dim = train_dataset[0]['x'].shape[-1]
    if method["input_dim"]is None:
        method["input_dim"] = actual_input_dim
        print("input feature automatically detected as", actual_input_dim)
    if method["input_dim"] != actual_input_dim:
        raise ValueError(f"Mismatch: Expected input_dim={method['input_dim']}, but dataset has {actual_input_dim} features.")

    # output dimension
    actual_output_dim = train_dataset[0]['y'].shape[-1]
    if method["output_dim"] is None:
        method["output_dim"] = actual_output_dim
        print("output target automatically detected as", actual_output_dim)
    if method["output_dim"] != actual_output_dim:
        raise ValueError(f"Mismatch: Expected output_dim={method['output_dim']}, but dataset has {actual_output_dim} targets.")

    # for MPNN, edge_attr_dim is the number of edge features
    actual_edge_attr_dim = train_dataset[0]['edge_attr'].shape[-1]
    if method["edge_attr_dim"] is None:
        method["edge_attr_dim"] = actual_edge_attr_dim
        print("edge feature automatically detected as", actual_edge_attr_dim)
    if method["edge_attr_dim"] != actual_edge_attr_dim:
        raise ValueError(f"Mismatch in MPNN: Expected edge_attr_dim={method['edge_attr_dim']}, but dataset has {actual_edge_attr_dim} edge features.")

    # for diabatic cost function term
    if method["diabatic_restrain"] == True:
        actual_diabatic_unique_elements = train_dataset[0]['y'].shape[-1]
        expected_diabatic_unique_elements = int(actual_output_dim*(actual_output_dim+1)/2)
        if actual_diabatic_unique_elements != expected_diabatic_unique_elements:
            raise ValueError(f"""Mismatch: Expected number of unique diabatic matrix elements={expected_diabatic_unique_elements},
                                 but diabatic dataset has {actual_diabatic_unique_elements} targets.""")

    if method["matrix_type"] is None:
        method["matrix_type"] = 1 + (method["output_dim"] - 1) * 2

    if method["matrix_type"] % 2 == 0 or method["matrix_type"] < 1 or method["matrix_type"] > 1 + (method["output_dim"]- 1) * 2:
        raise ValueError(f"matrix_type must be an odd positive integer between 1 and {method["output_dim"] - 1} (default: {1 + (method["output_dim"] - 1) * 2}).")

    # ================================
    # parametrically managed activation function
    # ================================

    # currently only works for base_potential_type="atom+diatom"
    if method["parametric_mode"] == True:
        if method["PM_config"]["base_potential_type"] is None:
            method["PM_config"]["base_potential_type"] = "atom+diatom"
            print("the currently only supported base_potential_type is atom+diatom")

        # following is the convention for R_indicator which is basically distance matrix
        # 1. if the user uses "xyz" or "canonical_xyz" database_style, R_indicator is computed as distance tensor
        # 2. if the user uses "general" database_style, we assume that the features in "general" database is distance,
        #    and a warning will be printed 
        if method["PM_config"]["R_indicator"] is None:
            if method["database_style"] in ["xyz", "canonical_xyz"]:
                # for main database:
                pm_main_distance_dataset = create_dataset(
                    dataset_input, 
                    method,
                    permutation_override=(False if method["permutation"] == "restrain" else None),
                    representation_override="distance"
                ) 
                method["PM_config"]["R_indicator_train_dataset"] = copy.deepcopy(
                    torch.stack([data['x'] if isinstance(data['x'], torch.Tensor) else torch.tensor(data['x'])
                    for data in pm_main_distance_dataset])
                )
                # for permutation database:
                if method["permutation_restrain"] == True:
                    pm_permutation_distance_dataset = create_dataset(
                        dataset_input,
                        method,
                        representation_override="distance"
                    )
                    method["PM_config"]["R_indicator_permutation_dataset"] = copy.deepcopy(
                        torch.stack([data['x_perm'] if isinstance(data['x_perm'], torch.Tensor) else torch.tensor(data['x_perm'])
                        for data in pm_permutation_distance_dataset])
                    )
            elif method["database_style"] == "general":
                print("""Warning:Using parametrically managed activation function, and 
                         database_style is set to general. Becasue the system is "atom+diatom",
                         we assume you have used "distances" in a general database style. The 
                         "R_indicator" is simply copy the general database. If not, please return
                         to use "xyz" or "canonical_xyz" database_style.
                    """)
                # features is from obtaining main_dataset in previous parts of code
                method["PM_config"]["R_indicator_train_dataset"] = copy.deepcopy(
                    torch.stack([data['x'] if isinstance(data['x'], torch.Tensor) else torch.tensor(data['x'])
                    for data in train_dataset])
                )
                if method["permutation_restrain"] == True:
                    method["PM_config"]["R_indicator_permutation_dataset"] = copy.deepcopy(
                        torch.stack([data['x_perm'] if isinstance(data['x_perm'], torch.Tensor) else torch.tensor(data['x_perm'])
                        for data in permutation_dataset])
                    )

        if method["PM_config"]["pm"] is None:
            print("""Warning:parametrically managed activation function's smooth boxcar function parameters
                     was not set up, we use a reasonable guess of it by setting it to -1.0, 4.0, 2.0""")
            method["PM_config"]["pm"] = [-1.0, 4.0, 2.0]

        
        # following is the convention for base_potential which is basically the pair-wise additive potential 
        # 1. if the user uses "canonical_xyz" database_style, base_potential should be in metadata,
        #    and no additional database should be provided
        # 2. if the user uses either "xyz" or "general" database_style, the user should provide a database
        #    with same style and assigned to base_potential keyword. Then this database will be read, 
        #    notice that the energy part in this database should be base_potential, i.e. pair-wise additive potential
        if method["PM_config"]["base_potential"] is None:
            if method["database_style"] == "canonical_xyz":
                pm_main_base_potential_dataset = create_dataset(
                    dataset_input,
                    method,
                    permutation_override=(False if method["permutation"] == "restrain" else None),
                    representation_override="distance",
                    target_key_override="base_potential"
                )
                method["PM_config"]["base_potential_train_dataset"] = copy.deepcopy(
                    torch.stack([data['y'] if isinstance(data['y'], torch.Tensor) else torch.tensor(data['y'])
                    for data in pm_main_base_potential_dataset])
                )
                # for permutation database:
                if method["permutation_restrain"] == True:
                    pm_permutation_base_potential_dataset = create_dataset(
                        dataset_input,
                        method,
                        representation_override="distance",
                        target_key_override="base_potential"
                    )
                    method["PM_config"]["base_potential_permutation_dataset"] = copy.deepcopy(
                        torch.stack([data['y_perm'] if isinstance(data['y'], torch.Tensor) else torch.tensor(data['y'])
                        for data in pm_permutation_base_potential_dataset])
                    )
            if method["database_style"] in ["xyz", "general"]:
                raise ValueError(f""" When using parametrically managed activation function and the 
                                      database_style is "xyz" or "general", base_potential has to be provided""")
        else:
            if method["database_style"] == "canonical_xyz":
                raise ValueError("""Warning: when using parametrically managed activation function and database_style 
                                    is "canonical_xyz", the base_potential should be in metadata, instead of a separate
                                    database""")
            else:
                base_potential_set = method["PM_config"]["base_potential"]

                base_potential_parsed_data = {}

                try:
                    pm_main_base_potential_database, pm_main_base_potential_database_filled = parse_convenient_multistate_xyz(base_potential_set)
                    base_potential_parsed_data["xyz"] = pm_main_base_potential_database_filled
                except Exception as e:
                    base_potential_parsed_data["xyz"] = None

                try:
                    pm_main_base_potential_features, pm_main_base_potential_targets = parse_general_file(base_potential_set)
                    base_potential_parsed_data["general"] = (pm_main_base_potential_features, pm_main_base_potential_targets)
                except Exception as e:
                    base_potential_parsed_data["general"] = None

                # Identify the correct parsing method
                if method["database_style"] in base_potential_parsed_data and base_potential_parsed_data[method["database_style"]] is not None:
                    pm_main_base_potential_dataset_input = base_potential_parsed_data[method["database_style"]]
                else:
                    if autoparsing_happened is True:
                        raise ValueError(
                            "Error: The 'database_style' was already corrected once for the main database. "
                            "Now it is also inconsistent for the R_indicator database. Please manually check "
                            "and ensure the correct style is used for each database."
                        )
                    # Warn and auto-correct if possible
                    for style, data in base_potential_parsed_data.items():
                        if data is not None:
                            print(f"Warning: Provided 'database_style' ({method['database_style']}) is incorrect for base potential. "
                                  f"Automatically using '{style}' instead.")
                            pm_main_base_potential_dataset_input = data
                            method["database_style"] = style
                            break
                    else:
                        raise ValueError("Error: Could not parse the base potential set with any available method.")

                """
                following code has been replaced by auto-detection:

                if method["database_style"] == "xyz":
                    pm_main_base_potential_database, pm_main_base_potential_database_filled = parse_convenient_multistate_xyz(base_potential_set)
                    pm_main_base_potential_dataset_input = pm_main_base_potential_database_filled
                elif method["database_style"] == "general":
                    pm_main_base_potential_features, pm_main_base_potential_targets = parse_general_file(base_potential_set)
                    pm_main_base_potential_dataset_input = (pm_main_base_potential_features, pm_main_base_potential_targets)        

                """
                # Create main pm_main_base_potential_dataset
                pm_main_base_potential_dataset = create_dataset(
                    pm_main_base_potential_dataset_input,
                    method,
                    permutation_override=(False if method["permutation"] == "restrain" else None)
                )
                method["PM_config"]["base_potential_train_dataset"] = copy.deepcopy([data[1] for data in pm_main_base_potential_dataset])
                base_potential_feature_for_check = copy.deepcopy([data[0] for data in pm_main_base_potential_dataset])
                if method["permutation_restrain"] == True:
                    pm_main_base_potential_dataset = create_dataset(
                        pm_main_base_potential_dataset_input,
                        method
                    )
                    method["PM_config"]["base_potential_permutation_dataset"] = copy.deepcopy([data[1] for data in pm_permutation_base_potential_dataset])
                    base_potential_permuted_feature_for_check = copy.deepcopy([data[0] for data in pm_permutation_base_potential_dataset])
                #now becasue this is another database, we want to check and masure features are the same as original database 
                original_feature = copy.deepcopy([data[0] for data in train_dataset])
                if not torch.allclose(base_potential_feature_for_check, original_feature, atol=1e-8):
                    diff = base_potential_feature_for_check - original_feature
                    max_diff = torch.max(torch.abs(diff))
                    raise ValueError("Feature tensors between original database and base_potential are different! Max difference: {max_diff.item()}")
                if method["permutation_restrain"] == True:
                    original_permuted_feature = copy.deepcopy([data[0] for data in permutation_dataset])
                    if not torch.allclose(base_potential_permuted_feature_for_check, original_permuted_feature, atol=1e-8):
                        diff = base_potential_permuted_feature_for_check - original_permuted_feature
                        max_diff = torch.max(torch.abs(diff))
                        raise ValueError("Feature tensors between original permutation database and permuted base_potential are different! Max difference: {max_diff.item()}")


    print("=====================================================")
    print(" Input file parsing finished")
    print("=====================================================")


    return method


def parse_list(value, key_name, dtype=int):
    """Parses a list from different input formats and converts elements to the specified data type (int/float)."""
    try:
        if isinstance(value, list):
            return [dtype(v) for v in value]
        elif isinstance(value, str):
            value = value.strip()
            if not value:
                return []  # Handle empty string cases
            if "," in value:
                return [dtype(v.strip()) for v in value.split(",")]
            try:
                return [dtype(ast.literal_eval(value))]
            except ValueError:
                raise ValueError(f"Invalid format for {key_name}: {value}. Expected a list or comma-separated values.")
    except (ValueError, SyntaxError, TypeError):
        raise ValueError(f"Invalid format for {key_name}: {value}. Expected a list or comma-separated values.")


def validate_method_config(method, valid_options):
    """
    Validates only the keys present in `VALID_OPTIONS` without blocking extra keys.

    Args:
        method (dict): The user-defined method configuration.
        valid_options (dict): The dictionary containing valid options.

    Returns:
        None (raises an error if invalid values are found).
    """
    for key, value in method.items():
        if key in valid_options:  # Only check known keys
            # If valid_options[key] is a set, check if the value is valid
            if isinstance(valid_options[key], set):
                if isinstance(value, str):
                    if value.lower() not in {v.lower() for v in valid_options[key]}:
                        raise ValueError(f"Invalid value '{value}' for '{key}'. Must be one of {valid_options[key]}.")
                else:
                    raise ValueError(f"Expected string value for '{key}', got {type(value).__name__}.")

            # If valid_options[key] is a dictionary, validate nested options
            elif isinstance(valid_options[key], dict):
                if not isinstance(value, dict):
                    raise ValueError(f"Expected dictionary for '{key}', but got {type(value).__name__}.")

                for sub_key, sub_value in value.items():
                    if sub_key in valid_options[key]:  # only check valid keys
                        if isinstance(valid_options[key][sub_key], set):
                            # Convert input to list if it is a string (comma-separated)
                            if isinstance(sub_value, str):
                                sub_value = [v.strip() for v in sub_value.split(",")]
                                method[key][sub_key] = sub_value  # Update in-place

                            # Validate each item in the list
                            elif isinstance(sub_value, list):
                                for item in sub_value:
                                    if item not in valid_options[key][sub_key]:
                                        raise ValueError(
                                            f"Invalid value '{item}' for '{sub_key}' in '{key}'. "
                                            f"Must be one of {valid_options[key][sub_key]}."
                                        )
                            else:
                                raise ValueError(f"'{sub_key}' in '{key}' must be a list or comma-separated string.")
            else:
                raise ValueError(f"Invalid data structure for '{key}'.")


def safe_json_serialization(obj):
    """Convert non-serializable objects into a string representation for debugging."""
    if isinstance(obj, (int, float, str, bool, list, dict, type(None))):
        return obj  # Already serializable

    # Convert known objects to a readable format
    if hasattr(obj, "__class__"):
        return f"<{obj.__class__.__name__} object>"

    return str(obj)  # Fallback for unknown types

def print_datetime():
    now = datetime.datetime.now()
    print("Current Date and Time:", now.strftime("%Y-%m-%d %H:%M:%S"))

def header():
    print("=" * 50)
    print("  ")
    print(" Multi-State Surface Learning (MSLearn) Package")
    print("  ")
    print("==================================================")
    print("Authors: Yinan Shu, Donald G. Truhlar")
    print("        University of Minnesota")
    print("Version: 1.0")
    print("         Updated on Feb 14, 2025")
    print("  ")
    print("Features of current version")
    print("    Available coordinate files:")
    print("        general, xyz, canonical_xyz")
    print("    Available network architecture: ")
    print("        feed-forward neural network")
    print("        message-passing neural network")
    print("    Available molecular representation:")
    print("        xyz, zmatrix, distance")
    print("        permutation invaraint polynomials (PIPs) of distances,")
    print("        graph")
    print("    Available methods:")
    print("        diabatization by deep neural network (DDNN)")
    print("        compatibilization by deep neural network (CDNN)")
    print("            with controllable matrix form")
    print("  ")
    print("    physical properties:")
    print("        1. using permutation invariant molecular representation or")
    print("           a permuted database to constrain or restrain permutation symmetry")
    print("        2. using parametrically managed activation function to enforce")
    print("           correct behavior of potential energy surfaces (PESs) in asymptotic")
    print("           and/or repulsive regions")


    print("=" * 50)
    print(" ")
    print_datetime()


def save_history_csv(history, filename="training_history.csv"):
    """Save training history to CSV, handling different history lengths safely."""
    
    # Separate keys based on "true_" prefix
    normal_keys = [k for k in history.keys() if not k.startswith("true_")]
    true_keys = [k for k in history.keys() if k.startswith("true_")]

    # Find max length for normal and true histories
    normal_epochs = max(len(history[k]) for k in normal_keys) if normal_keys else 0
    true_epochs = max(len(history[k]) for k in true_keys) if true_keys else 0

    # Function to safely get values, handling missing data
    def get_value(hist_list, index):
        if index < len(hist_list):
            return hist_list[index].item() if isinstance(hist_list[index], torch.Tensor) else hist_list[index]
        else:
            return None  # Fill missing epochs with None

    # Save normal history
    if normal_keys:
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch"] + normal_keys)  # Write header
            
            for epoch in range(normal_epochs):
                row = [epoch] + [get_value(history[key], epoch) for key in normal_keys]
                writer.writerow(row)

    # Save "true_" history separately
    if true_keys:
        with open("true_" + filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch"] + true_keys)  # Write header
            
            for epoch in range(true_epochs):
                row = [epoch] + [get_value(history[key], epoch) for key in true_keys]
                writer.writerow(row)

    print(f"History saved to {filename} and true_{filename}")


def save_training_history_csv(history, csv_path, is_true_history=False):
    """Save normal or true training history to CSV format, handling different lengths of history keys."""

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Select appropriate history keys based on is_true_history flag
    history_keys = [key for key in history.keys() if key.startswith("true_")] if is_true_history else \
                   [key for key in history.keys() if not key.startswith("true_")]

    # Find max length among different history keys
    max_epochs = max(len(history[key]) for key in history_keys) if history_keys else 0

    with open(csv_path, mode="w") as f:
        writer = f.write
        header = ["epoch"] + history_keys + ["\n"]
        writer(",".join(header))

        # Iterate up to max_epochs, filling missing values with "NA"
        for epoch in range(max_epochs):
            row = [str(epoch)]
            for key in history_keys:
                if epoch < len(history[key]):  # Ensure index is within bounds
                    value = history[key][epoch]
                    row.append(f"{value.item():.6f}" if isinstance(value, torch.Tensor) else str(value))
                else:
                    row.append("NA")  # Fill missing values with "NA"
            writer(",".join(row) + "\n")

    print(f"Training history saved to {csv_path}")


def save_yaml_config(config, output_dir="output"):
    """
    Saves the training configuration in YAML format.

    Args:
        config (dict): The training configuration.
        output_dir (str): Directory to save the config file.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Remove non-serializable objects (e.g., dataset references)
    config_to_save = config.copy()
    
    # Replace dataset objects with file paths (if available)
    if "train_dataset" in config_to_save:
        config_to_save["train_dataset_path"] = "dataset_path_here.xyz"
        del config_to_save["train_dataset"]  # Remove the dataset object

    if "diabatic_dataset" in config_to_save:
        config_to_save["diabatic_dataset_path"] = "diabatic_dataset_path_here.xyz"
        del config_to_save["diabatic_dataset"]

    if "permutation_dataset" in config_to_save:
        config_to_save["permutation_dataset_path"] = "permutation_dataset_path_here.xyz"
        del config_to_save["permutation_dataset"]

    # Save to YAML
    yaml_path = os.path.join(output_dir, "model_config.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)

    print(f"Model configuration saved successfully at: {yaml_path}")




def save_training_outputs(model, config, history, output_dir="output"):
    """
    Saves the trained model, model configuration, training history (CSV only),
    model architecture summary, and weights & biases for NN or MPNN architectures.

    Args:
        model (torch.nn.Module): Trained model
        config (dict): Model configuration and hyperparameters
        history (dict): Training history with loss values (normal & "true_" versions)
        output_dir (str): Folder where all outputs are saved (default: 'output')
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(output_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved: {model_path}")

    # Save model configuration in json
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, default=str)
    print(f"Model configuration saved: {config_path}")

    # save model configuration in yaml
    save_yaml_config(config)
    print(f"Model configuration saved to {config_path}")

    # Save training history (CSV)
    normal_history_csv = os.path.join(output_dir, "training_history.csv")
    true_history_csv = os.path.join(output_dir, "true_training_history.csv")

    save_training_history_csv(history, normal_history_csv, is_true_history=False)
    save_training_history_csv(history, true_history_csv, is_true_history=True)

    # Save model architecture summary
    model_summary_path = os.path.join(output_dir, "model_summary.txt")
    with open(model_summary_path, "w") as f:
        f.write(str(model))
    print(f"Model summary saved: {model_summary_path}")

    # Save model weights & biases for NN
    if config["architecture"] == "nn":
        ws_bs_path = os.path.join(output_dir, "ws_and_bs.txt")
        with open(ws_bs_path, "w") as file:
            for layer_idx, (name, param) in enumerate(model.named_parameters()):
                if "weight" in name:
                    file.write(f"!coeff layer {layer_idx}\n")
                    for i, row in enumerate(param.data):
                        for j, value in enumerate(row):
                            file.write(f"       coeff_l{layer_idx}({i + 1},{j + 1})= {value.item(): .10e}\n")
                elif "bias" in name:
                    file.write(f"!bias layer {layer_idx}\n")
                    for i, value in enumerate(param.data):
                        file.write(f"       bias_l{layer_idx}({i + 1})= {value.item(): .10e}\n")
        print(f"NN Weights and biases saved: {ws_bs_path}")

    # Save Message Passing Neural Network (MPNN) parameters if applicable
    if config["architecture"] == "mpnn":
        mpnn_params_path = os.path.join(output_dir, "mpnn_params.txt")
        with open(mpnn_params_path, "w") as file:
            for name, param in model.named_parameters():
                if "msg_nn" in name:
                    file.write(f"! Message NN {name}\n")
                    if param.data.dim() == 0:  # If it's a scalar tensor
                        file.write(f"       msg_nn = {param.item(): .10e}\n")
                    else:
                        for i, row in enumerate(param.data):
                            if row.dim() == 0:  # Handle 0D tensor case inside the loop
                                file.write(f"       msg_nn({i + 1})= {row.item(): .10e}\n")
                            else:
                                for j, value in enumerate(row):
                                    file.write(f"       msg_nn({i + 1},{j + 1})= {value.item(): .10e}\n")

                elif "update_nn" in name:
                    file.write(f"! Update NN {name}\n")
                    if param.data.dim() == 0:  # If it's a scalar tensor
                        file.write(f"       update_nn = {param.item(): .10e}\n")
                    else:
                        for i, row in enumerate(param.data):
                            if row.dim() == 0:  # Handle 0D tensor case inside the loop
                                file.write(f"       update_nn({i + 1})= {row.item(): .10e}\n")
                            else:
                                for j, value in enumerate(row):
                                    file.write(f"       update_nn({i + 1},{j + 1})= {value.item(): .10e}\n")

        print(f"MPNN parameters saved: {mpnn_params_path}")

    print("\n===All training outputs saved successfully!===\n")
    print_datetime()


def main():
    # Parse command-line arguments

    header()

    parser = argparse.ArgumentParser(description="Train a model using parsed input configuration.")
    parser.add_argument("config_file", type=str, help="Path to the input configuration file.")
    args = parser.parse_args()

    config = parse_user_input(args.config_file)

    print("==========================================")
    print("           Loaded Configuration")
    print("==========================================")
    print(json.dumps(config, indent=4, default=str))

    set_seed(config["random_seed"])

    model_params = {
        "input_dim": config["input_dim"],
        "hidden_dims": config["hidden_layers"],
        "output_dim": config["output_dim"],
        "architecture": config["architecture"],
        "mpnn_structure": config.get("mpnn_structure", "linear"),
        "msg_nn": config.get("msg_nn"),
        "update_nn": config.get("update_nn"),
        "edge_attr_dim": config.get("edge_attr_dim", 0),
        "activation": config["activation"],
        "matrix_type": config.get("matrix_type"),
    }

    model = MSLP(**model_params)

    trainer = MSLPtrain(model, config)
    trained_model, history = trainer.train()

    print("==========================================")
    print("           Training finished")
    print("==========================================")

    save_training_outputs(model, config, history)


if __name__ == "__main__":
    main()


