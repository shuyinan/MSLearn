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
# The model module
#=============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
}

class MSLPMPNNLayer(MessagePassing):
    def __init__(self, in_channels, edge_attr_dim, out_channels, activation_fn, 
                 mpnn_structure='linear', msg_nn=None, update_nn=None):
        """
        Unified MPNN layer supporting both linear and nonlinear message passing.
        
        Args:
            in_channels (int): Number of input node features.
            edge_attr_dim (int): Number of edge features.
            out_channels (int): Number of output node features.
            activation_fn (callable): Activation function.
            mpnn_structure (str): 'linear' or 'nonlinear' to control message passing.
            msg_nn (list, optional): Hidden layer sizes for message network (for nonlinear).
            update_nn (list, optional): Hidden layer sizes for update network (for nonlinear).
        """
        super().__init__(aggr='add')  # Aggregate messages via summation
        self.activation_fn = activation_fn
        self.mpnn_structure = mpnn_structure

        if self.mpnn_structure=='linear':
            self.msg_nn = nn.Linear(in_channels + edge_attr_dim, out_channels)
            self.update_nn = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                self.activation_fn()
            )
        elif self.mpnn_structure=='nonlinear':
            self.msg_nn = MSLP.build_mlp(in_channels + edge_attr_dim, 
                                          msg_nn if msg_nn else [out_channels], 
                                          out_channels, self.activation_fn)
            self.update_nn = MSLP.build_mlp(out_channels, 
                                             update_nn if update_nn else [out_channels], 
                                             out_channels, self.activation_fn)
        else:
            raise ValueError(f"Unsupported mpnn_structure: {mpnn_structure}. Choose 'linear' or 'nonlinear'.")

    def forward(self, x, edge_index, edge_attr):
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        edge_input = torch.cat([x_j, edge_attr], dim=-1)  # Combine node & edge features
        return self.msg_nn(edge_input)

    def update(self, aggr_out):
        return self.update_nn(aggr_out)

    #def update(self, aggr_out):
    #    return self.update_nn(aggr_out) if self.mpnn_structure == 'nonlinear' else self.activation_fn(self.update_nn(aggr_out))


class MSLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, 
            architecture='nn', mpnn_structure='linear', 
            msg_nn=None, update_nn=None, edge_attr_dim=0,
            activation='gelu', matrix_type=None):
        """
        `matrix_type` is the number of non-zero off-diagonal elements, it is an odd number:
          1 - diagonal only
          3 - tri-diagonal 
          5 - penta-diagonal
          default is full(dense) matrix

        Args:
            input_dim (int): Number of input features.
            hidden_dims (list): List of hidden layer sizes.
            output_dim (int): Number of output dimensions.
            matrix_type (int): Must be an odd number (1, 3, 5, ...), indicating the range of off-diagonal elements.
            activation_fn (callable, optional): Activation function class (default: nn.GELU).
            base_potential_type (str, optional): Keyword to specify the type of baseline potential
        """
        super().__init__()

        self.matrix_type = matrix_type
        self.output_dim = output_dim
        self.architecture = architecture

        self.activation_fn = ACTIVATION_FUNCTIONS.get(activation.lower(), nn.GELU)
        DPEM_elements = output_dim + sum(output_dim - k for k in range(1, (self.matrix_type - 1) // 2 + 1))

        if self.architecture == 'mpnn':
            self.mpnn_layers = nn.ModuleList()
            self.mpnn_layers.append(MSLPMPNNLayer(input_dim, edge_attr_dim, hidden_dims[0], self.activation_fn, mpnn_structure, msg_nn, update_nn))
            for i in range(1, len(hidden_dims)):
                self.mpnn_layers.append(MSLPMPNNLayer(hidden_dims[i-1], edge_attr_dim, hidden_dims[i], self.activation_fn, mpnn_structure, msg_nn, update_nn))
            self.mpnn_layers.append(MSLPMPNNLayer(hidden_dims[-1], edge_attr_dim, DPEM_elements, nn.Identity, mpnn_structure, msg_nn, update_nn))

        elif self.architecture == 'nn': 
            self.feedforward = MSLP.build_mlp(input_dim, hidden_dims, hidden_dims[-1], self.activation_fn)
            # Adjust final layer output based on matrix_type
            self.final_layer = nn.Linear(
                hidden_dims[-1], 
                output_dim + sum(output_dim - k for k in range(1, (self.matrix_type - 1) // 2 + 1))
            )

    def forward(self, x, edge_index=None, edge_attr=None, batch=None, 
                parametric_mode=False, base_potential_type=None, 
                R_indicator=None, pm=None, base_potential=None):
        """
        Forward pass that constructs the Pmatrix dynamically based on `matrix_type` and `parametric_mode`.

        Args:
        x (Tensor): Input tensor.
        
            If using a feedforward neural network (`architecture='nn'`):
                - `x` has shape `(batch_size, input_dim)`, where `input_dim` represents feature dimensions 
                  (e.g., bond distances, atomic coordinates, or molecular descriptors).
        `parametric_mode` controls if the parametrically managed activation function is used.
        `base_potential_type` controls what type of parametrically managed activation function is used.
            Current only supported list:
                1.  `atom+diatom`, which is pair-wise additive potential.
                        required input for atom+diatom: R_indicator, pm, base_potential
            when base_potential_type is "atom+diatom":
                R_indicator (Tensor, optional): Can be distance matrix for potential calculations.
                pm (list, optional): Parameters for potential function.
                base_potential (Tensor, optional): Additional correction potential.

            If using a message-passing neural network (`architecture='mpnn'`):
                - Assume we have `N_mol` molecules, each with `N_atom` atoms.
                - `x` has shape `(N_mol * N_atom, input_dim)`, where `input_dim` is the number of atomic features.
                  (i.e., all nodes from all molecules are concatenated together).
                - `edge_index` has shape `(2, N_edge)`, where `N_edge` is the total number of edges.
                  It stores pairs of atom indices that define each bond (edge).
                - `edge_attr` has shape `(N_edge, edge_attr_dim)`, where `edge_attr_dim` is the number of edge features
                  (e.g., bond length, bond type, distance between atoms).
                - `batch` is a tensor of shape `(N_mol * N_atom,)` mapping each atom to its corresponding molecule.

        Returns:
            Tensor: Sorted eigenvalues of the computed Pmatrix.
        """

        if self.architecture == 'mpnn':
            for layer in self.mpnn_layers:
                x = layer(x, edge_index, edge_attr)
            if batch is None:
                raise ValueError("Batch tensor is required for message passing neural network  architectures.")
            # Aggregate node features into per-graph representations
            x = global_mean_pool(x, batch)
        elif self.architecture == 'nn':
            x = self.feedforward(x)
            x = self.final_layer(x)

        Pmatrix, unique_elements = self.compute_Pmatrix(x, parametric_mode=False, base_potential_type=None, R_indicator=None, pm=None, base_potential=None) 

        # Compute eigenvalues
        pre_eigenvalues, pre_eigenvectors = torch.linalg.eig(Pmatrix)
        eigen_real = pre_eigenvalues.real
        sorted_indices = torch.argsort(eigen_real, dim=1)

        eigenvalues = torch.gather(pre_eigenvalues, 1, sorted_indices)

        return unique_elements, eigenvalues


    def compute_Pmatrix(self, x, parametric_mode=False, base_potential_type=None, R_indicator=None, pm=None, base_potential=None):
        """
        Computes the Pmatrix for any odd-numbered matrix_type with or without parametric mode.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            Tensor: Computed Pmatrix.
        """
        batch_size = x.shape[0]
        num_off_diag = sum(self.output_dim - k for k in range(1, (self.matrix_type - 1) // 2 + 1))
        Pmatrix = torch.zeros(batch_size, self.output_dim, self.output_dim, device=x.device)
        unique_elements =  torch.zeros(batch_size, self.output_dim + num_off_diag, device=x.device)

        # the following 2 variables are used to assign off-diagonal elements
        index_offset = self.output_dim  # Offset for accessing correct parts of x
        max_offset = (self.matrix_type - 1) // 2  # Max off-diagonal distance
        unique_index = self.output_dim # Indexing unique_elements for off-diagonal elements

        # if using parametrically managed activation function:
        if parametric_mode:
            if base_potential_type == "atom+diatom":
                # Assign diagonal elements
                for k in range(self.output_dim):
                    Pmatrix[:, k, k] = self.PM_atom_diatom(R_indicator, x[:, k], pm) + base_potential[:, k]
                    unique_elements[:,k] = Pmatrix[:, k, k]
                # Assign off-diagonal elements
                for offset in range(1, max_offset + 1):
                    for k in range(self.output_dim - offset):
                        Pmatrix[:, k, k + offset] = self.PM_atom_diatom(R_indicator, x[:, index_offset], pm)
                        Pmatrix[:, k + offset, k] = Pmatrix[:, k, k + offset]
                        unique_elements[:, unique_index] = Pmatrix[:, k, k + offset]
                        index_offset += 1
                        unique_index += 1
            else:
                raise ValueError(f"Unknown base_potential_type type: {base_potential_type}")

        # if not using parametrically managed activation function:
        else:
            # Assign diagonal elements
            for k in range(self.output_dim):
                Pmatrix[:, k, k] = x[:, k]
                unique_elements[:,k] = Pmatrix[:, k, k]
            # Assign off-diagonal elements
            for offset in range(1, max_offset + 1):
                for k in range(self.output_dim - offset):
                    Pmatrix[:, k, k + offset] = x[:, index_offset]
                    Pmatrix[:, k + offset, k] = Pmatrix[:, k, k + offset]
                    unique_elements[:, unique_index] = Pmatrix[:, k, k + offset]
                    index_offset += 1
                    unique_index += 1

        return Pmatrix, unique_elements

    @staticmethod
    def build_mlp(input_dim, layer_sizes, last_layer_dim, activation_fn):
        layers = []
        prev_dim = input_dim  # Initial input size
        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())  # Ensure activation function is instantiated
            prev_dim = hidden_dim  # Update for next layer
        layers.append(nn.Linear(prev_dim, last_layer_dim))  # Final transformation to `out_channels`
        return nn.Sequential(*layers)

    def PM_atom_diatom(R_indicator, values, pm):
        """
        Parametrically Managed Activation Function for atom+diatom.

        Args:
            R_indicator (Tensor): Input tensor.
            values (Tensor): Values to modify.
            pm (list): Parametric activation values.

        Returns:
            Tensor: Modified values.
        """
        order = 4.0
        nsize = R_indicator.shape[0]
        pair_sum = sum((R_indicator[:, i] + R_indicator[:, j]) ** order for i in range(R_indicator.shape[1]) for j in range(i + 1, R_indicator.shape[1]))
        bond_sum = torch.sum(R_indicator ** order, axis=1)
        r_in = pair_sum ** (1 / order) - bond_sum ** (1 / order)

        Ar = 0.5 + 0.5 * torch.tanh(pm[2] * (r_in - pm[0]))
        Br = 0.5 + 0.5 * torch.tanh(pm[2] * (-r_in + pm[1]))
        return values * (Ar * Br)

