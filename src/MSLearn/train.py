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
# The training module
#=============================================
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Subset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from itertools import cycle
import os
import copy

from .dataset import CombinedDataset

class MSLPtrain:
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
       self.model = model.to(device)
       self.config = config
       self.device = device


    def train(self): 

        history = MSLPtrainer(
            self.model,
            **{
                "train_dataset": self.config["train_dataset"],
                "training_percent": self.config.get("training_percent", 100),
                "batch_size": self.config["batch_size"],
                "diabatic_restrain": self.config.get("diabatic_restrain", False),
                "diabatic_dataset": self.config.get("diabatic_dataset"),
                "diabatic_weight": self.config.get("diabatic_weight", 1.0),
                "permutation_restrain": self.config.get("permutation_restrain", False),
                "permutation_dataset": self.config.get("permutation_dataset"),
                "permutation_weight": self.config.get("permutation_weight", 1.0),
                "topology_attention": self.config.get("topology_attention", False),
                "topology_attention_weight": self.config.get("topology_attention_weight", 0),
                "regularization": self.config.get("regularization", True),
                "regularization_weight": self.config.get("regularization_weight", 1e-7),
                "parametric_mode": self.config.get("parametric_mode", False),
                "base_potential_type": self.config.get("base_potential_type"),
                "pm": self.config.get("pm"),
                "R_indicator": self.config.get("PM_config", {}).get("R_indicator_train_dataset"),
                "base_potential": self.config.get("PM_config", {}).get("base_potential_train_dataset"),
                "permuted_R_indicator": self.config.get("PM_config", {}).get("R_indicator_permutation_dataset"),
                "permuted_base_potential": self.config.get("PM_config", {}).get("base_potential_permutation_dataset"),
                "num_epochs": self.config["num_epochs"],
                "learning_rate": self.config["learning_rate"],
                "optimizer_type": self.config["optimizer_type"],
                "scheduler_type": self.config.get("scheduler_type"),
                "loss_threshold": self.config.get("loss_threshold", 0.002),
                "early_stopping_patience": self.config.get("early_stopping_patience", 10),
                "early_stopping_delta": self.config.get("early_stopping_delta", 0),
                "loss_function": self.config["loss_function"],
                "checkpoint_path": self.config["checkpoint_path"],
                "print_interval": self.config["print_interval"],
                "save_interval": self.config["save_interval"],
            },
            device=self.device

        )

        return self.model, history

def save_checkpoint(model, optimizer, scheduler, epoch, history, best_val_loss, early_stopping_counter,
                    ema_loss, convergence_counter, previous_total_loss, checkpoint_path="checkpoint.pth"):
    """Save model, optimizer, scheduler states, and training history."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'history': history,
        'best_val_loss': best_val_loss,
        'early_stopping_counter': early_stopping_counter,
        'ema_loss': ema_loss,
        'convergence_counter': convergence_counter,
        'previous_total_loss': previous_total_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path="checkpoint.pth"):
    """Load model, optimizer, scheduler states, and training history if checkpoint exists."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1

        best_val_loss = checkpoint['best_val_loss']
        early_stopping_counter = checkpoint['early_stopping_counter']
        ema_loss = checkpoint['ema_loss']
        convergence_counter = checkpoint['convergence_counter']
        previous_total_loss = checkpoint['previous_total_loss']

        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return start_epoch, history, best_val_loss, early_stopping_counter, ema_loss, convergence_counter, previous_total_loss
    return 0, {}, float('inf'), 0, None, 0, float('inf')


def get_next(iterator, loader):
    try:
        return next(iterator)
    except StopIteration:
        return next(iter(loader))  # Reset and continue


def batch_loss_eval(batch, model, criterion, target, loss1, weight, loss2):
    if model.architecture == 'nn':
        x = batch.x.view(batch.num_graphs, -1)
        edge_index, edge_attr, batch_idx = None, None, None
        y = batch.y.view(batch.num_graphs, -1)
    elif model.architecture == 'mpnn':
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        y = batch.y.view(batch.num_graphs, -1)

    unique_elements, output = model(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_idx)
    if target == "energy":
        batch_loss = criterion(output.real, y)
    elif target == "DPEM":
        batch_loss = criterion(unique_elements.real, y)
    else:
        raise ValueError(f"unknow target for batch loss")

    loss1 = loss1 + batch_loss
    loss2 = loss2 + weight*batch_loss
    return batch_loss, loss1, loss2


def batch_loss_eval_pm(batch, base_potential_type, batch_R, pm, batch_P, model, criterion, target, loss1, weight, loss2):
    if model.architecture == 'nn':
        x = batch.x.view(batch.num_graphs, -1)
        edge_index, edge_attr, batch_idx = None, None, None
        y = batch.y.view(batch.num_graphs, -1)
    elif model.architecture == 'mpnn':
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        y = batch.y.view(batch.num_graphs, -1)

    unique_elements, output = model(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_idx,
                                    parametric_mode=True, base_potential_type=base_potential_type,
                                    R_indicator=batch_R, pm=pm, base_potential=batch_P)
    if target == "energy":
        batch_loss = criterion(output.real, y)
    elif target == "DPEM":
        batch_loss = criterion(unique_elements.real, y)
    else:
        raise ValueError(f"unknow target for batch loss")

    loss1 = loss1 + batch_loss
    loss2 = loss2 + weight*batch_loss
    return batch_loss, loss1, loss2
 
def get_ema(ema_loss, total_batch_loss):
    if ema_loss is None:
        ema_loss = total_batch_loss.item()
    else:
        ema_loss = 0.9 * ema_loss + 0.1 * total_batch_loss.item()
    return ema_loss

def custom_collate(batch):
    """Collate function for handling PyG Data objects."""
    return Batch.from_data_list(batch)

def MSLPtrainer(model, 
              train_dataset, training_percent=100, batch_size=32, 
              # diabatic, permutation, topology attention, and regularization terms
              diabatic_restrain=False, diabatic_dataset=None, diabatic_weight=1.0,
              permutation_restrain=False, permutation_dataset=None, permutation_weight=1.0,
              topology_attention=False, topology_attention_weight=0, 
              regularization=True, regularization_weight=1e-7, 
              # parametrically managed activation function term 
              parametric_mode=False, base_potential_type=None, pm=None, 
              R_indicator=None, base_potential=None,
              permuted_R_indicator=None, permuted_base_potential=None,
              # optimizer hyperparameters
              num_epochs=10000, learning_rate=1e-3,  
              optimizer_type='adam', scheduler_type=None, 
              loss_threshold=0.002, early_stopping_patience=10, early_stopping_delta=0, 
              # loss function
              loss_function='mse',
              # restart
              checkpoint_path="checkpoint.pth", 
              # print
              print_interval=100, 
              # save
              save_interval=10,
              # device
              device='cuda' if torch.cuda.is_available() else 'cpu'):

    """
    Utility function to train a model (supports both nn and mpnn architectures).
    """

    assert train_dataset is not None, "train_dataset must be provided!"

    model = model.to(device)

    if loss_function == 'mse':
        criterion = nn.MSELoss()
    elif loss_function == 'mae':
        criterion = nn.L1Loss()

    # Shuffle the datasets manually, so we can control the correspondence between train_dataset, R_indicator, and base_potential
    indices = torch.randperm(len(train_dataset)).tolist()
    train_dataset = Subset(train_dataset, indices)
    #train_dataset = [train_dataset[i] for i in indices]
 
    if parametric_mode:
        R_indicator = R_indicator[indices]
        base_potential = base_potential[indices]

    if permutation_restrain and permutation_dataset:
        permuted_indices = torch.randperm(len(permutation_dataset)).tolist()
        permutation_dataset = Subset(permutation_dataset, permuted_indices)
        if parametric_mode:
            permuted_R_indicator = permuted_R_indicator[permuted_indices]
            permuted_base_potential = permuted_base_potential[permuted_indices]

    train_size = int(training_percent / 100 * len(train_dataset))
    val_size = len(train_dataset) - train_size


    # Main and permutation datasets will be combined with R_indicator and base_potential if using parametric_mode
    if not parametric_mode:
        train_dataset = Subset(train_dataset, list(range(train_size)))
        val_dataset = Subset(train_dataset, list(range(train_size, len(train_dataset))))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        train_loader_test = DataLoader(copy.deepcopy(train_dataset), batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        val_loader_test = DataLoader(copy.deepcopy(val_dataset), batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        if permutation_restrain and permutation_dataset:
            permutation_loader = DataLoader(permutation_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
            permutation_iter = iter(permutation_loader) if permutation_loader else None
            permutation_loader_test = DataLoader(copy.deepcopy(permutation_dataset), batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        else:
            permutation_loader = None
            permutation_loader_test = None
    else:
        train_data_combined = CombinedDataset(train_dataset, R_indicator, base_potential)
        train_data_combined, val_data_combined = Subset(train_data_combined, list(range(train_size))), Subset(train_data_combined, list(range(train_size, len(train_data_combined))))
        train_loader = DataLoader(train_data_combined, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        train_loader_test = DataLoader(copy.deepcopy(train_data_combined), batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        val_loader = DataLoader(val_data_combined, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        val_loader_test = DataLoader(copy.deepcopy(val_data_combined), batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        if permutation_restrain and permutation_dataset:
            permutation_data_combined = CombinedDataset(permutation_dataset, permuted_R_indicator, permuted_base_potential)
            permutation_loader = DataLoader(permutation_data_combined, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
            permutation_iter = iter(permutation_loader) if permutation_loader else None
            permutation_loader_test = DataLoader(copy.deepcopy(permutation_data_combined), batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        else:
            permutation_loader = None
            permutation_loader_test = None

    # Validation and diabatic datasets are not affected by parametric_mode
    if diabatic_restrain and diabatic_dataset:
        diabatic_loader = DataLoader(diabatic_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        diabatic_iter = iter(diabatic_loader) if diabatic_loader else None
        diabatic_loader_test = DataLoader(copy.deepcopy(diabatic_dataset), batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    else:
        diabatic_loader = None
        diabatic_loader_test = None


    # select optimizer
    weight_decay = regularization_weight if regularization else 0
    if optimizer_type == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None

    history = {
        "total_loss": [],
        "exponential_moving_average_of_total_loss": [],
        "adiabatic_loss": [],
        "diabatic_loss": [],
        "permutation_loss": [],
        "regularization_loss": [],
        "topology_loss": [],
        "validation_loss": [],
        # these true values are only evaluated every 1000 epoches
        "true_total_loss": [],
        "true_adiabatic_loss": [],
        "true_diabatic_loss": [],
        "true_permutation_loss": [],
        "true_topology_loss": [],
        "true_regularization_term": [],
    }


    best_val_loss = float('inf')
    early_stopping_counter = 0
    ema_loss = None
    convergence_counter = 0
    previous_total_loss = float('inf')

    start_epoch, loaded_history, best_val_loss, early_stopping_counter, ema_loss, convergence_counter, previous_total_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    if loaded_history:
        history.update(loaded_history)

    for epoch in range(start_epoch, num_epochs):
        model.train()

        # Initialize loss accumulators
        total_loss = 0
        adiabatic_loss = 0
        diabatic_loss = 0
        permutation_loss = 0
        topology_loss = 0
        regularization_loss = 0

        # Iterate over the main training dataset
        if not parametric_mode:
            for batch in train_loader:
                total_batch_loss=0
                optimizer.zero_grad()

                batch = batch.to(device)
                adiabatic_batch_loss, adiabatic_loss, total_batch_loss = batch_loss_eval(
                        batch, model, criterion, "energy", adiabatic_loss, 1.0, total_batch_loss)

                if topology_attention:
                    off_diag_elements = torch.cat(unique_elements[:,model.output_dim:], dim=0)
                    topology_batch_loss = torch.mean(torch.sign(off_diag_elements))
                    # we only add topology_loss to total_loss and total_batch_loss if there is no permutation term
                    if not permutation_loader:
                        topology_loss += torch.abs(topology_batch_loss.item())
                        total_batch_loss += topology_attention_weight * torch.abs(topology_batch_loss)

                if diabatic_loader:
                    diabatic_batch = get_next(diabatic_iter, diabatic_loader).to(device)
                    diabatic_batch_loss, diabatic_loss, total_batch_loss = batch_loss_eval(
                            diabatic_batch, model, criterion, "DPEM", diabatic_loss, diabatic_weight, total_batch_loss)
            
                if permutation_loader:
                    permutation_batch = get_next(permutation_iter, permutation_loader).to(device)
                    permutation_batch_loss, permutation_loss, total_batch_loss = batch_loss_eval(
                            permutation_batch, model, criterion, "energy", permutation_loss, permutation_weight, total_batch_loss)

                    if topology_attention:
                        off_diag_elements = torch.cat(permutation_unique_elements[:,model.output_dim:], dim=0)
                        # notice that this topology_batch_loss is added to previous topology_batch_loss
                        topology_batch_loss = 0.5*topology_batch_loss + 0.5*torch.mean(torch.sign(off_diag_elements))
                        topology_loss += torch.abs(topology_batch_loss.item())
                        total_batch_loss += topology_attention_weight * torch.abs(topology_batch_loss)

                # Perform backpropagation
                total_batch_loss.backward()
                optimizer.step()

                # Update EMA loss
                ema_loss = get_ema(ema_loss, total_batch_loss)

        # ==========parametric mode==========
        else:
            for batch, batch_R, batch_P in train_loader:
                total_batch_loss=0
                optimizer.zero_grad()

                batch = batch.to(device)
                batch_R = batch_R.to(device)
                batch_P = batch_P.to(device)
 
                adiabatic_batch_loss, adiabatic_loss, total_batch_loss = batch_loss_eval_pm(
                        batch, base_potential_type, batch_R, pm, batch_P, model, criterion, "energy",
                        adiabatic_loss, 1.0, total_batch_loss)

                if topology_attention:
                    off_diag_elements = torch.cat(unique_elements[:, model.output_dim:], dim=0)
                    topology_batch_loss = torch.mean(torch.sign(off_diag_elements))
                    if not permutation_loader:
                        topology_loss += torch.abs(topology_batch_loss.item())
                        total_batch_loss += topology_attention_weight * torch.abs(topology_batch_loss)

                if diabatic_loader:
                    diabatic_batch = get_next(diabatic_iter, diabatic_loader).to(device)
                    diabatic_batch_loss, diabatic_loss, total_batch_loss = batch_loss_eval(
                            diabatic_batch, model, criterion, "DPEM", diabatic_loss, diabatic_weight, diabatic_batch_loss)

                if permutation_loader:
                    permutation_batch, permutation_batch_R, permutation_batch_P = get_next(permutation_iter, permutation_loader)
                    permutation_batch = permutation_batch.to(device)
                    permutation_batch_R = permutation_batch_R.to(device)
                    permutation_batch_P = permutation_batch_P.to(device)
                    permutation_batch_loss, permutation_loss, total_batch_loss = batch_loss_eval_pm(
                            permutation_batch, base_potential_type, permutation_batch_R, 
                            pm, permutation_batch_P, model, criterion, "energy",
                            permutation_loss, permutation_weight, total_batch_loss)

                    if topology_attention:
                        off_diag_elements = torch.cat(permutation_unique_elements[:, model.output_dim:], dim=0)
                        topology_batch_loss = 0.5 * topology_batch_loss + 0.5 * torch.mean(torch.sign(off_diag_elements))
                        topology_loss += torch.abs(topology_batch_loss.item())
                        total_batch_loss += topology_attention_weight * torch.abs(topology_batch_loss)

                # Perform backpropagation
                total_batch_loss.backward()
                optimizer.step()

                # Update EMA loss
                ema_loss = get_ema(ema_loss, total_batch_loss)


        # Average the accumulated losses
        adiabatic_loss /= len(train_loader)
        diabatic_loss /= len(train_loader)
        permutation_loss /= len(train_loader)
        total_loss = (
            adiabatic_loss 
            + diabatic_weight*diabatic_loss
            + permutation_weight*permutation_loss
            + topology_attention_weight*topology_loss
            + regularization_weight*regularization_loss
        )

        loss_change = total_loss - previous_total_loss
        previous_total_loss = total_loss

        # Store history
        history["total_loss"].append(total_loss)
        history["exponential_moving_average_of_total_loss"].append(ema_loss)
        history["adiabatic_loss"].append(adiabatic_loss)
        history["diabatic_loss"].append(diabatic_loss)
        history["permutation_loss"].append(permutation_loss)
        history["topology_loss"].append(topology_loss)
        history["regularization_loss"].append(regularization_loss.item() if isinstance(regularization_loss, torch.Tensor) else regularization_loss)

        # convergence based on loss threshold
        if loss_threshold is not None and loss_change is not None:
            if total_loss < loss_threshold and abs(loss_change) < early_stopping_delta:
                convergence_counter += 1
                print(f"Epoch {epoch}: Loss {total_loss:.7f} < {loss_threshold:.7f} and Loss Change {loss_change:.7f} < {early_stopping_delta:.7f} (Count: {convergence_counter}/{early_stopping_patience})")
                # Stop if loss remains below threshold & stable for `convergence_patience` epochs
                if convergence_counter >= early_stopping_patience:
                    print(f"Stopping early: Loss {total_loss:.7f} has remained below threshold for {early_stopping_patience} consecutive epochs with minimal change.")
                    break
            else:
                convergence_counter = 0  # Reset if loss changes significantly

        # perform validation
        if val_loader:
            with torch.no_grad():
                val_loss = 0
                tmp=0
                if not parametric_mode:
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device)
                        val_batch_loss, val_loss, tmp = batch_loss_eval(
                            val_batch, model, criterion, "energy", val_loss, 1.0, tmp)
                    val_loss /= len(val_loader)
                    history["validation_loss"].append(val_loss)
                    # check early stopping based on validation loss
                    if early_stopping_patience is not None:
                        if val_loss < best_val_loss - early_stopping_delta:
                            best_val_loss = val_loss
                            early_stopping_counter = 0  # Reset counter
                        else:
                            early_stopping_counter += 1
                            if early_stopping_counter >= early_stopping_patience:
                                print(f"Early stopping: No significant improvement in {early_stopping_patience} epochs.")
                                break
                else: # parametric mode 
                    for val_batch, val_batch_R, val_batch_P in val_loader:
                        val_batch = val_batch.to(device)
                        val_batch_R = val_batch_R.to(device)
                        val_batch_P = val_batch_P.to(device)
                        val_batch_loss, val_loss, tmp = batch_loss_eval_pm(
                                val_batch, base_potential_type, val_batch_R, pm, val_batch_P, model, criterion, 
                                "energy", val_loss, 1.0, tmp)
                    val_loss /= len(val_loader)
                    history["validation_loss"].append(val_loss)
                    # check early stopping based on validation loss
                    if early_stopping_patience is not None:
                        if val_loss < best_val_loss - early_stopping_delta:
                            best_val_loss = val_loss
                            early_stopping_counter = 0  # Reset counter
                        else:
                            early_stopping_counter += 1
                            if early_stopping_counter >= early_stopping_patience:
                                print(f"Early stopping: No significant improvement in {early_stopping_patience} epochs.")
                                break


        # Logging for every 1000 epoches
        if epoch % print_interval == 0:
            model.eval()  # Set model to evaluation mode (no dropout, batch norm fixed)
            # ====Notice that we will evaluate true loss here. This is a real loss
            #     The intermediate loss we stored may be different from true loss
            #     This is becasue we use the mini-batch. And adiabatic dataset has different number 
            #         of geometries compared to diabatic and permutation datasets.
            true_adiabatic_loss = 0
            true_diabatic_loss = 0
            true_permutation_loss = 0
            true_topology_loss = 0
            tmp=0
            with torch.no_grad():
                # Compute true adiabatic loss
                if not parametric_mode:
                    for batch in train_loader_test:
                        batch = batch.to(device)
                        adiabatic_batch_loss, true_adiabatic_loss, tmp = batch_loss_eval(
                            batch, model, criterion, "energy", true_adiabatic_loss, 1.0, tmp)
                        if topology_attention:
                            off_diag_elements = torch.cat(unique_elements[:,model.output_dim:], dim=0)
                            true_topology_loss += torch.mean(torch.sign(off_diag_elements))
                    true_adiabatic_loss /= len(train_loader)
                    if topology_attention and not permutation_loader:
                        true_topology_loss = torch.abs(true_topology_loss/len(train_loader))

                    # Compute true diabatic loss
                    if diabatic_loader:
                        for batch in diabatic_loader_test:
                            diabatic_batch_loss, true_diabatic_loss, tmp = batch_loss_eval(
                                batch, model, criterion, "DPEM", true_diabatic_loss, diabatic_weight, tmp)
                        true_diabatic_loss /= len(diabatic_loader)

                    # Compute true permutation loss
                    if permutation_loader:
                        for batch in permutation_loader_test:
                            permutation_batch_loss, true_permutation_loss, tmp = batch_loss_eval(
                            batch, model, criterion, "energy", true_permutation_loss, permutation_weight, tmp)
                        true_permutation_loss /= len(permutation_loader)
                        if topology_attention:
                            off_diag_elements = torch.cat(unique_elements[:,model.output_dim:], dim=0)
                            true_topology_loss += torch.mean(torch.sign(off_diag_elements))
                            true_topology_loss = torch.abs(true_topology_loss/(len(train_loader)+len(permutation_loader))) 

                else: # parametric mode
                    for batch, batch_R, batch_P in train_loader_test:
                        batch = batch.to(device)
                        batch_R = batch_R.to(device)
                        batch_P = batch_P.to(device)
                        adiabatic_batch_loss, true_adiabatic_loss, tmp = batch_loss_eval_pm(
                            batch, base_potential_type, batch_R, pm, batch_P,  
                            model, criterion, "energy", true_adiabatic_loss, 1.0, tmp)
                        if topology_attention:
                            off_diag_elements = torch.cat(unique_elements[:,model.output_dim:], dim=0)
                            true_topology_loss += torch.mean(torch.sign(off_diag_elements))
                    true_adiabatic_loss /= len(train_loader)
                    if topology_attention and not permutation_loader:
                        true_topology_loss = torch.abs(true_topology_loss/len(train_loader))

                    # Compute true diabatic loss
                    if diabatic_loader:
                        for batch in diabatic_loader_test:
                            diabatic_batch_loss, true_diabatic_loss, tmp = batch_loss_eval(
                                batch, model, criterion, "DPEM", true_diabatic_loss, diabatic_weight, tmp)
                        true_diabatic_loss /= len(diabatic_loader)

                    # Compute true permutation loss
                    if permutation_loader:
                        for batch, batch_R, batch_P in permutation_loader_test:
                            permutation_batch_loss, true_permutation_loss, tmp = batch_loss_eval(
                            batch, ase_potential_type, batch_R, pm, batch_P, 
                            model, criterion, "energy", true_permutation_loss, permutation_weight, tmp)
                        true_permutation_loss /= len(permutation_loader)
                        if topology_attention:
                            off_diag_elements = torch.cat(unique_elements[:,model.output_dim:], dim=0)
                            true_topology_loss += torch.mean(torch.sign(off_diag_elements))
                            true_topology_loss = torch.abs(true_topology_loss/(len(train_loader)+len(permutation_loader)))

                # Compute regularization term 
                if regularization:
                    true_regularization_term = sum(0.5 * torch.sum(param ** 2) for param in model.parameters())

                # Compute true total loss
                true_total_loss = (
                    true_adiabatic_loss
                    + diabatic_weight * true_diabatic_loss
                    + permutation_weight * true_permutation_loss
                    + topology_attention_weight*true_topology_loss
                )

            # Store true loss history
            history["true_total_loss"].append(true_total_loss)
            history["true_adiabatic_loss"].append(true_adiabatic_loss)
            history["true_diabatic_loss"].append(true_diabatic_loss)
            history["true_permutation_loss"].append(true_permutation_loss)
            history["true_topology_loss"].append(true_topology_loss)
            history["true_regularization_term"].append(true_regularization_term)
            print(f"Epoch {epoch} - True Loss Computed:"
                  f"Total_Training_Loss = {true_total_loss:.7f}, "
                  f"Adiabatic_Loss = {true_adiabatic_loss:.7f}, "
                  f"{'Diabatic_Loss = ' + f'{diabatic_weight*true_diabatic_loss:.7f}, ' if diabatic_loader else ''}"
                  f"{'Permutation_Loss = ' + f'{permutation_weight*true_permutation_loss:.7f}, ' if permutation_loader else ''}"
                  f"{'Topology_Loss = ' + f'{topology_attention_weight*topology_loss:.7f}, ' if topology_attention else ''}"
                  f"Loss_change (from batch) = {loss_change:.7f}, "
                  f"{'Validation_Loss = '+ f'{val_loss:.7f}, ' if val_loader else ''}"
                  f"{'Regularization_Term = ' + f'{regularization_weight*true_regularization_term:.7f}, ' if regularization else ''}", 
                  flush=True)
            # Switch back to training mode
            model.train()

        if scheduler:
            scheduler.step()

        if epoch % save_interval == 0:  # Save checkpoint every 100 epochs
            save_checkpoint(model, optimizer, scheduler, epoch, history, best_val_loss, 
                            early_stopping_counter, ema_loss, convergence_counter, 
                            previous_total_loss, checkpoint_path)

    return history


