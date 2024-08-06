
# Import libraries
import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Define training function
def train_model(
        model, dataset_train, dataset_val=None,
        batch_size=16, n_epochs=50, lr=1e-3,
        verbose=True,
    ):

    # Set up environment
    device = next(model.parameters()).device

    # Print status
    if verbose:
        status = ' '.join([
            f'Training model',
            f'with {sum(p.numel() for p in model.parameters())} parameters',
            f'on {device}.',
        ])
        print(status)

    # Set up data loaders and best model
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, 
        num_workers=os.cpu_count(), pin_memory=True  # Uncomment for local machine
    )
    if dataset_val is not None:
        dataloader_val = DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False, 
            num_workers=os.cpu_count(), pin_memory=True  # Uncomment for local machine
        )

    # Set up best model
    if dataset_val is not None:
        min_val_loss = float('inf')
        best_model_state = copy.deepcopy(model.state_dict())

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    max_grad_norm = 1/lr  # Max parameter update is 1
    
    # Set up loss function
    def loss_fn(y_pred, y, model=None):
        loss = F.mse_loss(y_pred, y)
        if model is not None:
            reg_loss = (
                sum(p.pow(2.0).sum() for p in model.parameters()) 
                / sum(p.numel() for p in model.parameters())
            )
        return loss + reg_loss

    # Train model
    for epoch in range(n_epochs):
        t = time.time()
        if verbose:
            print(f'Epoch {epoch+1}/{n_epochs}')

        # Initialize loss
        total_train_loss = 0

        # Iterate over batches
        for i, x, y in enumerate(dataloader_train):
            t_batch = time.time()  # Get batch time

            # Zero gradients
            optimizer.zero_grad()

            # Get batch
            y_pred = model(x)

            # Calculate loss
            loss = loss_fn(y_pred, y, model)

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Update loss
            total_train_loss += float(loss.item())

            # Print status
            if verbose and (
                    (i == 0)                                   # Print first
                    or (i == len(dataloader_train) - 1)        # Print last
                    or ((i + 1) % 50 == 0)                     # Print every 50
                    or (len(dataloader_train) < 20)            # Print all if small dataset
                ):
                status = ' '.join([                            # Set up status
                    f'--',                                     # Indent
                    f'Batch {i+1}/{len(dataloader_train)}',    # Batch number
                    f'({(time.time()-t_batch):.2f} s/batch)',  # Time per batch
                ])
                print(status)  # Print status


        # Get validation loss
        if dataset_val is not None:
            if verbose:
                print('Validating')
            total_val_loss = 0                                     # Initialize loss
            for i, x, y in enumerate(dataloader_val):              # Iterate over batches
                t_batch = time.time()                              # Get batch
                y_pred = model(x)                                  # Get output
                loss = loss_fn(y_pred, y, model)                   # Calculate loss
                total_val_loss += float(loss.item())               # Update loss
                if verbose and (
                        (i == 0)                                   # Print first
                        or (i == len(dataloader_val) - 1)          # Print last
                        or ((i + 1) % 50 == 0)                     # Print every 50
                        or (len(dataloader_val) < 20)              # Print all if small dataset
                    ):
                    status = ' '.join([                            # Set up status
                        f'--',                                     # Indent
                        f'Batch {i+1}/{len(dataloader_val)}',      # Batch number
                        f'({(time.time()-t_batch):.2f} s/batch)',  # Time per batch
                    ])
                    print(status)  # Print status
    
        # Save model if validation loss is lower
        if verbose:
            print('Updating model')
        if total_val_loss < min_val_loss:
            min_val_loss = total_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        # Print status
        if verbose:
            status = ':::' + '\n:::'.join([             # Set up status
                f'Train loss: {total_train_loss:.4e}',  # Print training loss
                f'Val loss: {total_val_loss:.4e}',      # Print validation loss
                f'Time: {time.time()-t:.2f} sec.'       # Print time
            ])
            print(status)

    # Load best model
    model.load_state_dict(best_model_state)
    
    # Return model
    if verbose:
        print('Training complete.')
    return model

