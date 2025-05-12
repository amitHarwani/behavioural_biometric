import numpy as np
import sys
import pickle
import torch
import torch.nn.functional as F
import os
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import math
import inspect
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d


from model_basic import ModelConfig, Model
from data_loader import get_training_dataloader
from validation import validate

# import sys

# Open file for writing
# sys.stdout = open('output.txt', 'w')

def normalize_and_init_dataset(data, screen_dim_x, screen_dim_y, split="train"):

    idx = 0 # Global index of sequences across all users and sessions
    sequences = []
    user_ids = []
    user_to_indices = defaultdict(list) # Indices of sequences belonging to the user.

    for u_idx, user in enumerate(data):
        for s_idx, session in enumerate(user):
            for seq_idx, sequence in enumerate(session if split == "train" else session[:50]):
                sequence[:, 0:9] /= 1000 # Times - Normalized to seconds
                sequence[:, 9] /= 255 # Keycode Normalization
                sequence[:, 10:13] /= 10 # Accelerometer x, y, z
                sequence[:, 28:31] /= 100 # Magnetometer x, y, z
                sequence[:, 44: 50] /= 1000 # Accelerometer and Gyroscope FFT
                sequence[:, 50: 53] /= 10000 # Magnetometer FFT

                # Removing touch data.
                data[u_idx][s_idx][seq_idx] = np.delete(sequence, [37, 38, 39, 40, 41, 42, 43, 53, 54], axis=1)

                sequences.append(data[u_idx][s_idx][seq_idx]) # Appending the sequence ndarray to sequences list
                user_ids.append(u_idx) # Appending the user id of the sequence to user_ids
                user_to_indices[u_idx].append(idx) # Appending to user_to_indices 
                idx += 1

                # Touch x, y clipping
                # sequence[:, 37] = np.clip(sequence[:, 37], 0, screen_dim_x) 
                # sequence[:, 38] = np.clip(sequence[:, 38], 0, screen_dim_y)
                # Touch normalization 
                # sequence[:, 37] /= screen_dim_x
                # sequence[:, 38] /= screen_dim_y

                # sequence[:, 53:55] /= 100 # Touch FFT
    
    return sequences, user_ids, user_to_indices

    
def extract_fft_features(data: pd.DataFrame):
    # Columns for which fft needs to be calculated
    fft_cols = ["a_x", "a_y", "a_z","g_x", "g_y", "g_z","m_x", "m_y", "m_z","t_x", "t_y"]
    
    # Calculating fft for each of the columns
    for col in fft_cols:
        data[f"{col}_fft"] = np.abs(np.fft.fft(data[col].values))

    return data

def merge_sequences_overlap(data: list[list[pd.DataFrame]], merge_length, overlap_length):
    merged_data = []
    step_size = merge_length - overlap_length  # How much we move each time

    num_users_done = 0
    for user in data:
        user_data = []
        for session in user:
            session_values = session.to_numpy(dtype=np.float32)
            final_session_data = []
            i = 0
            while i + merge_length <= len(session):
                sequence = session_values[i: i + merge_length]
                sequence = extract_fft_features(pd.DataFrame(sequence, columns=session.columns)).to_numpy()
                final_session_data.append(sequence) # (merge_length, 55)
                i += step_size  # Move by step_size instead of merge_length

            # Remaining rows: These will be padded.
            # if i < len(session):
            #     sequence = session_values[i:]
            #     sequence = extract_fft_features(pd.DataFrame(sequence, columns=session.columns)).to_numpy()
            #     session_data.append(sequence)
            user_data.append(final_session_data)
        merged_data.append(user_data)
        num_users_done += 1
        print("Num users done", num_users_done)
    return merged_data


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait for improvement.
            delta_ratio (float): Fraction of best loss to use as min_delta.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True

if __name__ == "__main__":
    # Parameters
    model_config = ModelConfig(
        n_modalities = 2,
        raw_d_model = 46, # Num. of features in raw dataset
        d_model= 64, # Num. of features
        seq_len= 200, # Block size/seq. length
        n_temporal_heads= 4, # Num. of temporal heads
        n_layers= 5, # Number of layers or transformer encoders
        n_users = 79, # Number of users (For classification)
        contrastive_loss_alpha = 1 # Contrastive loss importance hyperparameter (Alpha)
    )
    screen_dim_x=1903 # Screen width (For touch data)
    screen_dim_y=1920 # Screen height (For touch data)
    batch_size = 16 # Batch size
    actual_batch_size = 64
    accum_steps = actual_batch_size // batch_size
    n_epochs = 10 # Number of epochs
    overlap_len = 100
    version = "v1"
    

    # Preprocessed files
    train_dataset_file = f"{version}_training_users_data_tw10ms.pickle"
    val_dataset_file = f"{version}_validation_users_data_tw10ms.pickle"

    train_dataset_merged_file = f"{version}_merged_training_users_data_tw10ms.pickle"
    val_dataset_merged_file = f"{version}_merged_validation_users_data_tw10ms.pickle"

    # If the preprocessed files have been merged
    if os.path.exists(train_dataset_merged_file) and os.path.exists(val_dataset_merged_file):
        # Loading the preprocessed merged objects
        with open(train_dataset_merged_file, "rb") as infile:
            train_dataset = pickle.load(infile)

        with open(val_dataset_merged_file, "rb") as infile:
            val_dataset = pickle.load(infile)
    else:
        # Loading the preprocessed objects
        with open(train_dataset_file, "rb") as infile:
            train_dataset = pickle.load(infile)

        with open(val_dataset_file, "rb") as infile:
            val_dataset = pickle.load(infile)

        # Merging the sequences
        train_dataset = merge_sequences_overlap(train_dataset, merge_length=model_config.seq_len, overlap_length=overlap_len)
        val_dataset = merge_sequences_overlap(val_dataset, merge_length=model_config.seq_len, overlap_length=overlap_len)

        # Inserting into the file
        with open(train_dataset_merged_file, "wb") as outfile:
            pickle.dump(train_dataset, outfile)
        with open(val_dataset_merged_file, "wb") as outfile:
            pickle.dump(val_dataset, outfile)
        

    print("Starting Normalization")
    # Normalizing the datasets
    train_sequences, train_user_ids, train_user_to_indices = normalize_and_init_dataset(train_dataset, screen_dim_x=screen_dim_x, screen_dim_y=screen_dim_y, split="train")
    val_sequences, val_user_ids, val_user_to_indices = normalize_and_init_dataset(val_dataset, screen_dim_x=screen_dim_x, screen_dim_y=screen_dim_y, split="val")
    print("Normalization Complete")

    print("Train sequences", len(train_user_ids), "Val sequences", len(val_user_ids))

    # Identifying Device
    device = "cpu"
    if torch.cuda.is_available(): # GPU
        device="cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # Apple Silicon
        device="mps"
    print("Device: ", device)

    # Data Loaders | Training dataloader uses a Contrastive Sampler
    training_dataloader = get_training_dataloader(train_sequences=train_sequences, train_user_ids=train_user_ids, train_user_to_indices=train_user_to_indices, 
                                                  batch_size=batch_size, sequence_length=model_config.seq_len, num_workers=0)
    
    
    # Enabling Tensor Flow 32 (TF32) to make calculations faster 
    torch.set_float32_matmul_precision('high')

    # Model
    model = Model(model_config)
    num_of_parameters = sum(p.numel() for p in model.parameters())
    print("Num. of parameters", num_of_parameters)
    # Moving the model to the device used for training
    model.to(device)

    steps_per_epoch = len(training_dataloader) # Number of steps in an epoch OR number of batches
    total_steps_to_train = steps_per_epoch * n_epochs # Total number of steps to train the model

    print("steps_per_epoch", steps_per_epoch, "total_steps_to_train", total_steps_to_train)

    max_lr = 6e-4 # 0.0006 # 1e-4
    min_lr = max_lr * 0.1 # 0.00006
    warmup_steps = int(total_steps_to_train * 0.1) # 10% of total steps

    # Cosine decay with linear warmup learning rate schedule
    def get_lr(iteration):
        # Linear warmup
        if iteration < warmup_steps:
            return max_lr * ((iteration + 1) / warmup_steps)
        if iteration >= total_steps_to_train:
            return min_lr
        # Cosine decay
        decay_ratio = (iteration - warmup_steps) / (total_steps_to_train - warmup_steps) # Between 0 and 1
        assert 0<=decay_ratio<=1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff  starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr) 
    
    # Optimizer
    # Create AdamW optimizer and use the fused version if it is available - when running on CUDA
    # Instead of iterating over all the tensors and updating them which would launch many kernels, Fused would fuse all these kernels 
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, fused=use_fused) # AdamW optimizer

    # Tell PyTorch to print the full tensor
    # torch.autograd.set_detect_anomaly(True)

    start_epoch_count = 0
    step_count = 0

    # To store validation eers list, for plotting 
    val_eers = []

    # Loading the model from checkpoint if training is being continued
    mode = {'type': "SCRATCH", 'checkpoint_file': ""}
    if mode['type'] == "RESUME":
        checkpoint = torch.load(mode['checkpoint_file'], weights_only=False)
        model_config = checkpoint['config']
        model = Model(model_config)
        model.to(device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch_count = checkpoint['epoch'] + 1
        step_count = checkpoint['step_count']
        val_eers = checkpoint['val_eers']
        max_lr = checkpoint['max_lr']
        min_lr = checkpoint['min_lr']

    # Early stopper
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)
    
    for epoch in range(start_epoch_count, n_epochs):
        loss_accum = 0.0
        for batch_step, batch in enumerate(training_dataloader):
            print("Batch-Step", batch_step)
            sequences = batch['sequences'].to(device) # (batch_size (B), sequence_length (T), embedding size (C))
            labels = batch['user_ids'].to(device) # User IDs (batch_size (B))
            modality_mask = batch['modality_mask'].to(device) # (B,T, 3)

            emb, logits, cos_loss, cross_entropy_loss, loss = model(inputs=sequences, modality_mask=modality_mask, targets=labels)

            cos_loss = cos_loss / accum_steps
            loss_accum += cos_loss.detach()
            # Backprop
            cos_loss.backward()

            # Once the desired batch size is reached
            if (batch_step + 1) % accum_steps == 0:
                # Clipping the global norm of the gradient
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

                # Updating the weights
                # determine and set the learning rate for this iteration
                lr = get_lr(step_count)
                for param_group in optimizer.param_groups: # Setting the learning rate in the optimizer
                    param_group['lr'] = lr
                optimizer.step() # Updating the weights

                optimizer.zero_grad() # Zeroing the gradients

                torch.cuda.synchronize() # wait for GPU to complete the work synchronizing with the CPU

                print(f"step {step_count} | lr: {lr} | cos_loss: {loss_accum.item():.6f} | norm: {norm:.4f}")
                print("--")
                loss_accum = 0.0
                step_count += 1

        # After every epoch - Validate
        avg_cos_eer, avg_maha_eer = validate(model=model, val_sequences=val_sequences, val_user_ids=val_user_ids, val_user_to_indices=val_user_to_indices,
                                             device=device, batch_size=16)
        print(f"Validation Result -------- avg_cos_eer: {avg_cos_eer:.4f} | avg_maha_eer: {avg_maha_eer:.4f}")
        val_eers.append((avg_cos_eer, avg_maha_eer))
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'avg_cos_eer': avg_cos_eer,
            'avg_maha_eer': avg_maha_eer,
            'config': model_config,
            'step_count': step_count,
            'max_lr': max_lr,
            'min_lr': min_lr,
            'val_eers': val_eers,
            'screen_dim_x': screen_dim_x,
            'screen_dim_y': screen_dim_y

        }
        torch.save(checkpoint, f"./checkpoints/v1_test_run1_{epoch}.pt")
    

        
    


