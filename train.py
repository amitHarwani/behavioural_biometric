import numpy as np
import sys
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import math
import inspect
import matplotlib.pyplot as plt
import random
from collections import defaultdict

from model import ModelConfig, Model
from data_loader import get_training_dataloader, get_validation_dataloader, get_testing_dataloader

# import sys

# Open file for writing
# sys.stdout = open('output.txt', 'w')

def normalize(data, screen_dim_x, screen_dim_y, split: str = "train"):
    global means_for_normalization
    global stds_for_normalization

    # If mean and std. are not computed yet and the split is not train
    if (len(means_for_normalization) == 0 or len(stds_for_normalization) == 0) and split != "train":
        print("Pass Training Data First To Compute Mean and Std.")
        sys.exit(0)
        return
    
    # For training split
    if split == "train":
        # To store all sequences
        all_data = []
        for user in data:
            for session in user:
                for sequence in session:
                    # Append sequence to all data
                    all_data.append(sequence) # Sequence is mostly (seq_length, 62)

        # Combining to a single Nd array
        combined_array = np.concatenate(all_data, axis=0) # Shape (num_samples, 62)

        # Calculating the means and std from the training dataset for each feature/col
        means = np.mean(combined_array, axis=0) # Shape (62,)
        stds = np.std(combined_array, axis=0) # Shape (62,)

        # To normalize Keystroke timings: we are just dividing by 1000's to convert them into seconds
        means[0:9] = 0
        stds[0:9] = 1000

        # To normalize Keystroke codes: we are just dividing by 255
        means[9: 17] = 0
        stds[9: 17] = 255

        # Ignoring first and second order derivatives of IMU for Z-score normalization as they are already within a good range of -1 and 1
        means[23: 29] = 0
        stds[23: 29] = 1
        means[35: 41] = 0
        stds[35: 41] = 1
        means[47: 53] = 0
        stds[47: 53] = 1

        # For touch x, y: divide by the screen dimension to normalize it
        means[53] = 0
        stds[53] = screen_dim_x
        means[54] = 0
        stds[54] = screen_dim_y

        # Ignoring contact size for normalization, as its already normalized
        means[55] = 0
        stds[55] = 1

        # Assign to the global variable
        means_for_normalization = means.copy()
        stds_for_normalization = stds.copy()

    # Applying Z-score notmalization
    for user_idx, user in enumerate(data): 
        for sess_idx, session in enumerate(user):
            for seq_idx, sequence in enumerate(session): 
                # Clipping the touch x and y values within the screen dimension range
                sequence[:, 53] = np.clip(sequence[:, 53], 0, screen_dim_x)
                sequence[:, 54] = np.clip(sequence[:, 54], 0, screen_dim_y)
                
                data[user_idx][sess_idx][seq_idx] = (sequence - means_for_normalization) / stds_for_normalization

def merge_sequences(data, merge_length):
    merged_data = []
    for user in data:
        user_data = []
        for session in user:
            session_data = []
            # Flatten the session into a single array of shape (N*10, 62)
            flat_session = np.concatenate(session, axis=0)
            print("Flat Session shape", flat_session.shape)
            # Split into chunks of merge_length
            for i in range(0, len(flat_session), merge_length):
                if i + merge_length <= len(flat_session):
                    print("Adding from ",i, "to", i+merge_length)
                    session_data.append(flat_session[i:i + merge_length])
            if i < len(flat_session):
                session_data.append(flat_session[i:])
            user_data.append(session_data)
        merged_data.append(user_data)
    return merged_data


def merge_sequences_overlap(training_data, merge_length, overlap_length):
    merged_data = []
    step_size = merge_length - overlap_length  # How much we move each time
    for user in training_data:
        user_data = []
        for session in user:
            session_data = []
            flat_session = np.concatenate(session, axis=0)
            i = 0
            while i + merge_length <= len(flat_session):
                session_data.append(flat_session[i:i + merge_length])
                i += step_size  # Move by step_size instead of merge_length
            if i < len(flat_session):
                session_data.append(flat_session[i:])
            user_data.append(session_data)
        merged_data.append(user_data)
    return merged_data


# Splitting validation data
def split_training_data_by_sessions(data, validation_ratio=0.2):
    random.seed(42) # For reproducibility
    train_data = []
    val_data = []

    for user in data:
        # Shuffle the sessions for randomness
        random.shuffle(user)
        num_val_sessions = int(len(user) * validation_ratio)

        # Split sessions into training and validation
        val_data.append(user[:num_val_sessions])
        train_data.append(user[num_val_sessions:])

    return train_data, val_data

def intra_inter_maha_streaming(X, y, cov_inv, chunk_size=512):
    # X: (N,D), y: (N,), cov_inv: (D,D)
    N = X.shape[0]
    # center X once:
    mu = X.mean(0, keepdim=True)
    Xc = X - mu

    # 1) diag terms δ_i
    #    δ_i = x_i^T Σ⁻¹ x_i
    #    shape: (N,)
    δ = (Xc @ cov_inv * Xc).sum(dim=1)

    # prepare masks and accumulators
    lbl_eq = y.unsqueeze(1) == y.unsqueeze(0)
    # we will accumulate sum of d^2 over same‐class and diff‐class
    same_sum = 0.0
    same_count = 0
    diff_sum = 0.0
    diff_count = 0

    # 2) process in chunks of rows
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Xi = Xc[start:end]               # (K, D)
        δi = δ[start:end].unsqueeze(1)   # (K, 1)

        # compute block M_block = Xi Σ⁻¹ Xcᵀ → shape (K, N)
        M_block = Xi @ cov_inv @ Xc.t()  # (K, N)

        # compute squared-distance block: δ_i + δ_j - 2 M_ij
        # broadcast δ_i over columns, δ over rows
        d2_block = δi + δ.unsqueeze(0) - 2.0 * M_block

        # now mask and accumulate
        mask_block = lbl_eq[start:end]   # (K, N)
        same_sum  += d2_block[mask_block].sum().item()
        same_count+= mask_block.sum().item()

        diff_mask = ~mask_block
        diff_sum  += d2_block[diff_mask].sum().item()
        diff_count+= diff_mask.sum().item()

        # drop the block → memory freed before next iteration

    # convert mean squared distances → mean distances
    if same_count != 0:
        intra_maha = float((same_sum  / same_count)**0.5)
    else:
        intra_maha = float('nan')
    if diff_count != 0:
        inter_maha = float((diff_sum  / diff_count)**0.5)
    else:
        inter_maha = float('nan')

    return intra_maha, inter_maha

@torch.no_grad()
def validate(val_dataloader):
    # Putting the model in eval mode
    model.eval()
    losses = torch.zeros(len(val_dataloader))

    all_embs = []  # emb: (N: number of samples, D)
    all_labels = []  # (N,)
    for idx, batch in enumerate(val_dataloader):
        sequences = batch['sequences'] # (batch_size (B), sequence_length (T), embedding size (C))
        labels = batch['user_ids'] # User IDs (batch_size (B))
        temporal_attention_mask = batch['temporal_attention_mask'] # (B, T)
        channel_attention_mask = batch['channel_attention_mask'] # (B, C)

        # Moving the tensors to device
        sequences = sequences.to(device)
        labels = labels.to(device)
        temporal_attention_mask = temporal_attention_mask.to(device)
        channel_attention_mask = channel_attention_mask.to(device)

        emb, logits, loss = model(inputs=sequences, temporal_attn_mask= temporal_attention_mask, channel_attn_mask=channel_attention_mask, targets=labels)
        losses[idx] = loss

        all_embs.append(emb)      
        all_labels.append(labels)

    model.train() # Putting the model back in training mode

    X = torch.cat(all_embs, dim=0)   # (N, D)
    y = torch.cat(all_labels, dim=0) # (N,)

    N, D = X.shape

    # 1) Cosine‐similarity matrix
    X_norm = F.normalize(X)      # (N, D)
    cosim = X_norm @ X_norm.t()              # (N, N)

    # create masks
    lbl_eq = y.unsqueeze(1) == y.unsqueeze(0)  # (N, N)
    same_mask = lbl_eq.fill_diagonal_(False)   # exclude self‐pairs
    diff_mask = ~lbl_eq

    intra_cos = cosim[same_mask].mean()
    inter_cos = cosim[diff_mask].mean()

     # 2) Mahalanobis distances
    # compute covariance of X
    mean = X.mean(dim=0, keepdim=True)        # (1, D)
    Xc = X - mean                             # (N, D)
    # unbiased covariance matrix
    cov = (Xc.t() @ Xc) / (N - 1)             # (D, D)
    # regularize & invert
    eps = 1e-5
    cov_inv = torch.linalg.inv(cov + eps * torch.eye(D, device=X.device))

    intra_maha, inter_maha = intra_inter_maha_streaming(X, y, cov_inv)


    return {'loss': losses.mean().item(), 'intra_cos': intra_cos, 'inter_cos': inter_cos, 'intra_maha': intra_maha, 'inter_maha': inter_maha}


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
        k = 6, # Number of gaussians in GRE
        raw_d_model = 62, # Num. of features in raw dataset
        d_model= 64, # Num. of features
        seq_len= 10, # Block size/seq. length
        n_temporal_heads= 4, # Num. of temporal heads
        n_channel_heads= 5, # Num. of channel heads
        dropout= 0.2, # Dropout probability 
        n_layers= 5, # Number of layers or transformer encoders
        d_output_emb= 64, # Output embedding dimension
        n_users = 84, # Number of users (For classification)
        contrastive_loss_alpha = 1 # Contrastive loss importance hyperparameter (Alpha)
    )
    screen_dim_x=1903 # Screen width (For touch data)
    screen_dim_y=1920 # Screen height (For touch data)
    batch_size = 64 # Batch size
    same_user_ratio_in_batch = 0.25 # Ratio of same user pair sequences in the batch
    n_epochs = 10 # Number of epochs

    # Preprocessed files
    train_dataset_file = "training_users_data_tw1_sq10_maxk8.pickle"
    val_dataset_file = "validation_users_data_tw1_sq10_maxk8.pickle"
    test_dataset_file = "test_users_data_tw1_sq10_maxk8.pickle"

    # Loading the preprocessed objects
    with open(train_dataset_file, "rb") as infile:
        train_dataset = pickle.load(infile)

    with open(val_dataset_file, "rb") as infile:
        val_dataset = pickle.load(infile)

    with open(test_dataset_file, "rb") as infile:
        test_dataset = pickle.load(infile)

    # Combining the training and validation users.
    train_dataset = train_dataset + val_dataset

    # Splitting into train and val dataset by sessions -> Num. of users remain the same
    train_dataset, val_dataset = split_training_data_by_sessions(train_dataset, 0.2)

    # UNCOMMENT BELOW TO MERGE SEQUENCES i.e Have more than 10 second sequences
    # train_dataset = merge_sequences_overlap(train_dataset, 30, 20)
    # val_dataset = merge_sequences_overlap(val_dataset, 30, 20)

    # Means and std. deviations for normalization
    means_for_normalization = np.array([]) 
    stds_for_normalization = np.array([]) 

    # Normalizing the datasets
    normalize(train_dataset, screen_dim_x=screen_dim_x, screen_dim_y=screen_dim_y, split="train")
    normalize(val_dataset, screen_dim_x=screen_dim_x, screen_dim_y=screen_dim_y, split="val")
    normalize(test_dataset, screen_dim_x=screen_dim_x, screen_dim_y=screen_dim_y, split="test")

    # Identifying Device
    device = "cpu"
    if torch.cuda.is_available(): # GPU
        device="cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # Apple Silicon
        device="mps"
    print("Device: ", device)

    # Data Loaders | Training dataloader uses a Contrastive Sampler
    training_dataloader = get_training_dataloader(training_data=train_dataset, batch_size=batch_size, same_user_ratio=same_user_ratio_in_batch, sequence_length=model_config.seq_len, 
                                         required_feature_dim=model_config.raw_d_model, num_workers=1)
    
    val_dataloader = get_validation_dataloader(val_data=val_dataset, batch_size=batch_size, sequence_length=model_config.seq_len, 
                                            required_feature_dim=model_config.raw_d_model, num_workers=1)
    
    # Enabling Tensor Flow 32 (TF32) to make calculations faster 
    torch.set_float32_matmul_precision('high')

    # Model
    model = Model(model_config)
    num_of_parameters = sum(p.numel() for p in model.parameters())

    # Moving the model to the device used for training
    model.to(device)

    steps_per_epoch = len(training_dataloader) # Number of steps in an epoch OR number of batches
    total_steps_to_train = steps_per_epoch * n_epochs # Total number of steps to train the model

    print("steps_per_epoch", steps_per_epoch, "total_steps_to_train", total_steps_to_train)

    max_lr = 5e-3 # 0.0006 # 1e-4
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
    torch.set_printoptions(threshold=torch.inf)
    torch.autograd.set_detect_anomaly(True)

    # Best validation loss
    best_val_loss = math.inf 

    start_epoch_count = 0
    step_count = 0

    # To store validation losses list, for plotting 
    val_losses = []

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
        val_losses = checkpoint['val_losses']
        best_val_loss = checkpoint['loss']
        max_lr = checkpoint['max_lr']
        min_lr = checkpoint['min_lr']

    # Early stopper
    early_stopper = EarlyStopping(patience=10, min_delta=0.001)

    for epoch in range(start_epoch_count, n_epochs):
        for batch in training_dataloader:
            optimizer.zero_grad() # Zeroing the gradients

            sequences = batch['sequences'] # (batch_size (B), sequence_length (T), embedding size (C))
            labels = batch['user_ids'] # User IDs (batch_size (B))
            temporal_attention_mask = batch['temporal_attention_mask'] # (B, T)
            channel_attention_mask = batch['channel_attention_mask'] # (B, C)

            # Moving the tensors to device
            sequences = sequences.to(device)
            labels = labels.to(device)
            temporal_attention_mask = temporal_attention_mask.to(device)
            channel_attention_mask = channel_attention_mask.to(device)

            emb, logits, loss = model(inputs=sequences, temporal_attn_mask= temporal_attention_mask, channel_attn_mask=channel_attention_mask, targets=labels)

            # Backprop
            loss.backward()

            # Clipping the global norm of the gradient
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Updating the weights
            # determine and set the learning rate for this iteration
            lr = get_lr(step_count)
            for param_group in optimizer.param_groups: # Setting the learning rate in the optimizer
                param_group['lr'] = lr
            optimizer.step() # Updating the weights

            torch.cuda.synchronize() # wait for GPU to complete the work synchronizing with the CPU

            print(f"step {step_count} | lr: {lr} | loss: {loss} | norm: {norm:.4f}")
            print("--")

            step_count += 1

        # After every epoch - Validate
        val_metrics = validate(val_dataloader=val_dataloader)
        val_loss = val_metrics['loss']
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss}")
        print(val_metrics)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': val_loss,
                'val_metrics': val_metrics,
                'config': model_config,
                'step_count': step_count,
                'max_lr': max_lr,
                'min_lr': min_lr,
                'val_losses': val_losses,
                'means_for_normalization': means_for_normalization,
                'stds_for_normalization': stds_for_normalization,
                'same_user_ratio_in_batch': same_user_ratio_in_batch,
                'screen_dim_x': screen_dim_x,
                'screen_dim_y': screen_dim_y

            }
            torch.save(checkpoint, f"./checkpoints/best_val_epoch{epoch}_seq_len{model_config.seq_len}_v1.pt")
        
         # Checking for early stop
        early_stopper(val_loss)
        if early_stopper.early_stop:
            break

        
    


