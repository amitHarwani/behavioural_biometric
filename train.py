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
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d


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


def compute_exact_eer(y_true, y_score, pos_label=1):
    # 1) Compute discrete ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    fnr = 1 - tpr

    # 2) Build continuous interpolants FPR(x), FNR(x) over x in [0,1]
    #    We parameterize by FPR, so we interpolate TPR and thresholds as functions of FPR.
    tpr_interp = interp1d(fpr, tpr)
    thresh_interp = interp1d(fpr, thresholds)

    # 3) Find EER by solving 1 - x - TPR(x) = 0  <=>  FPR = FNR
    #    i.e. find x in [0,1] such that x = 1 - TPR(x)
    eer = brentq(lambda x: x - (1 - tpr_interp(x)), 0.0, 1.0)  # root-finding :contentReference[oaicite:0]{index=0}

    # 4) Recover the threshold at that operating point
    eer_threshold = float(thresh_interp(eer))

    return eer, eer_threshold

# Example usage:
# eer_value, thresh = compute_exact_eer(y_true, y_scores)
# print(f"EER = {eer_value:.4f} ({eer_value*100:.2f}%), at threshold = {thresh:.4f}")

@torch.no_grad()
def validate(val_dataset, num_of_enrollment_sessions, num_of_verify_sessions, device, batch_size=32):
    """
    Validates the model by enrolling users and verifying them using their sequences.
    Calculates the Equal Error Rate (EER) for cosine similarity and Mahalanobis distance separately.
    
    Args:
        val_dataset: Dataset containing user sessions
        num_of_enrollment_sessions: Number of sessions to use for enrollment
        num_of_verify_sessions: Number of sessions to use for verification
        batch_size: Batch size for processing sequences
        
    Returns:
        tuple: (average_cosine_eer, average_mahalanobis_eer)
    """
    # Putting the model in evaluation mode
    model.eval()
    
    # To store EERs for all users
    cosine_user_eers = []
    exact_cosine_user_eers = []
    mahalanobis_user_eers = []
    exact_mahalanobis_user_eers = []
    
    total_users = len(val_dataset)  
    
    for user_idx, user_sessions in enumerate(val_dataset):
        # Enrollment and verification sessions of the user
        enrollment_sessions = user_sessions[:num_of_enrollment_sessions]
        verification_sessions = user_sessions[num_of_enrollment_sessions:num_of_enrollment_sessions + num_of_verify_sessions]
        
        # Process enrollment sequences in batches
        enrollment_embeddings = []

        # Validation Data Loader
        genuine_user_enrollment_loader = get_validation_dataloader([enrollment_sessions], batch_size=batch_size, 
                                  sequence_length=model_config.seq_len, required_feature_dim=model_config.raw_d_model, num_workers=1)
        
        for batch in genuine_user_enrollment_loader:
            sequences = batch['sequences'].to(device) # (batch_size (B), sequence_length (T), embedding size (C))
            labels = batch['user_ids'].to(device) # User IDs (batch_size (B))
            temporal_attention_mask = batch['temporal_attention_mask'].to(device) # (B, T)
            channel_attention_mask = batch['channel_attention_mask'].to(device) # (B, C)

            # Passing it through the model
            emb, logits, cos_loss, cross_entropy_loss, loss = model(inputs=sequences, temporal_attn_mask= temporal_attention_mask, 
                                                                    channel_attn_mask=channel_attention_mask, targets=labels)
            enrollment_embeddings.append(emb)
        
        enrollment_embeddings = torch.cat(enrollment_embeddings, dim=0).to(device)
        
        # Compute mean embedding for the user
        user_mean_embedding = enrollment_embeddings.mean(dim=0, keepdim=True)  # (1, d_output_emb)

        # Prepare Mahalanobis distance covariance matrix
        centered_embeddings = enrollment_embeddings - user_mean_embedding
        precision_matrix = LedoitWolf().fit(centered_embeddings.cpu().numpy()).precision_.astype(np.float32)
        precision_matrix = torch.tensor(precision_matrix, dtype=torch.float32).to(device)

        # Initialize score lists for this user
        cosine_genuine_scores = []
        cosine_impostor_scores = []
        mahalanobis_genuine_scores = []
        mahalanobis_impostor_scores = []

         # Validation Data Loader
        genuine_user_verification_loader = get_validation_dataloader([verification_sessions], batch_size=batch_size, 
                                  sequence_length=model_config.seq_len, required_feature_dim=model_config.raw_d_model, num_workers=1)
        
        
        # Process verification sequences in batches
        verification_embeddings = []
        for batch in genuine_user_verification_loader:
            sequences = batch['sequences'].to(device) # (batch_size (B), sequence_length (T), embedding size (C))
            labels = batch['user_ids'].to(device) # User IDs (batch_size (B))
            temporal_attention_mask = batch['temporal_attention_mask'].to(device) # (B, T)
            channel_attention_mask = batch['channel_attention_mask'].to(device) # (B, C)

            # Passing it through the model
            emb, logits, cos_loss, cross_entropy_loss, loss = model(inputs=sequences, temporal_attn_mask= temporal_attention_mask, 
                                                                    channel_attn_mask=channel_attention_mask, targets=labels)
            verification_embeddings.append(emb)
        
        verification_embeddings = torch.cat(verification_embeddings, dim=0).to(device)

        # Calculate cosine similarity scores
        cosine_scores_matrix = F.cosine_similarity(
            verification_embeddings.unsqueeze(1),  # (num_verification_sequences, 1, d_output_emb)
            enrollment_embeddings.unsqueeze(0),    # (1, num_enrollment_sequences, d_output_emb)
            dim=-1
        )  # Result: (num_verification_sequences, num_enrollment_sequences)

        # Average similarity across all enrollment sequences
        cosine_scores = cosine_scores_matrix.mean(dim=1)  # (num_verification_sequences,)
        cosine_genuine_scores.extend(cosine_scores.cpu().tolist())

        # Calculate Mahalanobis distance scores
        centered_verification_embeddings = verification_embeddings - user_mean_embedding
        
        # Process Mahalanobis calculation in batches for large verification sets
        maha_scores = []
        for i in range(0, centered_verification_embeddings.size(0), batch_size):
            batch = centered_verification_embeddings[i:i+batch_size]
            # For Mahalanobis, smaller values mean closer match, so we negate for consistency
            # with cosine (where larger means better match)
            batch_scores = -torch.diag(batch @ precision_matrix @ batch.T)
            maha_scores.append(batch_scores)
        
        maha_scores = torch.cat(maha_scores, dim=0)  # (num_verification_sequences,)
        mahalanobis_genuine_scores.extend(maha_scores.cpu().tolist())

        # Impostor scores: Compare sequences of other users
        for other_user_idx in np.random.permutation(total_users):
            if other_user_idx == user_idx:
                continue  # Skip the same user
            
            # Sessions of the user
            other_user_sessions = val_dataset[other_user_idx][:num_of_verify_sessions]
            
            # Sequences of the verify sessions
            # other_user_sequences = [
            #     sequence
            #     for session in other_user_sessions[:num_of_verify_sessions]
            #     for sequence in session
            # ]

            # Limit number of sequences per impostor to avoid bias
            # max_sequences = min(len(other_user_sequences), 20)
            # indices = np.random.choice(len(other_user_sequences), max_sequences, replace=False)
            # selected_sequences = [other_user_sequences[i] for i in indices]
            
             # Validation Data Loader - 1 user, 1 session, selected sequences
            imposter_user_verification_loader = get_validation_dataloader([other_user_sessions], batch_size=batch_size, 
                                  sequence_length=model_config.seq_len, required_feature_dim=model_config.raw_d_model, num_workers=1)
                    
            # Process impostor sequences in batches
            other_user_embeddings = []
            for batch in imposter_user_verification_loader:
                sequences = batch['sequences'].to(device) # (batch_size (B), sequence_length (T), embedding size (C))
                labels = batch['user_ids'].to(device) # User IDs (batch_size (B))
                temporal_attention_mask = batch['temporal_attention_mask'].to(device) # (B, T)
                channel_attention_mask = batch['channel_attention_mask'].to(device) # (B, C)

                # Passing it through the model
                emb, logits, cos_loss, cross_entropy_loss, loss = model(inputs=sequences, temporal_attn_mask= temporal_attention_mask, 
                                                                        channel_attn_mask=channel_attention_mask, targets=labels)
                
                other_user_embeddings.append(emb)
                
            other_user_embeddings = torch.cat(other_user_embeddings, dim=0).to(device)

            # Cosine similarity for impostor sequences
            cosine_scores_matrix = F.cosine_similarity(
                other_user_embeddings.unsqueeze(1),  # (num_impostor_sequences, 1, d_output_emb)
                enrollment_embeddings.unsqueeze(0),  # (1, num_enrollment_sequences, d_output_emb)
                dim=-1
            )  # Result: (num_impostor_sequences, num_enrollment_sequences)

            # Average similarity across all enrollment sequences
            cosine_scores = cosine_scores_matrix.mean(dim=1)  # (num_impostor_sequences,)
            cosine_impostor_scores.extend(cosine_scores.cpu().tolist())

            # Mahalanobis distance for impostor sequences
            centered_other_embeddings = other_user_embeddings - user_mean_embedding
            
            # Process Mahalanobis calculation in batches
            maha_scores = []
            for i in range(0, centered_other_embeddings.size(0), batch_size):
                batch = centered_other_embeddings[i:i+batch_size]
                batch_scores = -torch.diag(batch @ precision_matrix @ batch.T)
                maha_scores.append(batch_scores)
            
            maha_scores = torch.cat(maha_scores, dim=0)
            mahalanobis_impostor_scores.extend(maha_scores.cpu().tolist())
                                    

        # Calculate EER for the user using cosine similarity
        if len(cosine_genuine_scores) > 0 and len(cosine_impostor_scores) > 0:
            cosine_genuine_labels = np.ones(len(cosine_genuine_scores))
            cosine_impostor_labels = np.zeros(len(cosine_impostor_scores))

            cosine_all_scores = np.concatenate([cosine_genuine_scores, cosine_impostor_scores])
            cosine_all_labels = np.concatenate([cosine_genuine_labels, cosine_impostor_labels])
            
            exact_cosine_eer, _ = compute_exact_eer(cosine_all_labels, cosine_all_scores)
            exact_cosine_user_eers.append(exact_cosine_eer)

            fpr, tpr, thresholds = roc_curve(cosine_all_labels, cosine_all_scores)
            fnr = 1 - tpr
            
            # Check if there are valid values to compute EER
            if np.any(~np.isnan(fpr)) and np.any(~np.isnan(fnr)):
                valid_idx = ~np.isnan(fpr) & ~np.isnan(fnr)
                fpr_valid = fpr[valid_idx]
                fnr_valid = fnr[valid_idx]
                
                if len(fpr_valid) > 0:
                    idx = np.nanargmin(np.abs(fpr_valid - fnr_valid))
                    cosine_eer = fpr_valid[idx]
                    cosine_user_eers.append(cosine_eer)
                    
                    # Optionally compute AUC to evaluate performance
                    auc_score = roc_auc_score(cosine_all_labels, cosine_all_scores)
                    print(f"User {user_idx} - Cosine EER: {cosine_eer:.4f} - Exact: {exact_cosine_eer:.4f}, AUC: {auc_score:.4f}")

        # Calculate EER for the user using Mahalanobis distance
        if len(mahalanobis_genuine_scores) > 0 and len(mahalanobis_impostor_scores) > 0:
            mahalanobis_genuine_labels = np.ones(len(mahalanobis_genuine_scores))
            mahalanobis_impostor_labels = np.zeros(len(mahalanobis_impostor_scores))

            mahalanobis_all_scores = np.concatenate([mahalanobis_genuine_scores, mahalanobis_impostor_scores])
            mahalanobis_all_labels = np.concatenate([mahalanobis_genuine_labels, mahalanobis_impostor_labels])

            exact_mahalanobis_eer, _ = compute_exact_eer(mahalanobis_all_labels, mahalanobis_all_scores)
            exact_mahalanobis_user_eers.append(exact_mahalanobis_eer)
            
            fpr, tpr, thresholds = roc_curve(mahalanobis_all_labels, mahalanobis_all_scores)
            fnr = 1 - tpr
            
            # Check if there are valid values to compute EER
            if np.any(~np.isnan(fpr)) and np.any(~np.isnan(fnr)):
                valid_idx = ~np.isnan(fpr) & ~np.isnan(fnr)
                fpr_valid = fpr[valid_idx]
                fnr_valid = fnr[valid_idx]
                
                if len(fpr_valid) > 0:
                    idx = np.nanargmin(np.abs(fpr_valid - fnr_valid))
                    mahalanobis_eer = fpr_valid[idx]
                    mahalanobis_user_eers.append(mahalanobis_eer)
                    
                    # Optionally compute AUC to evaluate performance
                    auc_score = roc_auc_score(mahalanobis_all_labels, mahalanobis_all_scores)
                    print(f"User {user_idx} - Mahalanobis EER: {mahalanobis_eer:.4f} - Exact: {exact_mahalanobis_eer:.4f}, AUC: {auc_score:.4f}")

    # Average EER across all users
    avg_cosine_eer = np.mean(cosine_user_eers) if cosine_user_eers else float('inf')
    avg_mahalanobis_eer = np.mean(mahalanobis_user_eers) if mahalanobis_user_eers else float('inf')

    avg_exact_cosine_eer = np.mean(exact_cosine_user_eers) if exact_cosine_user_eers else float('inf')
    avg_exact_mahalanobis_eer = np.mean(exact_mahalanobis_user_eers) if exact_mahalanobis_user_eers else float('inf')
    
    
    print(f"\nValidation Complete:")
    print(f"Total users: {total_users}")
    print(f"Average Cosine EER: {avg_cosine_eer:.4f} (from {len(cosine_user_eers)} users)")
    print(f"Average Mahalanobis EER: {avg_mahalanobis_eer:.4f} (from {len(mahalanobis_user_eers)} users)")
    print(f"Average Exact Cosine EER: {avg_exact_cosine_eer:.4f} (from {len(exact_cosine_user_eers)} users)")
    print(f"Average Exact Mahalanobis EER: {avg_exact_mahalanobis_eer:.4f} (from {len(exact_mahalanobis_user_eers)} users)")

    model.train()
    # Return both average EERs
    return avg_cosine_eer, avg_mahalanobis_eer, avg_exact_cosine_eer, avg_exact_mahalanobis_eer

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
        k = 18, # Number of gaussians in GRE
        raw_d_model = 62, # Num. of features in raw dataset
        d_model= 64, # Num. of features
        seq_len= 30, # Block size/seq. length
        n_temporal_heads= 4, # Num. of temporal heads
        n_channel_heads= 5, # Num. of channel heads
        dropout= 0.1, # Dropout probability 
        n_layers= 5, # Number of layers or transformer encoders
        d_output_emb= 64, # Output embedding dimension
        n_users = 84, # Number of users (For classification)
        contrastive_loss_alpha = 1 # Contrastive loss importance hyperparameter (Alpha)
    )
    screen_dim_x=1903 # Screen width (For touch data)
    screen_dim_y=1920 # Screen height (For touch data)
    batch_size = 128 # Batch size
    same_user_ratio_in_batch = 0.25 # Ratio of same user pair sequences in the batch
    n_epochs = 100 # Number of epochs

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


    # UNCOMMENT BELOW TO MERGE SEQUENCES i.e Have more than 10 second sequences
    train_dataset = merge_sequences_overlap(train_dataset, merge_length=30, overlap_length=25)
    val_dataset = merge_sequences_overlap(val_dataset, merge_length=30, overlap_length=25)

    # Means and std. deviations for normalization
    means_for_normalization = np.array([]) 
    stds_for_normalization = np.array([]) 

    # Normalizing the datasets
    normalize(train_dataset, screen_dim_x=screen_dim_x, screen_dim_y=screen_dim_y, split="train")
    normalize(val_dataset, screen_dim_x=screen_dim_x, screen_dim_y=screen_dim_y, split="val")
    # normalize(test_dataset, screen_dim_x=screen_dim_x, screen_dim_y=screen_dim_y, split="test")

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

    max_lr = 5e-3 # 0.0006 # 1e-4
    min_lr = 1e-4 # 0.00006
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
    torch.autograd.set_detect_anomaly(True)

    # Best Validation EER's
    best_cosine_eer = math.inf
    best_maha_eer = math.inf

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
        best_cosine_eer = checkpoint['best_cosine_eer']
        best_maha_eer = checkpoint['best_maha_eer']
        max_lr = checkpoint['max_lr']
        min_lr = checkpoint['min_lr']

    # Early stopper
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)

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

            emb, logits, cos_loss, cross_entropy_loss, loss = model(inputs=sequences, temporal_attn_mask= temporal_attention_mask, 
                                                                    channel_attn_mask=channel_attention_mask, targets=labels)

            # Backprop
            cos_loss.backward()

            # Clipping the global norm of the gradient
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Updating the weights
            # determine and set the learning rate for this iteration
            lr = get_lr(step_count)
            for param_group in optimizer.param_groups: # Setting the learning rate in the optimizer
                param_group['lr'] = lr
            optimizer.step() # Updating the weights

            torch.cuda.synchronize() # wait for GPU to complete the work synchronizing with the CPU

            print(f"step {step_count} | lr: {lr} | cos_loss: {cos_loss} | norm: {norm:.4f}")
            print("--")

            step_count += 1

        if epoch % 5 == 0 and epoch != 0:
            # After every epoch - Validate
            avg_cosine_eer, avg_mahalanobis_eer, avg_exact_cosine_eer, avg_exact_mahalanobis_eer = validate(val_dataset=val_dataset, num_of_enrollment_sessions=4, num_of_verify_sessions=4, device=device, batch_size=32)
            val_eers.append((avg_cosine_eer, avg_mahalanobis_eer, avg_exact_cosine_eer, avg_exact_mahalanobis_eer))
            # Relying on mahalonobis eer
            # if avg_mahalanobis_eer < best_maha_eer:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_cosine_eer': avg_cosine_eer,
                'best_maha_eer': avg_mahalanobis_eer,
                'best_exact_cosine_eer': avg_exact_cosine_eer,
                'best_exact_maha_eer': avg_exact_mahalanobis_eer,
                'config': model_config,
                'step_count': step_count,
                'max_lr': max_lr,
                'min_lr': min_lr,
                'val_eers': val_eers,
                'means_for_normalization': means_for_normalization,
                'stds_for_normalization': stds_for_normalization,
                'same_user_ratio_in_batch': same_user_ratio_in_batch,
                'screen_dim_x': screen_dim_x,
                'screen_dim_y': screen_dim_y

            }
            torch.save(checkpoint, f"./checkpoints/best_val_epoch{epoch}_seq_len{model_config.seq_len}_run3.pt")
        
            # Checking for early stop
            early_stopper(avg_mahalanobis_eer)
            if early_stopper.early_stop:
                break
    
    torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_cosine_eer': best_cosine_eer,
                'best_maha_eer': best_maha_eer,
                'val_eers': val_eers,
                'config': model_config,
                'step_count': step_count,
                'max_lr': max_lr,
                'min_lr': min_lr,
                'val_eers': val_eers,
                'means_for_normalization': means_for_normalization,
                'stds_for_normalization': stds_for_normalization,
                'same_user_ratio_in_batch': same_user_ratio_in_batch,
                'screen_dim_x': screen_dim_x,
                'screen_dim_y': screen_dim_y}, 
               f"./checkpoints/final_run3.pt")

        
    


