import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import math
import inspect
import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.metrics import roc_curve, roc_auc_score
from model import Model, ModelConfig
from data_loader import get_training_dataloader, get_testing_dataloader



def prepare_validation_model(model: Model, num_of_valid_users,  device):
     # Copy of the model for validation
    validation_model = copy.deepcopy(model)

    # Update the classifer layer and the model config with the num of valid users
    validation_model.config.n_users = num_of_valid_users
    validation_model.classifier = nn.Linear(validation_model.config.d_output_emb, num_of_valid_users, bias=False)

    # Moving the model to the training device
    validation_model.to(device)

    return validation_model

def prepare_dataset(dataset, num_of_valid_users, num_of_enrollment_sessions):
    random.seed(42)
    # Indices of valid users, selected at random
    valid_user_indices = random.sample(range(len(dataset)), num_of_valid_users)
    print("Valid User Indices", valid_user_indices)

    # Valid users from the dataset
    valid_users = [dataset[i] for i in valid_user_indices]

    # Unauthorized users data from the validation dataset
    unauthorized_users_data = [dataset[i] for i in range(len(dataset)) if i not in valid_user_indices]

    # Enrollment and verification data for valid users
    valid_users_enrollment_data = []
    valid_users_verify_data = []

    # For all the valid users
    for user in valid_users:
        # Filter out sessions with keystroke data, 0th column as the first col is hold latency
        valid_sessions = [ idx for idx, session in enumerate(user) if any(sequence[:, 0].any() != 0 for sequence in session) ]
                    
        # Randomly select the num_of_enrollment_sessions from the sessions which the users has
        enrollment_session_indices = random.sample(valid_sessions, num_of_enrollment_sessions)
        print("User Enrollment Session Indices", enrollment_session_indices)

        # Adding enrollment sessions to the valid_users_enrollment_data
        valid_users_enrollment_data.append([user[i] for i in enrollment_session_indices])

        # Adding verification sessions to the valid_users_verify_data
        valid_users_verify_data.append([user[i] for i in range(len(user)) if i not in enrollment_session_indices])
    
    print("Enrollment Data Size", sum(len(sequence) for user in valid_users_enrollment_data for session in user for sequence in session ))
    return valid_users_enrollment_data, valid_users_verify_data, unauthorized_users_data

def finetune(model, model_config, valid_users_enrollment_data, batch_size, device):
    
    print("---------- FINETUNING ----------------")

    # Fine tuning the model on the enrollment data
    same_user_ratio_in_batch = 0.25 # Ratio of same user pairs in each batch

    # # Data Loader
    dataloader = get_training_dataloader(training_data=valid_users_enrollment_data, batch_size=batch_size, same_user_ratio=same_user_ratio_in_batch, sequence_length=model_config.seq_len, 
                                         required_feature_dim=model_config.d_model, num_workers=1)
    
    # # Enabling Tensor Flow 32 (TF32) to make calculations faster 
    torch.set_float32_matmul_precision('high')

    n_epochs = 10
    steps_per_epoch = len(dataloader) # Number of steps in an epoch OR number of batches
    total_steps_to_train = steps_per_epoch * n_epochs # Total number of steps to train the model

    print("steps_per_epoch", steps_per_epoch, "total_steps_to_train", total_steps_to_train)

    max_lr = 6e-4
    min_lr = max_lr * 0.1
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

    step_count = 0
    for epoch in range(n_epochs):
        for batch in dataloader:
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

            # Forward Pass
            emb, logits, loss = model(inputs=sequences, temporal_attn_mask=temporal_attention_mask, channel_attn_mask=channel_attention_mask, targets=labels)

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

            step_count += 1

    return dataloader


@torch.no_grad()
def prepare_ood(model: Model, dataloader, device):

    # To store the enrolled user embeddings and labels
    bank = None # Size (N: # of sequences, d_model: Embedding dimension) 
    label_bank = None # Size (N)
    # Putting the model in evaluation mode
    model.eval()

    # Going through the enrolled user's sequences
    for batch in dataloader:

        # Sequences, labels and masks
        sequences = batch['sequences'] # (batch_size (B), sequence_length (T), embedding size (C))
        labels = batch['user_ids'] # User IDs (batch_size (B))
        temporal_attention_mask = batch['temporal_attention_mask'] # (B, T)
        channel_attention_mask = batch['channel_attention_mask'] # (B, C)

        # Moving the tensors to device
        sequences = sequences.to(device)
        labels = labels.to(device)
        temporal_attention_mask = temporal_attention_mask.to(device)
        channel_attention_mask = channel_attention_mask.to(device)

        # Passing through the model to get the embeddings and the logits
        emb, logits, _ = model(inputs=sequences, temporal_attn_mask= temporal_attention_mask, channel_attn_mask=channel_attention_mask)

        # Storing the embeddings in bank and labels in label_bank
        if bank is None:
            bank = emb.clone().detach()
            label_bank = labels.clone().detach()
        else:
            temp_bank = emb.clone().detach()
            temp_label_bank = labels.clone().detach()
            bank = torch.cat([temp_bank, bank], dim=0) 
            label_bank = torch.cat([temp_label_bank, label_bank], dim=0)
    
    # Normalizing all the embeddings
    norm_bank = F.normalize(bank, dim=-1) # (N: num.of samples, d_output_emb)
    N, d = bank.size()
    all_classes = list(set(label_bank.tolist())) # List of all the class names / users

    # calculating the mean embeddings of each class : For Mahalonobis Distance 
    class_mean = torch.zeros(max(all_classes) + 1, d).to(device) # (C: num. of classes, d_output_emb: embedding dimension)
    for c in all_classes:
        class_mean[c] = (bank[label_bank == c].mean(0))

    # Centering all the embeddings
    centered_bank = (bank - class_mean[label_bank]).detach().cpu().numpy()

    # Calculating the Covariance matrix over all the classes (For Mahalonobis distance)
    precision = LedoitWolf().fit(centered_bank).precision_.astype(np.float32)
    class_var = torch.from_numpy(precision).float().to(device=device)

    model.train()
    return class_mean, class_var, norm_bank, all_classes


@torch.no_grad()
def compute_ood_scores(model: Model, dataloader, all_classes, class_mean, class_var, norm_bank, device):
    # Putting the model in eval model
    model.eval()

    # To store the scores of each batch
    scores = []

    # To store the true labels and predictions
    true_labels = []
    preds = []

    # For each batch in the dataloader
    for batch in dataloader:
        sequences = batch['sequences'] # (batch_size (B), sequence_length (T), embedding size (C))
        labels = batch['user_ids'] # User IDs (batch_size (B,))
        temporal_attention_mask = batch['temporal_attention_mask'] # (B, T)
        channel_attention_mask = batch['channel_attention_mask'] # (B, C)

        # Moving the tensors to device
        sequences = sequences.to(device)
        labels = labels.to(device)
        temporal_attention_mask = temporal_attention_mask.to(device)
        channel_attention_mask = channel_attention_mask.to(device)

        # Passing through the model to get the embeddings and the logits
        emb, logits, _ = model(inputs=sequences, temporal_attn_mask= temporal_attention_mask, channel_attn_mask=channel_attention_mask)

        # Probabilities
        probs = F.softmax(logits, dim=-1) # (B, n_classes)

        softmax_max = probs.max(-1) # Maximum value in softmax: Returns a tuple (max_value, max_index)

        # Softmax Score
        softmax_score = softmax_max[0] # (B,) # Low: OOD, High: ID

        batch_preds = softmax_max[1] # (B,) predictions
        true_labels = true_labels + labels.tolist() # Appending to true labels list
        preds = preds + batch_preds.tolist() # Appending to preductions list

        # Mahalonobis score
        maha_score = []
        for c in all_classes:
            centered_emb = emb - class_mean[c].unsqueeze(0) # (B, d_output_emb) - (1, d_output_emb) = (B, d_output_emb)
            ms = torch.diag(centered_emb @ class_var @ centered_emb.t()) # (B, d_output_emb) @ (d_output_emb, d_output_emb) @ (d_output_emb, B) = (B, B) -> B(torch.diag)
            maha_score.append(ms)
        maha_score = torch.stack(maha_score, dim=-1) # Combining the different tensors in the list (B, num_classes)
        maha_score = maha_score.min(-1)[0] # Minimum score for each sample (B,)
        maha_score = - maha_score # To make lower scores -> OOD and higher scores as In distribution

        # Cosine score  
        norm_emb = F.normalize(emb, dim=-1) # Normalizing the embeddings (B, d_output_emb)
        cosine_score = norm_emb @ norm_bank.t() # (B, d_output_emb) @ (d_output_emb, N: num.of samples in enrollment) = (B, N)
        cosine_score = cosine_score.max(-1)[0] # Maximum similarity: closest match between the test input and any ID sample, Low: OOD, High: ID: (B,)

        # Energy score
        energy_score = torch.logsumexp(logits, dim=-1) # Low: OOD, High: ID # (B,)

        # Appending each batches score to the list of scores
        scores.append({
            'softmax': softmax_score.tolist(),
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'energy': energy_score.tolist()
        })

    scores_dict = {} # Dictionary with keys equal to the scores and value as an array representing each samples score
    for key in list(scores[0].keys()): # For all the keys or scores
        scores_dict[key] = []  # Initializing the list 
        for batch_score in scores: # For each batches score
            scores_dict[key] += batch_score[key] # Appending the score to the list

    model.train() # Putting the model back in training mode
    return scores_dict, true_labels, preds



def evaluate(model, dataset, num_of_enrollment_sessions, device):
    
    print("--------------------- VALIDATION -----------------------")
    num_of_valid_users = 3

    # Validation Model Copy
    validation_model = prepare_validation_model(model=model, num_of_valid_users=num_of_valid_users, device=device)
    
    # Models config
    model_config: ModelConfig = validation_model.config

    # Dataset for finetuning and validation
    valid_users_enrollment_data, valid_users_verify_data, unauthorized_users_data = prepare_dataset(dataset=dataset, num_of_valid_users=num_of_valid_users, num_of_enrollment_sessions=num_of_enrollment_sessions)

    # Finetuning the model on the enrollment data
    finetune_dataloader = finetune(model=validation_model, model_config=model_config, valid_users_enrollment_data=valid_users_enrollment_data, batch_size=64, device=device)

    # Calculating the class means, variance and normalized embeddings from the enrollment data
    class_mean, class_var, norm_bank, all_classes = prepare_ood(model=validation_model, dataloader=finetune_dataloader, device=device)

    # Data loaders for valid user verification sequences, and unauthorized users
    valid_user_verification_data_loader = get_testing_dataloader(valid_users_verify_data, batch_size=64, sequence_length=model_config.seq_len, required_feature_dim=model_config.d_model, num_workers=1)
    unauthorized_users_data_loader = get_testing_dataloader(unauthorized_users_data, batch_size=64, sequence_length=model_config.seq_len, required_feature_dim=model_config.d_model, num_workers=1)
    
    # Computing scores for in distribution data(Valid users verification session)
    in_distribution_scores, true_labels, preds = compute_ood_scores(model=validation_model, dataloader=valid_user_verification_data_loader, 
                                                all_classes=all_classes, class_mean=class_mean, class_var=class_var, norm_bank=norm_bank, device=device)
    
    # Computing scores for OOD data (Unauthorized users)
    ood_scores, _ , _ = compute_ood_scores(model=validation_model, dataloader=unauthorized_users_data_loader, 
                                                all_classes=all_classes, class_mean=class_mean, class_var=class_var, norm_bank=norm_bank, device=device)
    
    # Classification accuracy
    classification_accuracy = sum(t == p for t, p in zip(true_labels, preds)) / len(true_labels)

    key_wise_stats = {}

    best_eer = math.inf
    for key in list(in_distribution_scores.keys()):

        # In distribution data scores and OOD data scores
        in_distribution_for_key = np.array(in_distribution_scores[key], dtype=np.float64)
        ood_for_key = np.array(ood_scores[key], dtype=np.float64)
        
        # Labels: 1: In distribution, 0: OOD
        in_distribution_actual = np.ones_like(in_distribution_for_key).astype(np.int64)
        ood_actual = np.zeros_like(ood_for_key).astype(np.int64)

        # Concatenating the In distribution and OOD scores and labels.
        scores_for_key = np.concatenate([in_distribution_for_key, ood_for_key], axis=0)
        labels = np.concatenate([in_distribution_actual, ood_actual], axis=0)
        
        # TPR: how many are correctly predicted in distribution out of total in distribution
        # FPR: how many are incorrectly predicted in distribution out of total OOD - Equal to "FAR"
        fpr, tpr, thresholds = roc_curve(labels, scores_for_key)

        # Area under the ROC curve
        auc = roc_auc_score(labels, scores_for_key)

        frr = 1 - tpr # FRR = 1 - TPR, Since: FRR = FN/(FN+TP) AND TPR = TP/(TP + FN) and TP + FN = num. of positive classes

        # Subtracting fpr with frr
        abs_diff = np.abs(fpr-frr)
        # Point where fpr and frr are closest to each other (or equal) => EER
        idx_eer = np.argmin(abs_diff)

        # EER
        eer = (fpr[idx_eer] + frr[idx_eer]) / 2.0 
        eer_threshold = thresholds[idx_eer] # Threshold where FPR = FRR

        # Best EER out of all scoring methods
        best_eer = min(eer, best_eer)

        key_wise_stats[key] = {'frr_at_threshold': frr[idx_eer], 'far_at_threshold': fpr[idx_eer], 'eer': eer, 'threshold': eer_threshold,  'auc': auc}


    # Returning classification accuracy and scores of each
    return classification_accuracy, key_wise_stats, best_eer






