import numpy as np
import sys
import pickle
import torch
import numpy as np
import math
import inspect
import matplotlib.pyplot as plt

from model import ModelConfig, Model
from data_loader import get_training_dataloader

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
                    all_data.append(sequence) # Sequence is mostly (10, 62)

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



if __name__ == "__main__":
    # Parameters
    model_config = ModelConfig(
        k = 6, # Number of gaussians in GRE
        d_model= 64, # Num. of features
        seq_len= 10, # Block size/seq. length
        n_temporal_heads= 4, # Num. of temporal heads
        n_channel_heads= 5, # Num. of channel heads
        dropout= 0.1, # Dropout probability 
        n_layers= 5, # Number of layers or transformer encoders
        d_output_emb= 64, # Output embedding dimension
        n_users = 69, # Number of users (For classification)
        contrastive_loss_alpha = 2 # Contrastive loss importance hyperparameter (Alpha)
    )
    screen_dim_x=1903 # Screen width (For touch data)
    screen_dim_y=1920 # Screen height (For touch data)
    batch_size = 64 # Batch size
    same_user_ratio_in_batch = 0.25 # Ratio of same user pair sequences in the batch
    sequence_length = 10 # Sequence length
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

    # Data Loader
    dataloader = get_training_dataloader(training_data=train_dataset, batch_size=batch_size, same_user_ratio=same_user_ratio_in_batch, sequence_length=sequence_length, 
                                         required_feature_dim=model_config.d_model, num_workers=1)
    
    # Enabling Tensor Flow 32 (TF32) to make calculations faster 
    torch.set_float32_matmul_precision('high')

    # Model
    model = Model(model_config)
    num_of_parameters = sum(p.numel() for p in model.parameters())

    # Moving the model to the device used for training
    model.to(device)

    # print("Model")
    # for k, v in model.state_dict().items():
    #     print(k, v.shape)


    steps_per_epoch = len(dataloader) # Number of steps in an epoch OR number of batches
    total_steps_to_train = steps_per_epoch * n_epochs # Total number of steps to train the model

    max_lr = 6e-4 # 0.0006
    min_lr = max_lr * 0.1 # 0.00006
    warmup_steps = int(total_steps_to_train * 0.1) # 10% of total steps

    # Cosine decay with linear warmup learning rate schedule
    def get_lr(iteration):
        # Linear warmup
        if iteration < warmup_steps:
            return max_lr * (iteration + 1 / warmup_steps)
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

    print("Total Steps", total_steps_to_train)
    step_count = 0
    for epoch in range(n_epochs):
        for batch in dataloader:
            print("--Step--", step_count)

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

             # Using BF16
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # Forward Pass
                emb, logits, loss = model(inputs=sequences, temporal_attn_mask= temporal_attention_mask, channel_attn_mask=channel_attention_mask, targets=labels)

            # Backprop
            loss.backward()

            raw_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2.0)
            print(f"Raw Gradient Norm: {raw_norm}")

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

            # print("Logits")
            # print(logits)
            # plt.figure(figsize=(20, 4))
            # legends = []
            # for name, param in model.named_parameters():
            #         if param.grad is not None and param.grad.numel() >= 2:
            #             grad = param.grad.cpu()
            #             grad_norm = torch.norm(grad).item()
            #             grad_mean = grad.mean().item()
            #             grad_var = grad.std().item()
            #             print(f"{name} | Grad Norm: {grad_norm:.4f} | Grad Mean: {grad_mean:.6f} | Grad var: {grad_var:.6f}")
            #             hy, hx = torch.histogram(grad, density=True)
            #             plt.plot(hx[:-1].detach(), hy.detach())
            #             legends.append(f'layer {name}')

            # plt.legend(legends)
            # plt.title('gradient distribution')
            # plt.show()

            step_count += 1
        

            

    


        
    


