import numpy as np
import sys
import pickle
import torch
import numpy as np

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
        d_output_emb= 64 # Output embedding dimension
    )
    screen_dim_x=1903
    screen_dim_y=1920
    batch_size = 64
    same_user_ratio_in_batch = 0.25
    sequence_length = 10

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

    # Model
    model = Model(model_config)
    # Moving the model to the device used for training
    # model.to(device)

    # print("Model")

    # for k, v in model.state_dict().items():
    #     print(k, v.shape)

    
    # dataloader = get_training_dataloader(training_data=train_dataset, batch_size=batch_size, same_user_ratio=same_user_ratio_in_batch, sequence_length=sequence_length, 
    #                                      required_feature_dim=model_config.d_model, num_workers=1)

        
    


