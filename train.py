import numpy as np
import sys
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from numpy.typing import NDArray
import pickle

from model import ModelConfig

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional


def normalize(data, screen_dim_x, screen_dim_y, split: str = "train"):

    # If mean and std. are not computed yet and the split is not train
    if (means_for_normalization == None or stds_for_normalization == None) and split != "train":
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


if __name__ == "main":
    # Parameters
    model_config = ModelConfig(
        k = 6, # Number of gaussians in GRE
        d_model= 62, # Num. of features
        seq_len= 10, # Block size/seq. length
        n_temporal_heads= 5, # Num. of temporal heads
        n_channel_heads= 5, # Num. of channel heads
        dropout= 0.1, # Dropout probability 
        n_layers= 5, # Number of layers or transformer encoders
        d_output_emb= 64 # Output embedding dimension
    )

    # Preprocessed files
    train_dataset_file = "training_users_data_tw1_sq10_maxk8.pickle"
    val_dataset_file = "validation_users_data_tw1_sq10_maxk8.pickle"
    test_dataset_file = "test_users_data_tw1_sq10_maxk8.pickle"

    # Loading the preprocessed objects
    with open(train_dataset_file, "rb") as infile:
        train_dataset = pickle.loads(infile)

    with open(val_dataset_file, "rb") as infile:
        val_dataset = pickle.loads(infile)

    with open(test_dataset_file, "rb") as infile:
        test_dataset = pickle.loads(infile)

    # Means and std. deviations for normalization
    means_for_normalization = None
    stds_for_normalization = None

    # Normalizing the datasets
    normalize(train_dataset, screen_dim_x=1903, screen_dim_y=1920, split="train")
    normalize(val_dataset, screen_dim_x=1903, screen_dim_y=1920, split="val")
    normalize(test_dataset, screen_dim_x=1903, screen_dim_y=1920, split="test")

    
    

        
    


