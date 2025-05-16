import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import numpy as np
from train_1 import merge_sequences_overlap, normalize_and_init_dataset
from model_basic import Model, ModelConfig
from validation import validate



if __name__ == "__main__":


    version = "v1"
    test_dataset_file = f"{version}_validation_users_data_tw10ms.pickle"
    test_dataset_merged_file = f"{version}_merged_validation_users_data_tw10ms.pickle"
    cp_files = ["./checkpoints/train_1_1.pt"]

      # If the preprocessed files have been merged
    if os.path.exists(test_dataset_merged_file) :
        # Loading the preprocessed merged objects
        with open(test_dataset_merged_file, "rb") as infile:
            test_dataset = pickle.load(infile)
    else:
        # Loading the preprocessed objects
        with open(test_dataset_file, "rb") as infile:
            test_dataset = pickle.load(infile)

        # Merging the sequences
        test_dataset = merge_sequences_overlap(test_dataset, merge_length=200, overlap_length=100)

        # Inserting into the file
        with open(test_dataset_merged_file, "wb") as outfile:
            pickle.dump(test_dataset, outfile)


    test_sequences, test_user_ids, test_user_to_indices = normalize_and_init_dataset(test_dataset, screen_dim_x=0, screen_dim_y=0, split="test")

    # Identifying Device
    device = "cpu"
    if torch.cuda.is_available(): # GPU
        device="cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # Apple Silicon
        device="mps"

    for cp_file in cp_files:
        cp = torch.load(cp_file, weights_only=False)
        model_config = cp['config']
        model = Model(model_config)
        model.load_state_dict(cp['model'])
        model.to(device)

        avg_cosine_eer, avg_maha_eer = validate(model, test_sequences, test_user_ids, test_user_to_indices, device=device)

        print(f"CP: {cp_file} | Test Results: avg_cosine_eer: {avg_cosine_eer:.4f} | avg_maha_eer: {avg_maha_eer:.4f}")





