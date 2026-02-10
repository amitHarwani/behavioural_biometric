import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import numpy as np
from collections import defaultdict

from train_3 import merge_sequences_overlap
from model_basic import Model, ModelConfig
from validation import validate, validate_multi


def normalize_and_init_dataset(data, screen_dim_x, screen_dim_y, split="train"):

    idx = 0 # Global index of sequences across all users and sessions
    sequences = []
    user_ids = []
    user_to_indices = defaultdict(list) # Indices of sequences belonging to the user.

    for u_idx, user in enumerate(data):
        for s_idx, session in enumerate(user):
            for seq_idx, sequence in enumerate(session if split == "train" else session):
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

if __name__ == "__main__":


    version = "v1"
    test_dataset_file = f"{version}_test_users_data_tw10ms.pickle"
    test_dataset_merged_file = f"{version}_merged_test_users_data_tw10ms.pickle"
    cp_files = ["./exp3/train_3_1_epoch_49.pt"]

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

        # multi_results = validate_multi(model, test_sequences, test_user_ids, test_user_to_indices, device=device, group_sizes=(2, 3, 4, 5, 6, 7, 8, 9))
        # print(multi_results)
        # import pickle
        # with open(f"./exp3/res/multi_user_results.pickle",'wb') as outfile:
        #     pickle.dump(multi_results, outfile)

        avg_cosine_eer, avg_maha_eer, avg_cosine_auc, avg_maha_auc = validate(model, test_sequences, test_user_ids, test_user_to_indices, device=device, all_imp=True, plot=False, get_cosine_score=False)

        print(f"CP: {cp_file} | Test Results: avg_cosine_eer: {avg_cosine_eer:.4f} | avg_maha_eer: {avg_maha_eer:.4f} | avg_cosine_auc: {avg_cosine_auc} | avg_maha_auc: {avg_maha_auc}")
        print("**********************************************************************************************************")





