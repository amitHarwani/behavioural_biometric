import torch
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler
from numpy.typing import NDArray
from functools import partial

class TrainingDataset(Dataset):
    
    def __init__(self, data: list[list[list[NDArray]]]):
        """
        Args:
            data: [users][sessions][sequences] hierarchical data
        """
        self.sequences = []
        self.user_ids = []
        self.user_to_indices = defaultdict(list) # Indices of sequences belonging to the user.
        
        # Flatten the data while tracking user IDs
        idx = 0 # Global index of sequences across all users and sessions
        for user_idx, user_data in enumerate(data):
            for session in user_data:
                for sequence in session:
                    self.sequences.append(torch.tensor(sequence)) # Appending the sequence(10,62) ndarray to sequences list
                    self.user_ids.append(user_idx) # Appending the user id of the sequence to user_ids
                    self.user_to_indices[user_idx].append(idx) # Appending to user_to_indices 
                    idx += 1
    
    def __len__(self):
        return len(self.sequences) # Size of dataset = num. of sequences
    
    def __getitem__(self, idx):
        # When getting a item return the sequence as well as the user whom it belongs to
        return {
            'sequence': self.sequences[idx],
            'user_id': self.user_ids[idx]
        }
    

class TestingDataset(Dataset):
    
    def __init__(self, data: list[list[list[NDArray]]]):
        """
        Args:
            data: [users][sessions][sequences] hierarchical data
        """
        self.sequences = []
        self.user_ids = []
        
        # Flatten the data while tracking user IDs
        idx = 0 # Global index of sequences across all users and sessions
        for user_idx, user_data in enumerate(data):
            for session in user_data:
                for sequence in session:
                    self.sequences.append(torch.tensor(sequence)) # Appending the sequence(10,62) ndarray to sequences list
                    self.user_ids.append(user_idx) # Appending the user id of the sequence to user_ids
                    idx += 1
    
    def __len__(self):
        return len(self.sequences) # Size of dataset = num. of sequences
    
    def __getitem__(self, idx):
        # When getting a item return the sequence as well as the user whom it belongs to
        return {
            'sequence': self.sequences[idx],
            'user_id': self.user_ids[idx]
        }

class ContrastiveSampler(Sampler):
    """
    Batch sampler that ensures:
    1. Each batch has multiple users
    2. Each batch has some same-user pairs
    3. All sequences are used in an epoch
    """
    
    def __init__(self, dataset: TrainingDataset, batch_size: int, 
                same_user_ratio: float = 0.25):
        """
        Args:
            dataset: The dataset to sample from
            batch_size: Size of each batch
            same_user_ratio: Target ratio of same-user pairs in each batch
        """
        self.dataset = dataset                          # Dataset
        self.batch_size = batch_size                    # Batch Size
        self.same_user_ratio = same_user_ratio          # Same User Raio
        self.user_to_indices = dataset.user_to_indices  # User to Indices from the dataset
        self.users = list(self.user_to_indices.keys())  # Keys of user_to_indices = list of users
        
    def __iter__(self):
        # Create batches for the entire dataset
        all_indices = list(range(len(self.dataset))) # All indices corresponding to the flat sequences in the dataset
        random.shuffle(all_indices) # Shuffling the indices
        
        # Create a working copy of user_to_indices
        available_indices = {user: set(indices.copy()) for user, indices in self.user_to_indices.items()}
        
        batches = [] # To store all the batches
        remaining = set(all_indices) # Indices remaining/those which can be selected
        
        # If indices are remaining
        while remaining:
            current_batch = []

            # Batch size = defined batch size or the length of remaining indices
            batch_size = min(self.batch_size, len(remaining))
            
            # First, add some same-user pairs
            # Pairs needed is either 1, or equivalent to the same_user_ratio -> Number of users whose same pairs must be added
            pairs_needed = max(1, int(batch_size * self.same_user_ratio) // 2)
            pairs_added = 0 # Pairs added
            
            # Find users with at least 2 remaining sequences, from available_indices dictionary
            eligible_users = [u for u in self.users if len(available_indices[u]) >= 2]
            random.shuffle(eligible_users) # Shuffle the eligible users
            
            for user in eligible_users:
                # If all pairs have been added or the batch is full, break
                if pairs_added >= pairs_needed or len(current_batch) >= batch_size - 1:
                    break
                
                # Sequence indices for the current user
                user_seqs = list(available_indices[user])

                # If 2 sequences are available - Double Check
                if len(user_seqs) >= 2:
                    # Add a pair of sequences from this user
                    pair = random.sample(user_seqs, 2) # Randomly sample 2 sequence indices
                    current_batch.extend(pair)
                    
                    # Update remaining sequences
                    for idx in pair:
                        available_indices[user].remove(idx) # Removing sequence indices from  available_indices
                        remaining.remove(idx) # Remove sequence indices from remaining
                        
                    pairs_added += 1 # Increment pair added
            
            # Fill the rest of the batch with random sequences
            needed = batch_size - len(current_batch) # Remaining sequence indices needed

            # If sequence indices are needed
            if needed > 0:
                # Copy remaining into random_indices list
                random_indices = list(remaining)
                random.shuffle(random_indices) # Shuffle the remaining indices
                
                # Prefer indices from users not already in batch if possible
                users_in_batch = set(self.dataset.user_ids[idx] for idx in current_batch) # Users already in the batch: Getting from user_ids in dataset
                diverse_indices = [idx for idx in random_indices 
                                  if self.dataset.user_ids[idx] not in users_in_batch] # Sequence indices which belong to users other than those in the batch
                
                # If diverse_indices can complete the batch, take all of the diverse_indices, else take everything from random_indices
                to_add = diverse_indices[:needed] if len(diverse_indices) >= needed else random_indices[:needed]
                current_batch.extend(to_add)
                
                # For indices added
                for idx in to_add:
                    user = self.dataset.user_ids[idx] # User Id of the sequence index
                    available_indices[user].remove(idx) # Remove from available indices dictionary
                    remaining.remove(idx) # Remove from remaining set
            
            # Add the batch
            random.shuffle(current_batch)  # Shuffle within batch
            batches.append(current_batch)
        
        random.shuffle(batches)  # Shuffle batch order
        return iter(batches)
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size # Equivalent to math.ceil(len(self.dataset)/ self.batch_size)


def collate_fn(batch, max_sequence_len, required_feature_dim=64):
    """Collate function to handle variable-length sequences."""
    sequences = [item['sequence'] for item in batch] # All the sequences in the batch
    user_ids = [item['user_id'] for item in batch] # User Ids belonging to the sequences in the batch
    
    # Get sequence lengths
    lengths = torch.tensor([seq.size(0) for seq in sequences]) # length for all the sequences in the batch.
    max_len = max_sequence_len #lengths.max().item() # Max length among all
    
    # Get feature dimension
    feature_dim = required_feature_dim # Required Feature dimension
    actual_feature_dim = sequences[0].size(1) 

    # Channel mask for channel head attention, Initializing with all trues i.e. Ignore all
    channel_mask = torch.ones(len(sequences), feature_dim, dtype=torch.bool); # (batch_size, feature_dim)
    channel_mask[:, :actual_feature_dim] = False # Setting all the actual features to False (Don't Ignore them)

    # Create padded tensor and mask
    padded = torch.zeros(len(sequences), max_len, feature_dim) # (batch_size, max_len of sequences, feature_dim)
    temporal_mask = torch.ones(len(sequences), max_len, dtype=torch.bool) # (batch_size, max_len of sequences)
    
    # Fill padded tensor
    for i, seq in enumerate(sequences): # Looping through the sequences in the batch
        end = lengths[i] # Current length of the sequence
        padded[i, :end, :actual_feature_dim] = seq # At the batch item (i), till the current sequence length: fill the current seq
        temporal_mask[i, :end] = False # At the batch item (i), the the current sequence length: fill false (0's) to indicate no mask.
    
    return {
        'sequences': padded, # (batch_size, max_len of sequences, feature_dim)
        'user_ids': torch.tensor(user_ids), # user_ids
        'temporal_attention_mask': temporal_mask, # Temporal Attention Mask,
        'channel_attention_mask': channel_mask,
        'lengths': lengths # Actual lengths if needed
    }


def get_training_dataloader(training_data, batch_size=64, same_user_ratio = 0.25, sequence_length=10, required_feature_dim = 64, num_workers=4) -> DataLoader:
    dataset = TrainingDataset(training_data)
    sampler = ContrastiveSampler(dataset, batch_size, same_user_ratio)
    
    # Using partial to fix the max_sequence_length
    collate_fn_initialized = partial(collate_fn, max_sequence_len=sequence_length, required_feature_dim = required_feature_dim)

    # Initializing and returning the data loader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_initialized
    )
    
    return dataloader

def get_testing_dataloader(test_data, batch_size=64, sequence_length=10, required_feature_dim=64, num_workers=4) -> DataLoader:
    # Testing dataset
    dataset = TestingDataset(test_data)

    # Using partial to fix the max_sequence_length
    collate_fn_initialized = partial(collate_fn, max_sequence_len=sequence_length, required_feature_dim = required_feature_dim)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn_initialized,
        shuffle=True
    )

    return dataloader