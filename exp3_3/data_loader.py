import torch
import random
import math
import numpy as np
from collections import defaultdict, deque
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from numpy.typing import NDArray
from functools import partial

class TrainingDataset(Dataset):
    
    def __init__(self, train_sequences: list, train_user_ids: list, train_user_to_indices: dict):

        self.sequences = train_sequences
        self.user_ids = train_user_ids
        self.user_to_indices = train_user_to_indices # Indices of sequences belonging to the user.
    
    def __len__(self):
        return len(self.sequences) # Size of dataset = num. of sequences
    
    def __getitem__(self, idx):
        # When getting a item return the sequence as well as the user whom it belongs to
        return {
            'sequence': torch.tensor(self.sequences[idx]),
            'user_id': self.user_ids[idx]
        }
    
class ValidationDataset(Dataset):
    
    def __init__(self, val_sequences: list, val_user_ids: list):

        self.sequences = val_sequences
        self.user_ids = val_user_ids
        
    def __len__(self):
        return len(self.sequences) # Size of dataset = num. of sequences
    
    def __getitem__(self, idx):
        # When getting a item return the sequence as well as the user whom it belongs to
        return {
            'sequence': torch.tensor(self.sequences[idx]),
            'user_id': self.user_ids[idx]
        }
    
class TestingDataset(Dataset):
    
    def __init__(self,  test_sequences: list, test_user_ids: list):
        """
        Args:
            data: [users][sessions][sequences] hierarchical data
        """
        self.sequences = test_sequences
        self.user_ids = test_user_ids
    
    def __len__(self):
        return len(self.sequences) # Size of dataset = num. of sequences
    
    def __getitem__(self, idx):
        # When getting a item return the sequence as well as the user whom it belongs to
        return {
            'sequence': torch.tensor(self.sequences[idx]),
            'user_id': self.user_ids[idx]
        }

class ContrastiveSampler(BatchSampler):
    """
    Produces batches of indices so that:
      - Every sequence index appears exactly once per epoch.
      - Within each batch, any class that appears, appears at least twice (i.e. in positive pairs).
    """
    def __init__(self, dataset:TrainingDataset, batch_size):
        """
        labels: list of class‐IDs for each sample in the dataset
        batch_size: must be even (we form pairs)
        """
        assert batch_size % 2 == 0, "batch_size must be even"
        self.dataset = dataset
        self.user_to_indices = dataset.user_to_indices
        self.batch_size = batch_size

        # Precompute total indices emitted per epoch (incl. duplicates)
        total = 0
        for idxs in self.user_to_indices.values():
            total += len(idxs) + (len(idxs) % 2)
        self.total_indices = total

    def _prepare_epoch(self):
        # Build list of (i,j) pairs for all classes
        pairs = []
        for user, idxs in self.user_to_indices.items():
            idxs_copy = np.array(idxs, dtype=int)
            np.random.shuffle(idxs_copy)
            # if odd count, duplicate one index to make a pair
            if len(idxs_copy) % 2 == 1:
                dup = np.random.choice(idxs_copy)
                idxs_copy = np.concatenate([idxs_copy, [dup]])
            # chunk into pairs
            for k in range(0, len(idxs_copy), 2):
                pairs.append((idxs_copy[k], idxs_copy[k+1]))
        random.shuffle(pairs)

        # flatten to a single list of indices
        self.epoch_indices = [i for pair in pairs for i in pair]

    def __iter__(self):
        # at each epoch, re-build & shuffle
        self._prepare_epoch()
        # yield in batch_size chunks
        for start in range(0, len(self.epoch_indices), self.batch_size):
            yield self.epoch_indices[start:start + self.batch_size]

    def __len__(self):
        # number of batches per epoch
        return math.ceil(self.total_indices / self.batch_size)


class PairLevelNumpyContrastiveSampler(BatchSampler):
    """
    A BatchSampler that yields batches where any class present appears at least twice (positive pairs),
    using a memory-efficient NumPy-based implementation.

    - Builds per-user paired indices in a 2D NumPy array of shape (num_pairs, 2)
    - Shuffles the pairs (rows) globally
    - Yields each batch by flattening batch_size/2 pairs => batch_size samples
    """
    def __init__(self, dataset: TrainingDataset, batch_size, drop_last=True):
        assert batch_size % 2 == 0, "batch_size must be even"
        self.user_to_indices = dataset.user_to_indices
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Compute total indices (with padding for odd counts)
        total = 0
        for idxs in self.user_to_indices.values():
            total += len(idxs) + (len(idxs) % 2)
        self.total_indices = total

    def _prepare_epoch(self):
        # Create a list to hold all pair arrays
        pair_list = []
        for idxs in self.user_to_indices.values():
            arr = np.array(idxs, dtype=np.int64)
            # If odd, duplicate one random sample to pad
            if arr.size % 2 == 1:
                dup = np.random.choice(arr)
                arr = np.concatenate([arr, [dup]])
            # Shuffle within-user
            np.random.shuffle(arr)
            # Reshape to (num_pairs, 2)
            pairs = arr.reshape(-1, 2)
            pair_list.append(pairs)

        # Concatenate all user pairs into one big 2D array
        self.all_pairs = np.vstack(pair_list)
        # Global shuffle of pairs (rows)
        np.random.shuffle(self.all_pairs)
        # Compute number of batches
        self.num_pairs_per_batch = self.batch_size // 2
        self.num_batches = math.floor(
            self.all_pairs.shape[0] / self.num_pairs_per_batch
        )

    def __iter__(self):
        # Prepare and shuffle epoch pairs
        self._prepare_epoch()

        # Yield each batch
        for b in range(self.num_batches):
            start = b * self.num_pairs_per_batch
            end = start + self.num_pairs_per_batch
            batch_pairs = self.all_pairs[start:end]
            # Flatten to 1D list of indices
            batch = batch_pairs.flatten().tolist()
            yield batch

        # Optionally handle leftover pairs
        if not self.drop_last:
            leftover_start = self.num_batches * self.num_pairs_per_batch
            if leftover_start < self.all_pairs.shape[0]:
                leftover = self.all_pairs[leftover_start:]
                batch = leftover.flatten().tolist()
                yield batch

    def __len__(self):
        return math.ceil(self.total_indices / self.batch_size)

class ClassBalancedBatchSampler(BatchSampler):

    def __init__(self,
                 dataset: TrainingDataset,
                 n_classes: int,
                 samples_per_class: int):
        
        self.user_to_indices = dataset.user_to_indices
        self.users = list(self.user_to_indices.keys())
        self.n_classes = n_classes
        self.samples_per_class = samples_per_class
        self.batch_size = n_classes * samples_per_class

    def __iter__(self):
        # Compute how many batches per epoch
        # We’ll define an “epoch” as consuming each user approximately equally:
        # total_samples = sum(len(idxs) for idxs in self.user_to_indices.values())
        # num_batches = total_samples // batch_size
        total_samples = sum(len(idxs) for idxs in self.user_to_indices.values())
        num_batches = total_samples // self.batch_size

        for _ in range(num_batches):
            # 1) sample C distinct users
            selected_users = random.sample(self.users, self.n_classes)

            batch = []
            # 2) for each user, sample K indices
            for u in selected_users:
                idxs = self.user_to_indices[u] # Users' sequence indices
                if len(idxs) >= self.samples_per_class:
                    chosen = random.sample(idxs, self.samples_per_class)
                else:
                    # with-replacement if user has fewer than K samples
                    chosen = random.choices(idxs, k=self.samples_per_class)
                batch.extend(chosen)

            yield batch

    def __len__(self):
        total_samples = sum(len(idxs) for idxs in self.user_to_indices.values())
        return total_samples // self.batch_size


def collate_fn(batch, max_sequence_len):
    """Collate function to handle variable-length sequences."""
    sequences = [item['sequence'] for item in batch] # All the sequences in the batch
    user_ids = [item['user_id'] for item in batch] # User Ids belonging to the sequences in the batch
    
    padded = torch.stack(sequences)
    
    # Modality mask for modality-specific attention
    modality_mask = torch.zeros(len(sequences), max_sequence_len, 2, dtype=torch.int)  # (batch_size, max_len, 2 modalities)

    # First modality: Check if the first feature is non-zero
    modality_mask[:, :, 0] = (padded[:, :, 0] != 0).int()

    # Second modality: Always true
    modality_mask[:, :, 1] = 1

    return {
        'sequences': padded, # (batch_size, max_len of sequences, feature_dim)
        'user_ids': torch.tensor(user_ids), # user_ids
        'modality_mask': modality_mask,
    }


def get_training_dataloader(train_sequences, train_user_ids, train_user_to_indices, n_classes_per_batch, n_samples_per_class, sequence_length=100, num_workers=4) -> DataLoader:
    dataset = TrainingDataset(train_sequences, train_user_ids, train_user_to_indices,)
    sampler = ClassBalancedBatchSampler(dataset=dataset, n_classes=n_classes_per_batch, samples_per_class=n_samples_per_class)
    
    # Using partial to fix the max_sequence_length
    collate_fn_initialized = partial(collate_fn, max_sequence_len=sequence_length)

    # Initializing and returning the data loader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_initialized
    )
    
    return dataloader

def get_validation_dataloader(val_sequences, val_user_ids, batch_size=64, sequence_length=100, num_workers=4) -> DataLoader:
    # Validation dataset
    dataset = ValidationDataset(val_sequences, val_user_ids)

    # Using partial to fix the max_sequence_length
    collate_fn_initialized = partial(collate_fn, max_sequence_len=sequence_length)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn_initialized,
        shuffle=True
    )

    return dataloader

def get_testing_dataloader(test_sequences, test_user_ids, batch_size=64, sequence_length=100, num_workers=4) -> DataLoader:
    # Testing dataset
    dataset = TestingDataset(test_sequences, test_user_ids)

    # Using partial to fix the max_sequence_length
    collate_fn_initialized = partial(collate_fn, max_sequence_len=sequence_length)

    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn_initialized,
        shuffle=True,
        generator=g
    )

    return dataloader