# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from math import floor


def sample_seqs(seqs: List[str], labels: List[bool], seed: int) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels.
        seed: int
            Random integer seed for reproducibility.

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    if not seqs:
        raise ValueError('Sequences must not be an empty list.')
    if not labels:
        raise ValueError('Labels must not be an empty list.')
    assert len(seqs) == len(labels)

    # Set seed
    np.random.seed(seed)

    # Find majority class
    seqs = np.array(seqs)
    labels = np.array(labels)

    true_indices = np.where(labels)[0]
    false_indices = np.where(~labels)[0]

    if len(true_indices) == 0 or len(false_indices) == 0:
        raise ValueError('Must have examples from both classes.')

    if len(true_indices) < len(false_indices):
        # True is minority class
        min_class_seqs = seqs[true_indices]
        num_samples = len(seqs) - 2 * len(min_class_seqs)
        new_indices = np.random.choice(true_indices, num_samples)
    else:
        # False is minority class
        min_class_seqs = seqs[false_indices]
        num_samples = len(seqs) - 2 * len(min_class_seqs)
        new_indices = np.random.choice(false_indices, num_samples)

    new_seqs = seqs[new_indices]
    new_labels = labels[new_indices]
    sampled_seqs = list(np.concatenate((seqs, new_seqs)))
    sampled_labels = list(np.concatenate((labels, new_labels)))
    
    return sampled_seqs, sampled_labels


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }
    encodings = []
    for seq in seq_arr:
        for base in seq:
            encodings += mapping[base]
    
    return np.array(encodings)

def one_hot_encode_digits(digit_arr: List[str]) -> ArrayLike:
    """
    One-hot encode a list of integers in [0, 9] of length n and return and array of size (n, 10), where each row is a length-10 one-hot encoded vector.

    Args:
        digit_arr: List[str]
            List of digits to encode.

    Returns:
        encodings: ArrayLike
            
    """
    mapping = {
        0: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        1: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        2: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        3: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        4: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        5: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        6: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        7: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        8: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        9: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    }
    encodings = []
    for digit in digit_arr:
        encodings.append(mapping[digit])
    
    return np.array(encodings)

def get_subseqs(seq_arr: List[str], new_length: int,  seed: int) -> List[str]:
    """
    Chop up sequences in a list to a list of shorter `new_length`-long sequences, tiling the original sequences with a random starting index.

    Args:
        seq_arr: List[str]
            List of sequences from which to generate subsequences.
        new_length: int
            Desired length of new subsequences.
        seed: int
            Random integer seed for reproducibility.

    Returns:
        subsequences: List[str]

    """
    # Set seed
    np.random.seed(seed)

    a = np.random.randint(0, new_length, len(seq_arr))
    subsequences = []

    for i, seq in enumerate(seq_arr):
        assert len(seq) >= new_length

        # Select a random starting index between 0 and new_length-1
        start_idx = a[i] 
        
        # Compute the number of subsequences to create 
        k_max = floor((len(seq)-start_idx)/new_length) - 1

        # Create subsequences
        indices = [new_length * k + start_idx for k in range(k_max + 1)]
        subseqs = [seq[idx:idx+new_length] for idx in indices]
        subsequences += subseqs
        
    return subsequences