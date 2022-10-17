import pickle
import random
from io import open

import numpy as np
from torch.utils.data import Dataset


class ParallelLanguageDataset(Dataset):
    def __init__(self, data_path_1, data_path_2, num_tokens, max_seq_length, mode, startid, src_len, tgt_len,
                 is_add_cls):
        """
        Initializes the dataset
                Parameters:
                        data_path_1 (str): Path to the exposure pickle file processed in process-data.py
                        data_path_2 (str): Path to the history pickle file processed in process-data.py
                        num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
                        max_seq_length (int): Maximum number of tokens in each sequence pair
        """
        self.num_tokens = num_tokens
        self.mode = mode
        self.startid = startid
        self.data_1, self.data_2, self.data_lengths = load_data(data_path_1, data_path_2, startid,
                                                                src_len, tgt_len)
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.is_add_cls = is_add_cls
        self.batches = gen_batches(num_tokens, self.data_lengths)

    def __getitem__(self, idx):
        src, src_mask = getitem(idx, self.data_1, self.batches, self.startid, True, self.is_add_cls)
        tgt, tgt_mask = getitem(idx, self.data_2, self.batches, self.startid, False, False)
        if self.mode == 'test':
            item_sequence_id = self.batches[idx]
            return src, src_mask, tgt, tgt_mask, item_sequence_id
        else:
            return src, src_mask, tgt, tgt_mask

    def __len__(self):
        return len(self.batches)

    def shuffle_batches(self):
        random.shuffle(self.batches)


def gen_batches(num_tokens, data_lengths):
    """
     Returns the batched data
             Parameters:
                     num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
                     data_lengths (dict): A dict with keys of tuples (length of Exposure sequence, length of corresponding History sequence)
                                         and values of the indices that correspond to these parallel sequences
             Returns:
                     batches (arr): List of each batch (which consists of an array of indices)
     """

    # Shuffle all the indices of same sequence length pair
    for k, v in data_lengths.items():
        random.shuffle(v)

    batches = []
    prev_tokens_in_batch = 1e10
    for k in sorted(data_lengths):
        # v contains indices of the sequences
        v = data_lengths[k]
        total_tokens = (k[0] + k[1]) * len(v)

        # Repeat until all the sequences in this key-value pair are in a batch
        while total_tokens > 0:
            tokens_in_batch = min(total_tokens, num_tokens) - min(total_tokens, num_tokens) % (k[0] + k[1])
            sequences_in_batch = tokens_in_batch // (k[0] + k[1])

            # Combine with previous batch if it can fit
            if tokens_in_batch + prev_tokens_in_batch <= num_tokens:
                batches[-1].extend(v[:sequences_in_batch])
                prev_tokens_in_batch += tokens_in_batch
            else:
                batches.append(v[:sequences_in_batch])
                prev_tokens_in_batch = tokens_in_batch
            # Remove indices from v that have been added in a batch
            v = v[sequences_in_batch:]

            total_tokens = (k[0] + k[1]) * len(v)
    return batches


def load_data(data_path_1, data_path_2, start_id, src_len, tgt_len):
    """
    Loads the pickle files created in preprocess-data.py
            Parameters:
                        data_path_1 (str): Path to the Exposure pickle file processed in process-data.py
                        data_path_2 (str): Path to the History pickle file processed in process-data.py
                        max_seq_length (int): Maximum number of tokens in each sequence pair

            Returns:
                    data_1 (arr): Array of tokenized Exposure sequences
                    data_2 (arr): Array of tokenized History sequences
                    data_lengths (dict): A dict with keys of tuples (length of Exposure sequence, length of corresponding History sequence)
                                         and values of the indices that correspond to these parallel sequences
    """
    with open(data_path_1, 'rb') as f:
        data_1 = pickle.load(f)
    with open(data_path_2, 'rb') as f:
        data_2 = pickle.load(f)

    data_lengths = {}
    for i, (str_1, str_2) in enumerate(zip(data_1, data_2)):
        i += start_id
        if len(str_1) == src_len and len(str_2) == tgt_len:
            if (len(str_1), len(str_2)) in data_lengths:
                data_lengths[(len(str_1), len(str_2))].append(i)  # (len1,len2) sequence indices
            else:
                data_lengths[(len(str_1), len(str_2))] = [i]
    return data_1, data_2, data_lengths


def getitem(idx, data, batches, start_id, src, is_add_cls):
    """
    Retrieves a batch given an index
            Parameters:
                        idx (int): Index of the batch
                        data (arr): Array of tokenized sequences
                        batches (arr): List of each batch (which consists of an array of indices)
                        src (bool): True if the language is the source language, False if it's the target language

            Returns:
                    batch (arr): Array of tokenized Exposure sequences, of size (num_sequences, num_tokens_in_sequence)
                    masks (arr): key_padding_masks for the sequences, of size (num_sequences, num_tokens_in_sequence)
    """

    sequence_indices = batches[idx]

    # sequence_indices  -> user id
    # collect item set by userid and return

    # index start at 0
    if src:
        if is_add_cls is True:
            batch = [[2] + data[i - start_id] for i in sequence_indices]
        else:
            batch = [data[i - start_id] for i in sequence_indices]
    else:
        # If it's in the target language, add [SOS] and [EOS] tokens
        batch = [[2] + data[i - start_id] + [3] for i in sequence_indices]

    # Get the maximum sequence length
    seq_length = 0
    for sequence in batch:
        if len(sequence) > seq_length:
            seq_length = len(sequence)

    masks = []
    for i, sequence in enumerate(batch):
        # Generate the masks for each sequence, False if there's a token, True if there's padding
        masks.append([False for _ in range(len(sequence))] + [True for _ in range(seq_length - len(sequence))])
        # Add 0 padding
        batch[i] = sequence + [0 for _ in range(seq_length - len(sequence))]

    return np.array(batch), np.array(masks)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.max_token = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.max_token = len(self.idx2word)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
