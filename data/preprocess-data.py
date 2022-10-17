import pickle
import random
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import Dictionary

dictionary = Dictionary()
max_token = 0

dataset = 'mind'


# dataset partition and tokenize
def main():
    cross_dim = False  # both of exposure data and history data cross domain or not
    project_path = str(Path(__file__).resolve().parents[0])
    # Create the random train, validation, and test indices
    train_indices, val_indices, test_indices = generate_indices(
        len(load_data(project_path + f'/{dataset}/expose.txt')))
    if cross_dim:
        process_lang_data(project_path + f'/{dataset}/history.txt', 'his', train_indices, val_indices, test_indices)
        process_lang_data(project_path + f'/{dataset}/expose.txt', 'exp', train_indices, val_indices, test_indices)
        # process_user_item_data(project_path + f'/{dataset}/interaction.txt')
        dictionary.word2idx = {}
        dictionary.idx2word = []
        dictionary.max_token = 0
    else:
        process_data(project_path + f'/{dataset}/history.txt', project_path + f'/{dataset}/expose.txt',
                     project_path + f'/{dataset}/interaction.txt', project_path + f'/{dataset}/system_behavior.txt',
                     train_indices, val_indices, test_indices)


def process_data(his_path, exp_path, itemset_path, system_behavior_path, train_indices, val_indices, test_indices):
    his_data = load_data(his_path)
    exp_data = load_data(exp_path)
    itemset_data = load_data(itemset_path)
    system_behavior_data = load_data(system_behavior_path)  # system exposure data which user clicked
    # Tokenize the sequences
    his_sequences = [process_sequences(sequence) for sequence in tqdm(his_data)]
    exp_sequences = [process_sequences(sequence) for sequence in tqdm(exp_data)]
    itemset = [process_sequences(sequence) for sequence in tqdm(itemset_data)]
    system_behavior_set = [process_sequences(sequence) for sequence in tqdm(system_behavior_data)]
    histrain = [his_sequences[i] for i in train_indices]
    exptrain = [exp_sequences[i] for i in train_indices]
    freq_list = Counter()
    for sequence in histrain:
        freq_list.update(sequence)
    for sequence in exptrain:
        freq_list.update(sequence)
    freq_list = freq_list.most_common(dictionary.max_token)
    print(dictionary.max_token)
    # 1 for out of vocabulary words, 2 for start-of-sequence and 3 for end-of-sequence
    freq_list = {freq[0]: i + 4 for i, freq in enumerate(freq_list)}  # # token move back 4 positions
    freq_list['[PAD]'] = 0
    freq_list['[OOV]'] = 1
    freq_list['[SOS]'] = 2
    freq_list['[EOS]'] = 3
    # for k in freq_list.keys():
    #     print(k)
    # map_words(processed_sequences[0], freq_list)
    his_sequences = [map_words(sequence, freq_list) for sequence in
                     tqdm(his_sequences)]  # Convert tokens to indices
    exp_sequences = [map_words(sequence, freq_list) for sequence in
                     tqdm(exp_sequences)]  # Convert tokens to indices
    itemset = [map_words(sequence, freq_list) for sequence in
               tqdm(itemset)]  # Convert tokens to indices
    system_behavior_set = [map_words(sequence, freq_list) for sequence in
                           tqdm(system_behavior_set)]  # Convert tokens to indices

    # Split the data
    histrain = [his_sequences[i] for i in train_indices]

    hisval = [his_sequences[i] for i in val_indices]
    histest = [his_sequences[i] for i in test_indices]
    exptrain = [exp_sequences[i] for i in train_indices]
    expval = [exp_sequences[i] for i in val_indices]
    exptest = [exp_sequences[i] for i in test_indices]
    his = histrain + hisval + histest
    with open(f'data/{dataset}/processed/his.txt', 'w') as f:
        for sequence in his:
            for index in range(0, len(sequence)):
                if index != len(sequence) - 1:
                    f.write(str(sequence[index]) + " ")
                else:
                    f.write(str(sequence[index]))
            f.write('\n')
    save_data('his', histrain, hisval, histest, freq_list)
    save_data('exp', exptrain, expval, exptest, freq_list)
    with open(f'data/{dataset}/processed/itemset.pkl', 'wb') as f:
        pickle.dump(itemset, f)
    with open(f'data/{dataset}/processed/system_behavior_set.pkl', 'wb') as f:
        pickle.dump(system_behavior_set, f)


def save_data(lang, train, val, test, freq_list):
    # Save the data
    with open(f'data/{dataset}/processed/{lang}/train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(f'data/{dataset}/processed/{lang}/val.pkl', 'wb') as f:
        pickle.dump(val, f)
    with open(f'data/{dataset}/processed/{lang}/test.pkl', 'wb') as f:
        pickle.dump(test, f)
    with open(f'data/{dataset}/processed/{lang}/freq_list.pkl', 'wb') as f:
        pickle.dump(freq_list, f)


def process_lang_data(data_path, lang, train_indices, val_indices, test_indices):
    """
        (Deprecated)
    """
    lang_data = load_data(data_path)
    # Tokenize the sequences
    processed_sequences = [process_sequences(sequence) for sequence in tqdm(lang_data)]

    train = [processed_sequences[i] for i in train_indices]

    freq_list = Counter()
    for sequence in train:
        freq_list.update(sequence)
    freq_list = freq_list.most_common(dictionary.max_token)
    freq_list = {freq[0]: i + 4 for i, freq in enumerate(freq_list)}  # token move back 4 positions
    freq_list['[PAD]'] = 0
    freq_list['[OOV]'] = 1
    freq_list['[SOS]'] = 2
    freq_list['[EOS]'] = 3
    old_pr = processed_sequences
    processed_sequences = [map_words(sequence, freq_list) for sequence in
                           tqdm(processed_sequences)]  # Convert tokens to indices

    # Split the data
    train = [processed_sequences[i] for i in train_indices]
    val = [processed_sequences[i] for i in val_indices]
    test = [processed_sequences[i] for i in test_indices]

    # # Save the data
    with open(f'data/{dataset}/processed/{lang}/train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(f'data/{dataset}/processed/{lang}/val.pkl', 'wb') as f:
        pickle.dump(val, f)
    with open(f'data/{dataset}/processed/{lang}/test.pkl', 'wb') as f:
        pickle.dump(test, f)
    with open(f'data/{dataset}/processed/{lang}/freq_list.pkl', 'wb') as f:
        pickle.dump(freq_list, f)
    with open(f'data/{dataset}/processed/{lang}/dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)


def process_sequences(sequence):
    """
     Processes sequences by lowercasing text, ignoring punctuation, and using Spacy tokenization
             Parameters:
                     lang_model: Spacy language model
                     sequence (str): sequence to be tokenized
                     punctuation (arr): Array of punctuation to be ignored
             Returns:
                     sequence (arr): Tokenized sequence
     """
    for word in sequence:
        dictionary.add_word(str(word))

    # Tokenize file content
    # each sequence has a ids ,denote contains word id
    idss = []
    ids = []
    for word in sequence:
        ids.append(dictionary.word2idx[str(word)])
    idss.append(torch.tensor(ids).type(torch.int64))
    sequence = ids
    # sequence = [tok.text for tok in lang_model.tokenizer(sequence) if tok.text not in punctuation]

    return sequence


def load_data(data_path):
    data = []
    with open(data_path, encoding='utf-8') as fp:
        for line in fp:
            data.append(line.strip().split('\t'))
    return data


def map_words(sequence, freq_list):
    list1 = [freq_list[int(word)] for word in sequence if int(word) in freq_list]
    return list1


def generate_indices(data_len):
    """
     Generate train, validation, and test indices
             Parameters:
                     data_len (int): Amount of sequences in the dataset
             Returns:
                     train_indices (arr): Array of indices for train dataset
                     val_indices (arr): Array of indices for validation dataset
                     test_indices (arr): Array of indices for test dataset
     """
    # 8:1:1 train validation test split
    train_idx = int(data_len * 0.8)
    val_idx = train_idx + int(data_len * 0.1)
    test_idx = val_idx + int(data_len * 0.1)

    train_indices = [i for i in range(0, int(train_idx))]
    valid_indices = [i for i in range(train_idx, val_idx)]
    test_indices = [i for i in range(val_idx, test_idx)]
    random.shuffle(train_indices)
    random.shuffle(valid_indices)
    # random.shuffle(test_indices)

    return train_indices, valid_indices, test_indices


if __name__ == "__main__":
    main()
