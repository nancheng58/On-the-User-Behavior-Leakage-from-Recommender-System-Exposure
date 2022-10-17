# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np


def filter_items(items, dataname):
    new_items = []
    token_size = 0
    if dataname == "zhihu-1M":
        token_size = 64577
    elif dataname == "mind":
        token_size = 34380
    for sequence in items:
        for item in sequence:
            if item <= token_size:
                new_items.append(item)
    return new_items


def sample_protection_data(items, dataname):
    """
    sample_type:
        random:  sample items randomly.
        pop: sample items according to item popularity.
    """

    # data_file = f'{data_name}.txt'
    # test_file = f'{data_name}_sample_random.txt'
    # np.random.seed(12345)
    items = filter_items(items, dataname)
    item_count = defaultdict(int)
    user_items = defaultdict()

    for item in items:
        item_count[item] += 1

    all_item = list(item_count.keys())
    count = list(item_count.values())
    sum_value = np.sum([x for x in count])
    probability = [value / sum_value for value in count]

    return all_item, probability

    #     test_samples = []
    #     while len(test_samples) < test_num:
    #         if sample_type == 'random':
    #             sample_ids = np.random.choice(all_item, sample_num, replace=False)
    #         else: # sample_type == 'pop':
    #             sample_ids = np.random.choice(all_item, sample_num, replace=False, p=probability)
    #         test_samples.extend(sample_ids)
    #
    # with open(test_file, 'w') as out:
    #     for user, samples in user_neg_items.items():
    #         out.write(user+' '+' '.join(samples)+'\n')

# dataset = "zhihu-1M"
# with open(f'data/{dataset}/processed/itemset.pkl', 'rb') as f:
#     itemset = pickle.load(f)
# itemset = sample_protection_data(itemset,dataset)
