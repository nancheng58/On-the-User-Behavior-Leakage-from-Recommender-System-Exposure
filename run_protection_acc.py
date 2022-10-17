import pickle
import random
from pathlib import Path

import click
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ParallelLanguageDataset
from model import *
from generate_samples import sample_protection_data

device = "cuda"
num_workers = 4
random.seed(12345)


@click.command()
@click.argument('dataset', type=str, default="mind")
@click.argument('num_epochs', type=int, default=200)
@click.argument('max_seq_length', type=int, default=10)
@click.argument('num_tokens', type=int, default=4000)
@click.argument('src_vocab_size', type=int, default=34376 + 4)  # zhihu: 24948 # mind: 34376
@click.argument('tgt_vocab_size', type=int, default=34376 + 4)
@click.argument('exp_length', type=int, default=10)
@click.argument('his_length', type=int, default=5)
@click.argument('d_model', type=int, default=128)
@click.argument('num_encoder_layers', type=int, default=1)
@click.argument('num_decoder_layers', type=int, default=1)
@click.argument('dim_feedforward', type=int, default=128)
@click.argument('nhead', type=int, default=1)
@click.argument('pos_dropout', type=float, default=0.1)
@click.argument('trans_dropout', type=float, default=0.1)
@click.argument('n_warmup_steps', type=int, default=4000)
@click.argument('sample', type=bool, default=True)
@click.argument('encoder', type=str, default='Transformer')  # 'Transformer,meanpool or maxpool'
@click.argument('decoder', type=str, default='Transformer')  # 'Transformer,GRU, LSTM, Point'
def main(**kwargs):
    project_path = str(Path(__file__).resolve().parents[0])
    if kwargs['decoder'] != 'Transformer':  # add [CLS] or not
        is_add_cls = True
    else:
        is_add_cls = False
    dataset = kwargs['dataset']
    with open(project_path + f'/data/{dataset}/processed/his/train.pkl', 'rb') as f:
        data_train = pickle.load(f)
    with open(project_path + f'/data/{dataset}/processed/his/val.pkl', 'rb') as f:
        data_val = pickle.load(f)
    test_dataset = ParallelLanguageDataset(project_path + f'/data/{dataset}/processed/exp/test.pkl',
                                           project_path + f'/data/{dataset}/processed/his/test.pkl',
                                           kwargs['num_tokens'], kwargs['max_seq_length'], 'test',
                                           len(data_train) + len(data_val), kwargs['exp_length'], kwargs['his_length'],
                                           is_add_cls)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
    if dataset == 'mind':
        epoch = 180
    else:  # zhihu-1M
        epoch = 380
    model = torch.load(
        f'output/{dataset}/dataset_{dataset}_encoder_Transformer_decoder_Transformer_length_10to5_epoch_{epoch}.pth').to(
        device)
    protect_sas_embed = np.loadtxt(f'output/{dataset}/SASRec-{dataset}-0.pth')
    item_embeddings = load_model_embedding(protect_sas_embed)
    # "pop-inbatch"  "pop"
    proportion = [0, 0.2, 0.4, 0.6, 0.8]  # 0, 0.2, 0.4, 0.6,
    for k in range(len(proportion)):
        protection(test_loader, model, kwargs['his_length'], kwargs['src_vocab_size'], kwargs['dataset'],
                   proportion[k], "random", "random", item_embeddings)
    for k in range(len(proportion)):
        protection(test_loader, model, kwargs['his_length'], kwargs['src_vocab_size'], kwargs['dataset'],
                   proportion[k], "pop", "random", item_embeddings)
    for k in range(len(proportion)):
        protection(test_loader, model, kwargs['his_length'], kwargs['src_vocab_size'], kwargs['dataset'],
                   proportion[k], "pop-inbatch", "random", item_embeddings)


def detokenize(sequence, freq_list):
    freq_list = {v: k for k, v in freq_list.items()}
    return [freq_list[token] for token in sequence]


def load_model_embedding(protect_sas_emb):
    embeddings = torch.as_tensor(protect_sas_emb)
    return embeddings
def replace(sample_num, replace_type, src_sequence, model_embeddings, B_emb):
    replace_list = []
    replace_set = set()
    user_behavoir_item_size = model_embeddings.shape[0]
    if replace_type == 'random':
        while len(replace_list) < sample_num:
            replace_position = random.randint(0, 9)  # [0~9]
            if replace_position not in replace_set:
                replace_set.add(replace_position)
                replace_list.append(replace_position)
        return replace_list
    else:
        # src_sequence = src_sequence.tolist()
        items_embedding = []
        for item in src_sequence:
            if item < user_behavoir_item_size:
                items_embedding.append(model_embeddings[item])
            else:
                items_embedding.append(model_embeddings[0])

        items_embedding = torch.stack(items_embedding)
        # src_embedding = torch.mean(items_embedding, dim=0)
        cosine_similarity = []
        for i in range(len(src_sequence)):
            x = torch.cosine_similarity(B_emb, items_embedding[i], dim=0)
            cosine_similarity.append(x)
        # cosine_similarity = cosine_similarity
        order1 = np.argsort(cosine_similarity).tolist()
        # list.reverse(order1)
        #probability = torch.softmax(torch.tensor(cosine_similarity), dim=0).tolist()
        # all_position = [i for i in range(0, 10)]  # [0,10) setting src length is 10 as default
        # x = sum(cosine_similarity)
        # probability = [p / x for p in cosine_similarity]
        #
        # replace_list = np.random.choice(all_position, sample_num, replace=False, p=probability)
        return order1[:sample_num]


# def replace(sample_num, replace_type, src_sequence, model_embeddings):
#     replace_list = []
#     replace_set = set()
#     user_behavoir_item_size = model_embeddings.shape[0]
#     if replace_type == 'random':
#         while len(replace_list) < sample_num:
#             replace_position = random.randint(0, 9)  # [0,9]
#             if replace_position not in replace_set:
#                 replace_set.add(replace_position)
#                 replace_list.append(replace_position)
#     else:
#         src_sequence = src_sequence.tolist()
#         items_embedding = []
#         for item in src_sequence:
#             if item < user_behavoir_item_size:
#                 items_embedding.append(model_embeddings[item])
#             else:
#                 items_embedding.append(model_embeddings[0])
#
#         items_embedding = torch.stack(items_embedding)
#         src_embedding = torch.mean(items_embedding, dim=0)
#         cosine_similarity = []
#         for i in range(len(src_sequence)):
#             cosine_similarity.append(torch.cosine_similarity(src_embedding, items_embedding[i], dim=0))
#         probability = torch.softmax(torch.tensor(cosine_similarity), dim=0).tolist()
#         all_position = [i for i in range(0, 10)]  # [0,10) setting src length is 10 as default
#         x = sum(probability)
#         probability = [p / x for p in probability]
#         replace_list = np.random.choice(all_position, sample_num, replace=False, p=probability)
#
#     return replace_list

def cal_acc(privacy_embedding, sequence, model_embeddings):
    cosine_similarity = []
    items_embedding = []
    user_behavoir_item_size = model_embeddings.shape[0]
    for item in sequence:
        if item < user_behavoir_item_size:
            items_embedding.append(model_embeddings[item])
        else:
            items_embedding.append(0)
    available_len = 0
    for i in range(len(sequence)):
        if isinstance(items_embedding[i], torch.Tensor) != 0:  # 不是int
            x = (torch.cosine_similarity(privacy_embedding, items_embedding[i], dim=0))
            cosine_similarity.append(x)
            available_len += 1
    sum_simi = float(sum(cosine_similarity))
    if available_len == 0:
        return 0
    else:
        sum_simi = sum_simi/available_len
        return sum_simi  # sum_simi*sum_simi

def protection(test_loader, model, tgt_len, tokensize, dataset, proportion, sample_type, replace_type, item_embeddings):
    # Load the trained model, Spacy tokenizer, and frequency lists

    with open(f'data/{dataset}/processed/itemset.pkl', 'rb') as f:
        itemset = pickle.load(f)
    with open(f'data/{dataset}/processed/system_behavior_set.pkl', 'rb') as f:
        system_behavior_set = pickle.load(f)
    with open(f'data/{dataset}/processed/sequence2user.pkl', 'rb') as f:
        sequence2user = pickle.load(f)
    src_len = 10
    sample_num = math.ceil(src_len * proportion)
    if sample_type == "random":
        all_item, _ = sample_protection_data(itemset, dataname=dataset)
    elif sample_type == "pop":
        all_item, probability = sample_protection_data(itemset, dataname=dataset)
    pbar = tqdm(total=len(iter(test_loader)), leave=False)
    model.eval()
    acc = 0
    total_sequence = 0
    # item_sequence is the sequence indice list in a batch
    for src, src_key_padding_mask, tgt, tgt_key_padding_mask, item_sequence in iter(test_loader):
        with torch.no_grad():
            src, src_key_padding_mask = src[0].to(device), src_key_padding_mask[0].to(device)
            tgt, tgt_key_padding_mask = tgt[0].to(device), tgt_key_padding_mask[0].to(device)

            assert len(src) == len(tgt)
            assert len(src) == len(item_sequence)

            # in-batch
            if sample_type == "pop-inbatch":
                items = []
                for index in range(len(src)):
                    item_sequenceid = int(item_sequence[index])
                    userid = sequence2user[item_sequenceid]
                    items.append(itemset[userid])
                all_item, probability = sample_protection_data(items, dataname=dataset)
            for index in range(len(src)):
                i = 0
                src_sequence = src[index]
                tgt_sequence = tgt[index]
                item_sequenceid = int(item_sequence[index])
                userid = sequence2user[item_sequenceid]
                interactionset = set(itemset[userid])
                checkset = set(tgt_sequence.tolist()[1:-1])
                B = set(system_behavior_set[userid])
                # B = interactionset
                # if len(B) <= 10:
                #     continue
                B_emb = []
                user_behavoir_item_size = item_embeddings.shape[0]
                for item in B:
                    if item < user_behavoir_item_size:
                        B_emb.append(item_embeddings[item])
                B_emb = torch.mean(torch.stack(B_emb), dim=0)
                assert len(interactionset & checkset) == len(checkset)
                assert len(interactionset & B) == len(B)
                sequence = src_sequence.clone().detach().tolist()
                if sample_type == 'random':
                    sample_ids = np.random.choice(all_item, sample_num, replace=False)
                elif sample_type == 'pop' or sample_type == 'pop-inbatch':
                    sample_ids = np.random.choice(all_item, sample_num, replace=False, p=probability)
                replace_position = replace(sample_num, replace_type, sequence, item_embeddings, B_emb)
                for j in range(len(replace_position)):
                    sequence[replace_position[j]] = sample_ids[j]
                E = set(sequence)
                acc += len(B & E) / len(E)
                total_sequence += 1
            pbar.update(1)
    acc = acc / total_sequence
    print_infor = {
        "sample_type ": sample_type,
        "replace_method": replace_type,
        "Proportion": proportion,
        "accuracy": acc
    }
    print(print_infor)
    pbar.close()


if __name__ == "__main__":
    main()
