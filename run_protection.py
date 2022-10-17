import pickle
import pickle
import random
from pathlib import Path

import click
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ParallelLanguageDataset
from metrics import metric_at_k_set
from model import *
from generate_samples import sample_protection_data

device = "cuda"
num_workers = 0
random.seed(12345)


@click.command()
@click.argument('dataset', type=str, default="zhihu-1M")
@click.argument('num_epochs', type=int, default=200)
@click.argument('max_seq_length', type=int, default=10)
@click.argument('num_tokens', type=int, default=4000)
@click.argument('src_vocab_size', type=int, default=64573 + 4)  # zhihu: 64573 # mind: 34376
@click.argument('tgt_vocab_size', type=int, default=64573 + 4)
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
    train_dataset = ParallelLanguageDataset(project_path + f'/data/{dataset}/processed/exp/train.pkl',
                                            project_path + f'/data/{dataset}/processed/his/train.pkl',
                                            kwargs['num_tokens'], kwargs['max_seq_length'], 'train', 0,
                                            kwargs['exp_length'], kwargs['his_length'], is_add_cls)
    with open(project_path + f'/data/{dataset}/processed/his/train.pkl', 'rb') as f:
        data_train = pickle.load(f)
    # Set batch_size=1 because all the batching is handled in the ParallelLanguageDataset class
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_dataset = ParallelLanguageDataset(project_path + f'/data/{dataset}/processed/exp/val.pkl',
                                            project_path + f'/data/{dataset}/processed/his/val.pkl',
                                            kwargs['num_tokens'], kwargs['max_seq_length'], 'valid', len(data_train),
                                            kwargs['exp_length'], kwargs['his_length'], is_add_cls)
    with open(project_path + f'/data/{dataset}/processed/his/val.pkl', 'rb') as f:
        data_val = pickle.load(f)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataset = ParallelLanguageDataset(project_path + f'/data/{dataset}/processed/exp/test.pkl',
                                           project_path + f'/data/{dataset}/processed/his/test.pkl',
                                           kwargs['num_tokens'], kwargs['max_seq_length'], 'test',
                                           len(data_train) + len(data_val), kwargs['exp_length'], kwargs['his_length'],
                                           is_add_cls)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
    if kwargs['encoder'] in ['meanpool', 'maxpool']:
        encoder = PoolEncoder(kwargs['src_vocab_size'], kwargs['d_model'], is_add_cls, kwargs['encoder']).to(device)
    else:
        encoder = TransformerEncoder(kwargs['src_vocab_size'], kwargs['d_model'], kwargs['nhead'],
                                     kwargs['num_encoder_layers'], kwargs['dim_feedforward'],
                                     kwargs['trans_dropout'], is_add_cls).to(device)
    if kwargs['decoder'] in ['LSTM', 'GRU']:
        decoder = RNNDecoder(kwargs['decoder'], kwargs['tgt_vocab_size'], kwargs['d_model'], kwargs['d_model'],
                             kwargs['num_decoder_layers'], kwargs['trans_dropout']).to(device)
    elif kwargs['decoder'] == 'Transformer':
        decoder = TransformerDecoder(kwargs['tgt_vocab_size'], kwargs['d_model'], kwargs['nhead'],
                                     kwargs['num_encoder_layers'],
                                     kwargs['dim_feedforward'], kwargs['max_seq_length'],
                                     kwargs['pos_dropout'], kwargs['trans_dropout']).to(device)
    else:
        decoder = PointDecoder(kwargs['tgt_vocab_size'], kwargs['d_model']).to(device)
    model = EncoderDecoder(encoder=encoder, decoder=decoder, encoder_type=kwargs['encoder'],
                           decoder_type=kwargs['decoder'])
    epoch = 0
    if dataset == 'mind':
        epoch = 180
    else:  # zhihu-1M
        epoch = 380
    model = torch.load(
        f'output/{dataset}/dataset_{dataset}_encoder_Transformer_decoder_Transformer_length_10to5_epoch_{epoch}.pth').to(
        device)
    protect_sas_embed = np.loadtxt(f'output/{dataset}/SASRec-{dataset}-0.pth')
    item_embeddings = load_model_embedding(protect_sas_embed)
    model_signature = 'dataset_{}_encoder_{}_decoder_{}_length_{}to{}'. \
        format(kwargs['dataset'], kwargs['encoder'], kwargs['decoder'], kwargs['exp_length'], kwargs['his_length'])
    # Use Xavier normal initialization in the transformer
    with open(project_path + f'/data/{dataset}/processed/his/test.pkl', 'rb') as f:
        data_test = pickle.load(f)
    # "pop-inbatch"  "pop"
    proportion = [0.2, 0.4, 0.6, 0.8]
    for k in range(len(proportion)):
        # protection(test_loader, model, kwargs['his_length'], kwargs['src_vocab_size'], kwargs['dataset'],
        #            proportion[k], "random", "random", item_embeddings)
        protection(test_loader, model, kwargs['his_length'], kwargs['src_vocab_size'], kwargs['dataset'],
                   proportion[k], "random", "h", item_embeddings)
        protection(test_loader, model, kwargs['his_length'], kwargs['src_vocab_size'], kwargs['dataset'],
                   proportion[k], "pop", "h", item_embeddings)
        protection(test_loader, model, kwargs['his_length'], kwargs['src_vocab_size'], kwargs['dataset'],
                   proportion[k], "pop-inbatch", "h", item_embeddings)


def detokenize(sequence, freq_list):
    freq_list = {v: k for k, v in freq_list.items()}
    return [freq_list[token] for token in sequence]


def load_model_embedding(protect_sas_emb):
    embeddings = torch.as_tensor(protect_sas_emb)
    return embeddings


def replace(sample_num, replace_type, src_sequence, model_embeddings):
    replace_list = []
    replace_set = set()
    user_behavoir_item_size = model_embeddings.shape[0]
    if replace_type == 'random':
        while len(replace_list) < sample_num:
            replace_position = random.randint(0, 9)  # [0,9]
            if replace_position not in replace_set:
                replace_set.add(replace_position)
                replace_list.append(replace_position)
        return replace_list
    else:
        src_sequence = src_sequence.tolist()
        items_embedding = []
        for item in src_sequence:
            if item < user_behavoir_item_size:
                items_embedding.append(model_embeddings[item])
            else:
                items_embedding.append(model_embeddings[0])

        items_embedding = torch.stack(items_embedding)
        src_embedding = torch.mean(items_embedding, dim=0)
        cosine_similarity = []
        for i in range(len(src_sequence)):
            x = 0.5 + (0.5 * torch.cosine_similarity(src_embedding, items_embedding[i], dim=0))
            cosine_similarity.append(x)
        # cosine_similarity = cosine_similarity
        order1 = np.argsort(cosine_similarity).tolist()
        list.reverse(order1)
        # probability = torch.softmax(torch.tensor(cosine_similarity), dim=0).tolist()
        # all_position = [i for i in range(0, 10)]  # [0,10) setting src length is 10 as default
        # x = sum(cosine_similarity)
        # probability = [p / x for p in cosine_similarity]
        #
        # replace_list = np.random.choice(all_position, sample_num, replace=False, p=probability)
        return order1[:sample_num]


def protection(test_loader, model, tgt_len, tokensize, dataset, proportion, sample_type, replace_type, item_embeddings):
    # Load the trained model, Spacy tokenizer, and frequency lists
    with open(f'data/{dataset}/processed/his/freq_list.pkl', 'rb') as f:
        fr_freq_list = pickle.load(f)
    with open(f'data/{dataset}/processed/itemset.pkl', 'rb') as f:
        itemset = pickle.load(f)
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
    pred_list_full = []
    tgt_list = []
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
                # B = set(system_behavior_set[userid])
                assert len(interactionset & checkset) == len(checkset)
                # assert len(interactionset & B) == len(B)
                pred_sequence_full = []
                if sample_type == 'random':
                    sample_ids = np.random.choice(all_item, sample_num, replace=False)
                elif sample_type == 'pop' or sample_type == 'pop-inbatch':
                    sample_ids = np.random.choice(all_item, sample_num, replace=False, p=probability)
                sequence = src_sequence.clone().detach()
                replace_position = replace(sample_num, replace_type, sequence, item_embeddings)

                for j in range(len(replace_position)):
                    sequence[replace_position[j]] = sample_ids[j]
                total_sequence += 1
                enc_state = model.encoder(src=sequence.unsqueeze(0)).to(device)
                translated_sequence = [fr_freq_list['[SOS]']]
                while int(translated_sequence[-1]) != fr_freq_list['[EOS]'] and i < tgt_len:  # inference
                    translated_sequence_tensor = torch.tensor(translated_sequence).unsqueeze(0).to(device)
                    # tgt_mask = gen_nopeek_mask(translated_sequence_tensor.shape[1]).to(device)
                    output = model.decoder(enc_states=enc_state,
                                           tgt=rearrange(translated_sequence_tensor, 'n s -> s n').long(),
                                           tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                           tgt_mask=None)
                    output = output.squeeze(0).to(device)
                    _, indices = torch.topk(output, tokensize)  # sort by Probability
                    indices = indices[-1].to('cpu').numpy()
                    pred_item_full = [int(indices[j]) for j in range(50)]
                    translated_sequence.append(int(indices[0]))
                    pred_sequence_full.append(pred_item_full)
                    i += 1
                pred_list_full.append(pred_sequence_full)  # shape[sequence, word, pred_token]
                tgt_list.append(tgt_sequence.tolist()[1:-1])
            pbar.update(1)
    print_infor = {
        "sample_type ": sample_type,
        "replace_method": replace_type,
        "Proportion": proportion
    }
    print(print_infor)
    get_scores_sample(pred_list_full, tgt_list, tgt_len)
    pbar.close()


def get_scores_sample(pred_list, tgt_list, tgt_len):
    for k in [10]:
        NDCG, MRR, Recall = metric_at_k_set(tgt_list, pred_list, k, tgt_len)
        post_fix = {
            "R@": k,
            "Recall": ('%.4f' % Recall),
            "NDCG": ('%.4f' % NDCG),
            "MRR": ('%.4f' % MRR),
        }
        print(post_fix)


if __name__ == "__main__":
    main()
