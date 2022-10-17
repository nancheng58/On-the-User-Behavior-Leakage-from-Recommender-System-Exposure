import pickle
from pathlib import Path

import click
import torch
import torch.nn as nn
from einops import rearrange
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ParallelLanguageDataset
from metrics import metric_at_k_set, metric_at_k_set_point
from model import TransformerDecoder, TransformerEncoder, RNNDecoder, EncoderDecoder, PoolEncoder, PointDecoder
from optim import ScheduledOptim

device = "cuda"
num_workers = 0


@click.command()
@click.argument('dataset', type=str, default="zhihu-1M")
@click.argument('num_epochs', type=int, default=200)
@click.argument('max_seq_length', type=int, default=10)
@click.argument('num_tokens', type=int, default=4000)
@click.argument('exp_item_size', type=int, default=64573 + 4)  # zhihu: 64573 # mind: 34376
@click.argument('his_item_size', type=int, default=64573 + 4)
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
@click.argument('decoder', type=str, default='Point')  # 'Transformer,GRU, LSTM, Point'
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
        encoder = PoolEncoder(kwargs['exp_item_size'], kwargs['d_model'], is_add_cls, kwargs['encoder']).to(device)
    else:
        encoder = TransformerEncoder(kwargs['exp_item_size'], kwargs['d_model'], kwargs['nhead'],
                                     kwargs['num_encoder_layers'], kwargs['dim_feedforward'],
                                     kwargs['trans_dropout'], is_add_cls).to(device)
    if kwargs['decoder'] in ['LSTM', 'GRU']:
        decoder = RNNDecoder(kwargs['decoder'], kwargs['his_item_size'], kwargs['d_model'], kwargs['d_model'],
                             kwargs['num_decoder_layers'], kwargs['trans_dropout']).to(device)
    elif kwargs['decoder'] == 'Transformer':
        decoder = TransformerDecoder(kwargs['his_item_size'], kwargs['d_model'], kwargs['nhead'],
                                     kwargs['num_encoder_layers'],
                                     kwargs['dim_feedforward'], kwargs['max_seq_length'],
                                     kwargs['pos_dropout'], kwargs['trans_dropout']).to(device)
    else:
        decoder = PointDecoder(kwargs['his_item_size'], kwargs['d_model']).to(device)
    model = EncoderDecoder(encoder=encoder, decoder=decoder, encoder_type=kwargs['encoder'],
                           decoder_type=kwargs['decoder'])
    model_signature = 'dataset_{}_encoder_{}_decoder_{}_length_{}to{}'. \
        format(kwargs['dataset'], kwargs['encoder'], kwargs['decoder'], kwargs['exp_length'], kwargs['his_length'])
    # Use Xavier normal initialization in the transformer
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    optim = ScheduledOptim(
        Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        kwargs['d_model'], kwargs['n_warmup_steps'])

    # Use cross entropy loss, ignoring any padding
    criterion = nn.CrossEntropyLoss()

    train(train_loader, valid_loader, test_loader, model, optim, criterion, kwargs['num_epochs'], kwargs['his_length'],
          kwargs['exp_item_size'], kwargs['dataset'], kwargs['decoder'], model_signature)


def forward_model(model, src, tgt):
    src = src.unsqueeze(0).long().to(device)
    tgt = torch.tensor(tgt).unsqueeze(0).to(device)
    tgt_mask = gen_nopeek_mask(tgt.shape[1]).to(device)
    output = model.forward(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                           tgt_mask=tgt_mask)

    return output.squeeze(0)


def detokenize(sequence, freq_list):
    freq_list = {v: k for k, v in freq_list.items()}
    return [freq_list[token] for token in sequence]


def train(train_loader, valid_loader, test_loader, model, optim, criterion, num_epochs, tgt_len, tokensize, dataset,
          decoder_type, model_signature):
    print_every = 500
    model.train()

    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0
    for epoch in range(num_epochs):
        pbar = tqdm(total=print_every, leave=False)
        total_loss = 0

        # Shuffle batches every epoch
        train_loader.dataset.shuffle_batches()
        for step, (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in enumerate(iter(train_loader)):
            total_step += 1
            src, src_key_padding_mask = src[0].to(device), src_key_padding_mask[0].to(device)
            tgt, tgt_key_padding_mask = tgt[0].to(device), tgt_key_padding_mask[0].to(device)
            memory_key_padding_mask = src_key_padding_mask.clone()
            # Send the batches and key_padding_masks to gpu (key_padding_masks : to solve sequence length not same)
            # Forward
            optim.zero_grad()
            if decoder_type == 'Point':
                # Point Decoder
                tgt_out = tgt[:, 1:-1]
                tgt_out = tgt_out.contiguous()
                tgt_out = tgt_out.long()
                outputs = model(src, src_key_padding_mask=src_key_padding_mask)
                probability = 1.0 / float(tgt_len)
                tgt_probabilities_list = []
                for index in range(len(src)):
                    tgt_sequence = tgt_out[index]
                    tgtset = tgt_sequence.tolist()
                    tgt_probabilities = torch.full((tokensize,), 1e-6, dtype=torch.float32)
                    for i in range(len(tgtset)):
                        tgt_probabilities[tgtset[i]] = probability
                    tgt_probabilities_list.append(tgt_probabilities)
                tgt_outs = torch.stack(tensors=tgt_probabilities_list).to(device)
                loss = criterion(outputs, tgt_outs)
            else:
                # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
                tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
                tgt_out = tgt_out.long()
                tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(device)
                if decoder_type in ['LSTM', 'GRU']:  # outputs and states
                    outputs, _ = model(src, tgt_inp, src_key_padding_mask)
                else:
                    outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1],
                                    memory_key_padding_mask, tgt_mask)
                loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))
            # Backpropagation and update optim
            loss.backward()
            optim.step_and_update_lr()

            total_loss += loss.item()
            train_losses.append((step, loss.item()))
            pbar.update(1)

            if step % print_every == print_every - 1:
                pbar.close()
                print(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_loader)}] \t '
                      f'Train Loss: {total_loss / print_every}')
                total_loss = 0

                pbar = tqdm(total=print_every, leave=False)
        if epoch >= 0 and epoch % 20 == 0:
            torch.save(model, fr'output/{dataset}/{model_signature}_epoch_{epoch}.pth')
            test(epoch, test_loader, model, tgt_len, tokensize, decoder_type, dataset)
        pbar.close()
        # Validate every epoch
        val_loss = validate(valid_loader, model, criterion, decoder_type, tgt_len, tokensize)
        val_losses.append((total_step, val_loss))
        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model, f'output/{dataset}/{model_signature}_epoch_0.pth')  # model with the lowest loss
        print(f'Val Loss: {val_loss}')
    return train_losses, val_losses


def validate(valid_loader, model, criterion, decoder_type, tgt_len, tokensize):
    pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    total_loss = 0
    for src, src_key_padding_mask, tgt, tgt_key_padding_mask in iter(valid_loader):
        with torch.no_grad():
            src, src_key_padding_mask = src[0].to(device), src_key_padding_mask[0].to(device)
            tgt, tgt_key_padding_mask = tgt[0].to(device), tgt_key_padding_mask[0].to(device)
            memory_key_padding_mask = src_key_padding_mask.clone()
            if decoder_type == 'Point':
                # Point Decoder
                tgt_out = tgt[:, 1:-1]
                tgt_out = tgt_out.contiguous()
                tgt_out = tgt_out.long()
                outputs = model(src, src_key_padding_mask=src_key_padding_mask)
                probability = 1.0 / float(tgt_len)
                tgt_probabilities_list = []
                for index in range(len(src)):
                    tgt_sequence = tgt_out[index]
                    tgtset = tgt_sequence.tolist()
                    tgt_probabilities = torch.full((tokensize,), 1e-6, dtype=torch.float32)
                    for i in range(len(tgtset)):
                        tgt_probabilities[tgtset[i]] = probability
                    tgt_probabilities_list.append(tgt_probabilities)
                tgt_outs = torch.stack(tensors=tgt_probabilities_list).to(device)
                loss = criterion(outputs, tgt_outs)
            else:
                # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
                tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
                tgt_out = tgt_out.long()
                tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(device)
                if decoder_type in ['LSTM', 'GRU']:  # outputs and states
                    outputs, _ = model(src, tgt_inp, src_key_padding_mask)
                else:
                    outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1],
                                    memory_key_padding_mask, tgt_mask)
                loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))
            total_loss += loss.item()
            pbar.update(1)

    pbar.close()
    model.train()
    return total_loss / len(valid_loader)


def test(epoch, test_loader, model, tgt_len, tokensize, decoder_type, dataset):
    # Load the trained model, Spacy tokenizer, and frequency lists
    with open(f'data/{dataset}/processed/his/freq_list.pkl', 'rb') as f:
        fr_freq_list = pickle.load(f)
    with open(f'data/{dataset}/processed/itemset.pkl', 'rb') as f:
        itemset = pickle.load(f)
    with open(f'data/{dataset}/processed/sequence2user.pkl', 'rb') as f:
        sequence2user = pickle.load(f)

    pbar = tqdm(total=len(iter(test_loader)), leave=False)
    model.eval()
    pred_list_sample = []
    pred_list_full = []
    tgt_list = []
    # item_sequence is the sequence indice list in this batch
    for src, src_key_padding_mask, tgt, tgt_key_padding_mask, item_sequence in iter(test_loader):
        with torch.no_grad():
            src, src_key_padding_mask = src[0].to(device), src_key_padding_mask[0].to(device)
            tgt, tgt_key_padding_mask = tgt[0].to(device), tgt_key_padding_mask[0].to(device)

            assert len(src) == len(tgt)
            assert len(src) == len(item_sequence)
            for index in range(len(src)):
                i = 0
                src_sequence = src[index]
                tgt_sequence = tgt[index]
                item_sequenceid = int(item_sequence[index])
                userid = sequence2user[item_sequenceid]
                interactionset = set(itemset[userid])
                checkset = set(tgt_sequence.tolist()[1:-1])
                assert len(interactionset & checkset) == len(checkset)
                pred_sequence_sample = []
                pred_sequence_full = []
                enc_state = model.encoder(src=src_sequence.unsqueeze(0)).to(device)
                if decoder_type == 'Point':
                    output = model.decoder(enc_state).squeeze(0).to(device)
                    # output = output[0, :]
                    _, indices = torch.topk(output, tokensize)  # sort by Probability
                    indices = indices.to('cpu').numpy()
                    for j in range(len(indices)):
                        if int(indices[j]) in interactionset:
                            pred_sequence_sample.append(int(indices[j]))
                            if len(pred_sequence_sample) >= 50:
                                break
                    pred_sequence_full = [int(indices[j]) for j in range(50)]
                    pred_list_sample.append(pred_sequence_sample)  # [sequence, word, pred_token]
                    pred_list_full.append(pred_sequence_full)
                    tgt_list.append(tgt_sequence.tolist()[1:-1])
                else:
                    if decoder_type in ['LSTM', 'GRU']:
                        enc_state = enc_state.unsqueeze(0).repeat(model.decoder.num_decoder_layers, 1, 1)
                        enc_cell = enc_state
                    translated_sequence = [fr_freq_list['[SOS]']]
                    while int(translated_sequence[-1]) != fr_freq_list['[EOS]'] and i < tgt_len:  # inference
                        if decoder_type in ['LSTM', 'GRU']:
                            if decoder_type == 'GRU':
                                output, enc_state = model.decoder(
                                    torch.tensor([translated_sequence[-1]]).unsqueeze(0).long()
                                        .to(device), enc_state, enc_state)
                                enc_state = enc_state.to(device)
                            else:
                                output, (enc_state, enc_cell) = model.decoder(
                                    torch.tensor([translated_sequence[-1]]).unsqueeze(0).to(device), enc_state,
                                    enc_state, enc_cell)
                                enc_state = enc_state.to(device)
                        else:
                            # src = src_sequence.unsqueeze(0).long().to(device)
                            translated_sequence_tensor = torch.tensor(translated_sequence).unsqueeze(0).to(device)
                            tgt_mask = gen_nopeek_mask(translated_sequence_tensor.shape[1]).to(device)
                            output = model.decoder(enc_states=enc_state,
                                                   tgt=rearrange(translated_sequence_tensor, 'n s -> s n').long(),
                                                   tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                                   tgt_mask=None)
                        output = output.squeeze(0).to(device)
                        _, indices = torch.topk(output, tokensize)  # sort by Probability
                        indices = indices[-1].to('cpu').numpy()
                        pred_item_sample = []

                        for j in range(len(indices)):
                            if int(indices[j]) in interactionset:
                                pred_item_sample.append(int(indices[j]))
                                if len(pred_item_sample) >= 50:
                                    break
                        if model.encoder_type in ['meanpool', 'maxpool'] and int(indices[0]) == fr_freq_list['[EOS]']:
                            translated_sequence.append(int(indices[1]))
                        else:
                            translated_sequence.append(int(indices[0]))
                        pred_sequence_sample.append(pred_item_sample)

                        i += 1
                    pred_list_sample.append(pred_sequence_sample)  # [sequence, word, pred_token]
                    tgt_list.append(tgt_sequence.tolist()[1:-1])

            pbar.update(1)

    get_scores_sample(epoch, pred_list_sample, tgt_list, tgt_len, decoder_type)

    pbar.close()

def get_scores_sample(epoch, pred_list, tgt_list, tgt_len, decoder_type):
    for k in [5, 10, 20]:
        if decoder_type == 'Point':
            NDCG, MRR, Recall = metric_at_k_set_point(tgt_list, pred_list, k, tgt_len)
        else:
            NDCG, MRR, Recall = metric_at_k_set(tgt_list, pred_list, k, tgt_len)
        post_fix = {
            "Metrics ": 'set sample',
            "Epoch": epoch,
            "R@": k,
            "Recall": Recall,
            "NDCG": NDCG,
            "MRR": MRR

        }
        print(post_fix)

def gen_nopeek_mask(length):
    """
     Returns the nopeek mask
             Parameters:
                     length (int): Number of tokens in each sequence in the target batch
             Returns:
                     mask (arr): tgt_mask, looks like [[0., -inf, -inf],
                                                      [0., 0., -inf],
                                                      [0., 0., 0.]]
     """
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


if __name__ == "__main__":
    main()
