import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import TransformerEncoderLayer, LayerNorm


class EncoderDecoder(nn.Module):
    """
        Encoder-Decoder framework.
        encoder_type: meanpool, maxpool or attention.
        decoder_type: (1) Point;
                      (2) Sequence: LSTM, GRU or attention.
    """

    def __init__(self, encoder, decoder, encoder_type, decoder_type):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

    def forward(self, src, tgt=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_mask=None):

        # Reverse the shape of the batches from (num_sequences, num_tokens_in_each_sequence)
        if tgt is not None:
            tgt = rearrange(tgt, 'n s -> s n').long()
        # Encoder
        if self.encoder_type in ['meanpool', 'maxpool']:
            enc_states = self.encoder(src=src)
        else:  # Transformer Encoder
            enc_states = self.encoder(src=src, src_key_padding_mask=src_key_padding_mask)

        # Decoder
        if self.decoder_type in ['LSTM', 'GRU']:
            enc_states = enc_states.unsqueeze(0).repeat(self.decoder.num_decoder_layers, 1,
                                                        1)  # shape of [layer, batch, hidden]
            if self.decoder_type == 'LSTM':
                # _, (enc_states, enc_cell) = self.encoder(src)
                decoder_state = self.decoder.init_hidden(enc_states)
                outputs, (hidden, cell) = self.decoder(tgt, decoder_state, decoder_state, decoder_state)
                return outputs, (hidden, cell)
            else:
                # _, enc_states = self.encoder(src)
                decoder_state = self.decoder.init_hidden(enc_states)
                outputs, hidden = self.decoder(tgt, decoder_state, decoder_state)
                return outputs, hidden
        elif self.decoder_type == 'Transformer':  # Transformer Decoder
            output = self.decoder.forward(tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask, enc_states=enc_states)
            return output
        else:  # Point Decoder
            output = self.decoder.forward(enc_states)
            return output


class PoolEncoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, is_add_cls, encoder_type):
        """
        Initializes the model
                Parameters:
                        src_vocab_size (int): The amount of tokens in both vocabularies (including start, end, etc tokens)
                        d_model (int): Expected number of features in the encoder/decoder inputs, also used in embeddings
                        encoder_type (string): The type of encoder
                        is_add_cls(bool): The [CLS] position is the embedding of the sequence-level. If the input sequence needs add cls, is_add_cl is True.
        """
        super().__init__()
        self.d_model = d_model
        self.embed_src = nn.Embedding(src_vocab_size, d_model)
        self.is_add_cls = is_add_cls
        self.encoder_type = encoder_type

    def forward(self, src):
        # Embed the batches, scale by sqrt(d_model)
        src_maxlen = len(src[0])
        src = self.embed_src(src) * math.sqrt(self.d_model)  # Embed the batches, scale by sqrt(d_model)
        if self.encoder_type == 'maxpool':
            pool, _ = torch.max(src, dim=1)
        else:  # meanpooling
            pool = torch.mean(src, dim=1)
        enc_states = pool
        if self.is_add_cls:
            return enc_states
        else:
            enc_states = enc_states.unsqueeze(1).repeat(1, src_maxlen, 1)
            return rearrange(enc_states, 'n s e-> s n e')


class RNNEncoder(nn.Module):
    """Container module with a recurrent network encoder."""

    def __init__(self, rnn_type, token_size, d_model, hidden, nlayers, dropout=0.5, tie_weights=False):
        super(RNNEncoder, self).__init__()
        self.token_size = token_size
        self.rnn_type = rnn_type
        self.hidden = hidden
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.emc = nn.Embedding(token_size, d_model)
        # self.init_hidden(self)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(d_model, hidden, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(d_model, hidden, nlayers, nonlinearity=nonlinearity, dropout=dropout)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.emc.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src):
        emb = self.emc(src)
        if self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.rnn(emb)  # output and next hidden
            return output, (hidden, cell)
        else:
            output, hidden = self.rnn(emb)
            return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.hidden),
                    weight.new_zeros(self.nlayers, bsz, self.hidden))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.hidden)


class RNNDecoder(nn.Module):
    """Container module with a recurrent network decoder."""

    def __init__(self, rnn_type, token_size, d_model, hidden, num_decoder_layers, dropout=0.5, tie_weights=True):
        super(RNNDecoder, self).__init__()
        self.token_size = token_size
        self.rnn_type = rnn_type
        self.hidden = hidden
        self.num_decoder_layers = num_decoder_layers
        self.drop = nn.Dropout(dropout)
        self.emc = nn.Embedding(token_size, d_model)
        self.fc = nn.Linear(hidden, token_size)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(d_model + hidden, hidden, num_decoder_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(d_model + hidden, hidden, num_decoder_layers, nonlinearity=nonlinearity, dropout=dropout)

        if tie_weights:
            if hidden != d_model:
                raise ValueError('When using the tied flag, hidden must be equal to emsize')
            self.fc.weight = self.emc.weight

    def forward(self, dec_input, enc_state, state=None, cell=None):
        dec_input = self.emc(dec_input)
        context = enc_state[-1].repeat(dec_input.shape[0], 1, 1)  # last time step hidden
        input_and_context = torch.cat((dec_input, context), 2)
        if self.rnn_type == 'LSTM':
            output, (state, cell) = self.rnn(input_and_context, (state, cell))
        else:
            output, state = self.rnn(input_and_context, state)
        output = self.fc(output)
        output = rearrange(output, 't n e -> n t e')
        if self.rnn_type == 'LSTM':
            return output, (state, cell)
        else:
            return output, state

    def init_hidden(self, enc_states):
        return enc_states


class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, nhead, num_encoder_layers,
                 dim_feedforward, trans_dropout, is_add_cls):
        """
        Initializes the model
                Parameters:
                        src_vocab_size (int): The amount of tokens in both vocabularies (including start, end, etc tokens)
                        d_model (int): Expected number of features in the encoder/decoder inputs, also used in embeddings
                        nhead (int): Number of heads in the transformer
                        num_encoder_layers (int): Number of sub-encoder layers in the transformer
                        dim_feedforward (int): Dimension of the feedforward network in the transformer
                        trans_dropout (float): Dropout value in the transformer
                        is_add_cls(bool): The [CLS] token obtain the embedding of the sequence-level. If the input sequence needs add cls, is_add_cl is True.
        """
        super().__init__()
        self.d_model = d_model
        self.embed_src = nn.Embedding(src_vocab_size, d_model)
        self.is_add_cls = is_add_cls
        self.num_encoder_layers = num_encoder_layers
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=trans_dropout,
                                                norm_first=False)
        encoder_norm = LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, src, src_key_padding_mask=None):
        # Embed the batches, scale by sqrt(d_model)
        src = rearrange(src, 'n s -> s n').long()
        src = self.embed_src(src) * math.sqrt(self.d_model)  # without src pos
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        if self.is_add_cls:
            memory = rearrange(memory, 't n e -> n t e')
            return memory[:, 0, :]
        else:
            return memory


class PointDecoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model):
        """
        Initializes the model
                Parameters:
                        tgt_vocab_size (int): The amount of tokens in both vocabularies (including start, end, etc tokens)
                        d_model (int): Expected number of features in the encoder/decoder inputs, also used in embeddings
        """
        super().__init__()
        self.d_model = d_model
        self.embed_tgt = nn.Embedding(tgt_vocab_size, d_model)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, enc_states):
        # Run the output through a fc layer to return values for each token in the vocab
        fc = self.fc(enc_states)
        return fc


class TransformerDecoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward,
                 max_seq_length, pos_dropout, trans_dropout):
        """
        Initializes the model
                Parameters:
                        tgt_vocab_size (int): The amount of tokens in both vocabularies (including start, end, etc tokens)
                        d_model (int): Expected number of features in the encoder/decoder inputs, also used in embeddings
                        nhead (int): Number of heads in the transformer
                        num_decoder_layers (int): Number of sub-decoder layers in the transformer
                        dim_feedforward (int): Dimension of the feedforward network in the transformer
                        max_seq_length (int): Maximum length of each tokenized sequence
                        pos_dropout (float): Dropout value in the positional encoding
                        trans_dropout (float): Dropout value in the transformer
        """
        super().__init__()
        self.d_model = d_model
        self.embed_tgt = nn.Embedding(tgt_vocab_size, d_model)
        self.num_decoder_layers = num_decoder_layers
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead,
                                                        dropout=trans_dropout)
        self.decoder_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=num_decoder_layers,
                                             norm=self.decoder_norm)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, tgt, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask, enc_states):
        # Embed the batches, scale by sqrt(d_model), and add the positional encoding
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        output = self.decoder.forward(tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask, memory=enc_states)
        # Rearrange to batch-first
        output = rearrange(output, 't n e -> n t e')
        # output = self.dense(output)
        # Run the output through a fc layer to return values for each token in the vocab
        return self.fc(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
