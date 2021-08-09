import pickle
import numpy as np
import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
from torch.utils.data import dataset


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size, lbl2ind=None):
        super(TokenEmbedding, self).__init__()
        self.emb_size = emb_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, emb_size).to(self.device)
        self.lbl2ind = lbl2ind

    def forward(self, tokens: Tensor, no_append_bs=False):
        # no_append_bs is used when testing. Automatic "BS" token appending is done while training
        token_tensors = []
        if self.lbl2ind is not None:
            max_len = max([len(tk_seq) for tk_seq in tokens])

            for tk_seq in tokens:
                tk_tensor = [self.lbl2ind["BS"]]
                tk_tensor.extend(tk_seq)
                tk_tensor.extend([self.lbl2ind["PD"]] * (max_len - len(tk_seq)))
                tk_tensor = torch.tensor(tk_tensor, device=self.device)
                token_tensors.append(tk_tensor)
            tokens = torch.vstack(token_tensors)
            return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class InputEmbeddingEncoder(nn.Module):
    def __init__(self, partitions=[0, 1, 2], data_folder="sp_data/"):
        # only create dictionaries from sequences to embeddings (as sequence embeddings are already computed by a bert
        # model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_folder = data_folder
        super().__init__()
        full_dict = {}
        for p in partitions:
            part_dict = pickle.load(open(self.data_folder + "sp6_partitioned_data_{}_0.bin".format(p), "rb"))
            full_dict.update({seq: emb for seq, (emb, _) in part_dict.items()})
        self.full_dict = full_dict
        # we do need the beginning of sequence and end of sequence tokens for the predictions, so we add a linear layer
        # 2x1024 for that
        self.eos_bos_embs = TokenEmbedding(2, 1024)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_empty_mask(self, shape):
        src_mask = torch.zeros((shape, shape), device=self.device).type(torch.bool)
        return src_mask

    def add_bos_eos_tkns(self, tensor):
        bos = self.eos_bos_embs(torch.tensor(0, device=self.device)).reshape(1, -1)
        eos = self.eos_bos_embs(torch.tensor(1, device=self.device)).reshape(1, -1)
        return torch.cat([bos, tensor, eos], dim=0)

    def forward(self, seqs):
        tensor_inputs = [self.add_bos_eos_tkns(torch.tensor(self.full_dict[s], device=self.device)) for s in seqs]
        input_lens = [ti.shape[0] for ti in tensor_inputs]
        output_lens = [ti.shape[0] - 1 for ti in tensor_inputs]
        max_len = max(input_lens)
        max_len_out = max(output_lens)
        padding_mask_src = torch.arange(max_len)[None, :] < torch.tensor(input_lens)[:, None]
        # src and padding are always of the same size
        padding_mask_tgt = torch.arange(max_len_out)[None, :] < torch.tensor(output_lens)[:, None]
        tgt_mask = self.generate_square_subsequent_mask(max_len_out)
        src_mask = self.create_empty_mask(max_len)
        return src_mask, tgt_mask, ~padding_mask_src.to(self.device), ~padding_mask_tgt.to(self.device), \
               tensor_inputs

    def update(self, partitions=[2]):
        full_dict = {}
        for p in partitions:
            part_dict = pickle.load(open(self.data_folder + "sp6_partitioned_data_{}_0.bin", "rb"))
            full_dict.update({seq: emb for seq, (emb, _) in part_dict.items()})
        self.full_dict = full_dict


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, partitions=[0, 1], data_folder="sp_data/", lbl2ind={}):
        super().__init__()
        self.model_type = 'Transformer'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        self.transformer = Transformer(d_model=d_hid,
                                       nhead=nhead,
                                       num_encoder_layers=nlayers,
                                       num_decoder_layers=nlayers,
                                       dim_feedforward=d_hid * 4,
                                       dropout=dropout).to(self.device)
        # input_encoder is just a dictionary with {sequence:embedding} with embeddings from bert LM
        self.input_encoder = InputEmbeddingEncoder(partitions=[0, 1, 2], data_folder=data_folder)
        # the label encoder is an actualy encoder layer with dim (10 x 1000)
        self.label_encoder = TokenEmbedding(ntoken, d_hid, lbl2ind=lbl2ind)
        self.d_model = d_model
        self.generator = nn.Linear(d_model, ntoken).to(self.device)

    def encode(self, src):
        src_mask, tgt_mask, padding_mask_src, padding_mask_tgt, src = self.input_encoder(src)
        if len(src) == 1:
            src = src[0].reshape(1, *src[0].shape).transpose(0,1)
        return self.transformer.encoder(self.pos_encoder(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt = self.pos_encoder(self.label_encoder(tgt))
        return self.transformer.decoder(tgt.transpose(0,1), memory, tgt_mask)

    def forward(self, src: Tensor, tgt: list) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            tgt: list of lists containing sequence labels
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        src_mask, tgt_mask, padding_mask_src, padding_mask_tgt, src = self.input_encoder(src)
        padded_src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True)
        padded_src = self.pos_encoder(padded_src)
        padded_tgt = torch.nn.utils.rnn.pad_sequence(self.label_encoder(tgt), batch_first=True).to(self.device)
        padded_tgt = self.pos_encoder(padded_tgt)
        padded_src, padded_tgt = padded_src.transpose(0, 1), padded_tgt.transpose(0, 1)
        # print(src_mask.shape)
        outs = self.transformer(padded_src, padded_tgt, src_mask, tgt_mask, None, padding_mask_src, padding_mask_tgt,
                                padding_mask_src)
        return self.generator(outs)

        # * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        position = torch.arange(max_len).unsqueeze(1).to(self.device)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(self.device)
        pe = torch.zeros(max_len, 1, d_model).to(self.device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


# a = generate_square_subsequent_mask(10)
# model = TransformerModel(10, 10, 2, 10, 2, )
#
# a = np.random.randint(0, 9, size=(100,10))
# b = np.random.randint(0, 9, size=(100,10))
#
# UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


#
#
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
#
# src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(a,b)
# print(src_padding_mask[-1], tgt_padding_mask[-1])
# print(a[-1], b[-1])
# print(src_mask, tgt_mask)
