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
    def __init__(self, partitions=[0, 1, 2], data_folder="sp_data/", lg2ind=None, use_glbl_lbls=False):
        # only create dictionaries from sequences to embeddings (as sequence embeddings are already computed by a bert
        # model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_folder = get_data_folder()
        super().__init__()
        seq2emb = {}
        seq2lg = {}
        self.use_lg = lg2ind is not None
        for p in partitions:
            for t in ["test", "train"]:
                part_dict = pickle.load(open(self.data_folder + "sp6_partitioned_data_{}_{}.bin".format(t, p), "rb"))
                seq2emb.update({seq: emb for seq, (emb, _, _, _) in part_dict.items()})
                if lg2ind is not None:
                    seq2lg.update({seq: lg2ind[lg] for seq, (_, _, lg, _) in part_dict.items()})
        self.seq2lg = seq2lg
        self.seq2emb = seq2emb
        # we do need the beginning of sequence and end of sequence tokens for the predictions, so we add a linear layer
        # 2x1024 for that. When using global classification, an additional <CLS> token will be added during training,
        # that will be used for global classification
        self.eos_bos_cls_embs = TokenEmbedding(3, 1024) if use_glbl_lbls else TokenEmbedding(2, 1024)
        self.lg_embs = TokenEmbedding(len(lg2ind.keys()), 1024) if lg2ind is not None else None
        self.use_glbl_lbls = use_glbl_lbls

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_empty_mask(self, shape):
        src_mask = torch.zeros((shape, shape), device=self.device).type(torch.bool)
        return src_mask

    def add_bos_eos_lg_glb_cls_tkns(self, seq):
        aa_embedding = torch.tensor(self.seq2emb[seq], device=self.device)
        bos = self.eos_bos_cls_embs(torch.tensor(0, device=self.device)).reshape(1, -1)
        eos = self.eos_bos_cls_embs(torch.tensor(1, device=self.device)).reshape(1, -1)
        input_tensor = [bos]
        if self.use_glbl_lbls:
            cls_emb = self.eos_bos_cls_embs(torch.tensor(2, device=self.device)).reshape(1,-1)
            input_tensor.append(cls_emb)
        if self.use_lg:
            lg_emb = self.lg_embs(torch.tensor([self.seq2lg[seq]], device=self.device))
            input_tensor.append(lg_emb)
        input_tensor.append(aa_embedding)
        input_tensor.append(eos)
        return torch.cat(input_tensor, dim=0)

    def forward(self, seqs):
        tensor_inputs = [self.add_bos_eos_lg_glb_cls_tkns(s) for s in seqs]
        input_lens = [ti.shape[0] for ti in tensor_inputs]
        # additional inputs BOS tokens, glbl_lbl (<cls> token), and lg
        additional_inp_tkns = 1
        if self.use_lg:
            additional_inp_tkns +=1
        if self.use_glbl_lbls:
            additional_inp_tkns +=1
        output_lens = [ti.shape[0] - additional_inp_tkns for ti in tensor_inputs]
        max_len = max(input_lens)
        max_len_out = max(output_lens)
        padding_mask_src = torch.arange(max_len)[None, :] < torch.tensor(input_lens)[:, None]
        # src and padding are always of the same size
        padding_mask_tgt = torch.arange(max_len_out)[None, :] < torch.tensor(output_lens)[:, None]
        tgt_mask = self.generate_square_subsequent_mask(max_len_out)
        src_mask = self.create_empty_mask(max_len)
        return src_mask, tgt_mask, ~padding_mask_src.to(self.device), ~padding_mask_tgt.to(self.device), \
               tensor_inputs

    def update(self, partitions=[0,1,2]):
        seq2emb = {}
        for p in partitions:
            for t in ["test", "train"]:
                part_dict = pickle.load(open(self.data_folder + "sp6_partitioned_data_{}_{}.bin".format(t, p), "rb"))
                seq2emb.update({seq: emb for seq, (emb, _, _, _) in part_dict.items()})
        self.seq2emb = seq2emb


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5, 
                 partitions=[0, 1], data_folder="sp_data/", lbl2ind={}, lg2ind=None, use_glbl_lbls=False,
                 no_glbl_lbls=6, ff_dim=4096):
        super().__init__()
        self.model_type = 'Transformer'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = Transformer(d_model=d_hid,
                                       nhead=nhead,
                                       num_encoder_layers=nlayers,
                                       num_decoder_layers=nlayers,
                                       dim_feedforward=ff_dim,
                                       dropout=dropout).to(self.device)
        # input_encoder is just a dictionary with {sequence:embedding} with embeddings from bert LM
        self.input_encoder = InputEmbeddingEncoder(partitions=[0, 1, 2], data_folder=data_folder, lg2ind=lg2ind, 
                                                   use_glbl_lbls=use_glbl_lbls)
        # the label encoder is an actualy encoder layer with dim (10 x 1000)
        self.label_encoder = TokenEmbedding(ntoken, d_hid, lbl2ind=lbl2ind)
        self.d_model = d_model
        self.generator = nn.Linear(d_model, ntoken).to(self.device)
        self.use_glbl_lbls = use_glbl_lbls
        if use_glbl_lbls:
            self.glbl_generator = nn.Linear(d_model, no_glbl_lbls).to(self.device)

    def encode(self, src):
        src_mask, tgt_mask, padding_mask_src, padding_mask_tgt, src = self.input_encoder(src)
        if len(src) == 1:
            src = src[0].reshape(1, *src[0].shape).transpose(0,1)
        else:
            src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True).transpose(0,1)

        return self.transformer.encoder(self.pos_encoder(src), src_mask, padding_mask_src)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt = self.pos_encoder(self.label_encoder(tgt).transpose(0,1))
        return self.transformer.decoder(tgt, memory, tgt_mask)

    def forward_glb_lbls(self, src, tgt):
        src_mask, tgt_mask, padding_mask_src, padding_mask_tgt, src = self.input_encoder(src)
        padded_src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True)
        padded_src = self.pos_encoder(padded_src.transpose(0,1))
        padded_tgt = torch.nn.utils.rnn.pad_sequence(self.label_encoder(tgt), batch_first=True).to(self.device)
        padded_tgt = self.pos_encoder(padded_tgt.transpose(0,1))
        # [ FALSE FALSE ... TRUE TRUE FALSE FALSE FALSE ... TRUE TRUE ...]
        memory = self.transformer.encoder(padded_src, src_mask, padding_mask_src)
        outs = self.transformer.decoder(padded_tgt, memory, tgt_mask, tgt_key_padding_mask=padding_mask_tgt)
        return self.generator(outs), self.glbl_generator(memory.transpose(0,1)[:,1,:])

    def forward(self, src: Tensor, tgt: list) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            tgt: list of lists containing sequence labels
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        if self.use_glbl_lbls:
            return self.forward_glb_lbls(src, tgt)
        src_mask, tgt_mask, padding_mask_src, padding_mask_tgt, src = self.input_encoder(src)
        padded_src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True)
        padded_src = self.pos_encoder(padded_src.transpose(0,1))
        padded_tgt = torch.nn.utils.rnn.pad_sequence(self.label_encoder(tgt), batch_first=True).to(self.device)
        padded_tgt = self.pos_encoder(padded_tgt.transpose(0,1))
        # [ FALSE FALSE ... TRUE TRUE FALSE FALSE FALSE ... TRUE TRUE ...]
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
        return x


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
def get_data_folder():
    import os
    if os.path.exists("/scratch/work/dumitra1"):
        return "/scratch/work/dumitra1/sp_data/"
    elif os.path.exists("/home/alex"):
        return "sp_data/"
    else:
        return "/scratch/project2003818/dumitra1/sp_data/"