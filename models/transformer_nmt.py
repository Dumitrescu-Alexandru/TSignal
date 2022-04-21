import os
import pickle
import numpy as np
import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
from torch.utils.data import dataset
from models.binary_sp_classifier import BinarySPClassifier, CNN3


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
    def __init__(self, partitions=[0, 1, 2], data_folder="sp_data/", lg2ind=None, use_glbl_lbls=False, aa2ind=None,
                 glbl_lbl_version=1, form_sp_reg_data=False, tuned_bert_embs_prefix="", tuning_bert=False,
                 residue_emb_extra_dims=0, use_blosum=False, use_extra_oh=False):
        # only create dictionaries from sequences to embeddings (as sequence embeddings are already computed by a bert
        # model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_folder = get_data_folder()
        self.use_extra_oh = use_extra_oh
        self.use_blosum = use_blosum
        residue_emb_extra_dims = 32 if use_blosum else residue_emb_extra_dims
        if tuning_bert and self.data_folder == "sp_data/":
            if not os.path.exists("sp_data/"):
                self.data_folder = "./"
        super().__init__()
        self.residue_emb_extra_dims = residue_emb_extra_dims
        if residue_emb_extra_dims != 0:
            if use_blosum:
                self.extra_embs_dec_input = pickle.load(open("blusum_m.bin", "rb"))
            else:
                self.extra_embs_dec_input = nn.Embedding(20, residue_emb_extra_dims)
                aa_dict = pickle.load(open("sp6_dicts.bin", "rb"))
                aa_dict = {k: v for k, v in aa_dict[-1].items() if v not in ['ES', 'PD', 'BS']}
                aa_dict['X'] = 20
                self.aa2ind_extra = aa_dict
                self.extra_emb_layer_norm = nn.LayerNorm(residue_emb_extra_dims)
        else:
            self.extra_embs_dec_input = None

        self.glbl_lbl_version = glbl_lbl_version
        seq2emb = {}
        self.lg2ind = pickle.load(open("sp6_dicts.bin", "rb"))[1]
        seq2lg = {}
        self.use_lg = lg2ind is not None
        if aa2ind is None:
            # if aa2ind is None, train on oh using an InputEmbeddingLayer. A single emb layer having everything (EOS, BOS,
            # CLS, LG) embs should be good
            for p in partitions:
                for t in ["test", "train"]:
                    part_dict = pickle.load(open(self.data_folder + tuned_bert_embs_prefix +
                                                 "sp6_partitioned_data_sublbls_{}_{}.bin".format(t, p), "rb"))
                    if not tuning_bert:
                        seq2emb.update({seq: emb for seq, (emb, _, _, _) in part_dict.items()})
                    if lg2ind is not None:
                        seq2lg.update({seq: lg2ind[lg] for seq, (_, _, lg, _) in part_dict.items()})
            self.seq2lg = seq2lg
            self.seq2emb = seq2emb
            self.aa2ind = aa2ind
            # we do need the beginning of sequence and end of sequence tokens for the predictions, so we add a linear layer
            # 2x1024 for that. When using global classification, an additional <CLS> token will be added during training,
            # that will be used for global classification
            self.eos_bos_cls_embs = TokenEmbedding(3, 1024) if use_glbl_lbls else TokenEmbedding(2, 1024)
            self.lg_embs = TokenEmbedding(len(lg2ind.keys()), 1024) if lg2ind is not None else None
            self.use_glbl_lbls = use_glbl_lbls
        else:
            # add CLS, LG tokens:
            self.use_glbl_lbls = use_glbl_lbls
            self.use_lg = lg2ind is not None
            self.aa2ind = aa2ind
            no_of_tokens = len(self.aa2ind.keys())
            self.aa2ind['CLS'], self.aa2ind['LG'] = no_of_tokens, no_of_tokens+1
            self.emb_layer = TokenEmbedding(len(self.aa2ind.items()), 1024)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_empty_mask(self, shape):
        src_mask = torch.zeros((shape, shape), device=self.device).type(torch.bool)
        return src_mask

    def add_bos_eos_lg_glb_cls_tkns(self, seq, inp_seqs=None):
        aa_embedding = torch.tensor(self.seq2emb[seq], device=self.device) if inp_seqs is None else seq
        bos = self.eos_bos_cls_embs(torch.tensor(0, device=self.device)).reshape(1, -1)
        eos = self.eos_bos_cls_embs(torch.tensor(1, device=self.device)).reshape(1, -1)
        # input_tensor = [bos]
        input_tensor = [aa_embedding[:len(inp_seqs)]]
        if self.use_glbl_lbls and self.glbl_lbl_version == 1:
            cls_emb = self.eos_bos_cls_embs(torch.tensor(2, device=self.device)).reshape(1,-1)
            input_tensor.append(cls_emb)
        if self.use_lg:
            lg_emb = self.lg_embs(torch.tensor([self.seq2lg[seq]], device=self.device)) if inp_seqs is None else \
                self.lg_embs(torch.tensor([self.seq2lg[inp_seqs]], device=self.device))
            input_tensor.append(lg_emb)
        # input_tensor.append(eos)
        # add a padding/no residue if lg is used at the end of the sequence
        if self.extra_embs_dec_input:
            if self.use_blosum:
                inp_extra_emb = inp_seqs if not self.use_lg else inp_seqs + "X"
                extra_emb_tensor = torch.cat([ torch.tensor([self.extra_embs_dec_input[r] for r in inp_extra_emb], device=self.device,dtype=torch.float32) ])
                # print("mean inp", torch.mean(torch.cat(input_tensor, dim=0), dim=0))
                # print("std inp", torch.mean(torch.std(torch.cat(input_tensor, dim=0), dim=0)))
                # print("mean extra emb", torch.mean(extra_emb_tensor, dim=0))
                # print("std extra emb", torch.mean(torch.std(extra_emb_tensor, dim=0)))
                return torch.cat([torch.cat(input_tensor, dim=0), extra_emb_tensor], dim=1)
            elif self.use_extra_oh:
                inp_extra_emb = inp_seqs if not self.use_lg else inp_seqs + "X"
                extra_emb_tensor = self.make_oh(inp_extra_emb)
                return torch.cat([torch.cat(input_tensor, dim=0), extra_emb_tensor], dim=1)

            else:
                inp_extra_emb = inp_seqs if not self.use_lg else inp_seqs + "X"
                extra_emb_tensor = self.extra_emb_layer_norm(self.extra_embs_dec_input(torch.tensor([self.aa2ind_extra[r] for r in inp_extra_emb], device=self.device)))
                # extra_emb_tensor = self.extra_embs_dec_input(torch.tensor([self.aa2ind_extra[r] for r in inp_extra_emb], device=self.device))
                return torch.cat([torch.cat(input_tensor, dim=0), extra_emb_tensor], dim=1)
        return torch.cat(input_tensor, dim=0)

    def make_oh(self, seq):
        inds = torch.tensor([self.aa2ind_extra[r_] for r_ in seq], device=self.device)
        return torch.nn.functional.one_hot(inds, num_classes=32)

    def oh_forward(self, seqs):
        # add BOS, EOS token and , cls, lg tkns if necessary
        tensor_inputs=[]
        for s in seqs:
            current_s = [self.aa2ind['BS']]
            if self.use_glbl_lbls and self.glbl_lbl_version == 1:
                current_s.append(self.aa2ind['CLS'])
            if self.use_lg:
                current_s.append(self.aa2ind['LG'])
            current_s.extend([self.aa2ind[aa] for aa in s])
            current_s.append(self.aa2ind['ES'])
            tensor_inputs.append(self.emb_layer(torch.tensor(current_s, device=self.device)))
        input_lens = [ti.shape[0] for ti in tensor_inputs]
        # additional inputs BOS tokens, glbl_lbl (<cls> token), and lg
        additional_inp_tkns = 1
        if self.use_lg:
            additional_inp_tkns += 1
        if self.use_glbl_lbls:
            additional_inp_tkns += 1
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

    def forward(self, seqs, inp_seqs=None):
        if self.aa2ind is not None:
            return self.oh_forward(seqs)
        if inp_seqs is not None:
            tensor_inputs = [self.add_bos_eos_lg_glb_cls_tkns(s, inp_seqs=is_) for (s, is_) in zip(seqs, inp_seqs)]
        else:
            tensor_inputs = [self.add_bos_eos_lg_glb_cls_tkns(s) for s in seqs]
        if inp_seqs is None:
            input_lens = [ti.shape[0] for ti in tensor_inputs]
        else:
            input_lens = [len(i) if not self.use_lg else len(i) + 1 for i in inp_seqs]
        # additional inputs BOS tokens, glbl_lbl (<cls> token), and lg
        additional_inp_tkns = 1
        if self.use_lg:
            additional_inp_tkns +=1
        if self.use_glbl_lbls and self.glbl_lbl_version == 1:
            additional_inp_tkns +=1
        output_lens = [ti.shape[0] - additional_inp_tkns + 2 for ti in tensor_inputs]
        max_len = max(input_lens)
        max_len_out = max(output_lens)
        padding_mask_src = torch.arange(max_len)[None, :] < torch.tensor(input_lens)[:, None]
        # src and padding are always of the same size
        padding_mask_tgt = torch.arange(max_len_out)[None, :] < torch.tensor(output_lens)[:, None]
        tgt_mask = self.generate_square_subsequent_mask(max_len_out)
        src_mask = self.create_empty_mask(max_len)
        return src_mask, tgt_mask, ~padding_mask_src.to(self.device), ~padding_mask_tgt.to(self.device), \
               tensor_inputs

    def update(self, partitions=[0,1,2], emb_f_name=None, tuned_bert_embs_prefix=""):
        self.data_folder = get_data_folder()
        seq2emb = {}
        if emb_f_name is not None:
            self.lg2ind = pickle.load(open("sp6_dicts.bin", "rb"))[1]
            dict_ = pickle.load(open(self.data_folder + emb_f_name, "rb"))
            seq2emb.update({seq: emb for seq, (emb, _, _, _) in dict_.items()})
            self.seq2lg = {seq: self.lg2ind[lg] for seq, (_, _, lg, _) in dict_.items()}
        else:
            for p in partitions:
                for t in ["test", "train"]:
                    part_dict = pickle.load(open(self.data_folder + tuned_bert_embs_prefix +
                                             "sp6_partitioned_data_sublbls_{}_{}.bin".format(t, p), "rb"))
                    seq2emb.update({seq: emb for seq, (emb, _, _, _) in part_dict.items()})
        self.seq2emb = seq2emb



class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5,
                 data_folder="sp_data/", lbl2ind={}, lg2ind=None, use_glbl_lbls=False,
                 no_glbl_lbls=6, ff_dim=4096, aa2ind = None, train_oh=False, glbl_lbl_version=1, form_sp_reg_data=False,
                 version2_agregation="max", input_drop=False, no_pos_enc=False, linear_pos_enc=False, scale_input=False,
                 tuned_bert_embs_prefix="", tuning_bert=False,train_only_decoder=False, add_bert_pe_from_dec_to_bert_out=False,
                 concat_pos_enc=False, pe_extra_dims=64,residue_emb_extra_dims=0, use_blosum=False,use_extra_oh=False):
        super().__init__()
        # use_blosum = True
        use_blosum = False
        use_extra_oh = False

        if use_blosum or use_extra_oh:
            residue_emb_extra_dims = 32
        self.use_extra_oh = use_extra_oh
        self.bert_pe_for_decoder = False
        self.pe_extra_dims = pe_extra_dims
        self.residue_emb_extra_dims = residue_emb_extra_dims
        self.concat_pos_enc = concat_pos_enc
        self.add_bert_pe_from_dec_to_bert_out=add_bert_pe_from_dec_to_bert_out
        self.train_only_decoder = train_only_decoder
        self.add_lg_info = lg2ind is not None
        self.form_sp_reg_data = form_sp_reg_data
        self.model_type = 'Transformer'
        self.version2_agregation = "max"
        self.no_pos_enc=no_pos_enc
        self.scale_input = scale_input
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout if input_drop else 0, no_pos_enc=no_pos_enc,
                                              linear_pos_enc=linear_pos_enc, concat_pos_enc=concat_pos_enc,pe_extra_dims=pe_extra_dims,
                                              residue_emb_extra_dims=residue_emb_extra_dims, use_blosum=use_blosum,)
        self.transformer = Transformer(d_model=d_hid + pe_extra_dims  + residue_emb_extra_dims if concat_pos_enc else d_hid + residue_emb_extra_dims,
                                       nhead=nhead,
                                       num_encoder_layers=nlayers,
                                       num_decoder_layers=nlayers,
                                       dim_feedforward=ff_dim,
                                       dropout=dropout).to(self.device)
        # input_encoder is just a dictionary with {sequence:embedding} with embeddings from bert LM
        aa2ind = None if not train_oh else aa2ind
        self.input_encoder = InputEmbeddingEncoder(partitions=[0, 1, 2], data_folder=data_folder, lg2ind=lg2ind,
                                                   use_glbl_lbls=use_glbl_lbls, aa2ind=aa2ind, glbl_lbl_version=glbl_lbl_version,
                                                   form_sp_reg_data=form_sp_reg_data,
                                                   tuned_bert_embs_prefix=tuned_bert_embs_prefix, tuning_bert=tuning_bert,
                                                   residue_emb_extra_dims=residue_emb_extra_dims, use_blosum=use_blosum,
                                                   use_extra_oh=use_extra_oh)
        # the label encoder is an actualy encoder layer with dim (10 x 1000)
        self.label_encoder = TokenEmbedding(ntoken, d_hid + residue_emb_extra_dims, lbl2ind=lbl2ind)
        self.d_model = d_model
        self.generator = nn.Linear(d_model + pe_extra_dims + residue_emb_extra_dims if concat_pos_enc else d_model + residue_emb_extra_dims, ntoken).to(self.device)
        self.use_glbl_lbls = use_glbl_lbls
        self.glbl_lbl_version = glbl_lbl_version
        if self.form_sp_reg_data and not use_glbl_lbls:
            self.glbl_generator = nn.Linear(ntoken, no_glbl_lbls).to(self.device)
        elif use_glbl_lbls and self.glbl_lbl_version != 3 and self.glbl_lbl_version != 4:
            self.glbl_generator = nn.Linear(d_model, no_glbl_lbls).to(self.device)
        elif self.use_glbl_lbls and self.glbl_lbl_version == 3 and self.glbl_lbl_version != 4:
            self.glbl_generator = CNN3(input_size=1024, output_size=no_glbl_lbls).to(self.device)
            # self.glbl_generator = BinarySPClassifier(input_size=1024, output_size=no_glbl_lbls).to(self.device)


    def update_pe(self, pe, freeze_pe):
        # For now, we only add positional encoding to the decoder (not the encoder also), since the encoder (BERT)
        # already adds this same pe in the beginning
        self.no_pos_enc = True
        self.bert_pe_for_decoder = True
        self.pos_encoder.update_pe(pe)
        for name, param in self.pos_encoder.pos_enc.named_parameters():
            param.requires_grad = False

    def encode(self, src, inp_seqs=None):
        if inp_seqs is not None:
            src = [src_[:len(s_)] for src_, s_ in zip(src, inp_seqs)]
        src_mask, tgt_mask, padding_mask_src, padding_mask_tgt, src = self.input_encoder(src, inp_seqs=inp_seqs)
        if len(src) == 1:
            src = src[0].reshape(1, *src[0].shape).transpose(0,1)
        else:
            src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True).transpose(0,1)
        return self.transformer.encoder(self.pos_encoder(src, scale=self.scale_input, no_pos_enc=self.no_pos_enc),
                                        src_mask, padding_mask_src)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, padding_mask_src:Tensor):
        tgt = self.pos_encoder(self.label_encoder(tgt).transpose(0,1))
        return self.transformer.decoder(tgt, memory, tgt_mask, memory_key_padding_mask=padding_mask_src)

    def forward_glb_lbls(self, src, tgt):
        src_mask, tgt_mask, padding_mask_src, padding_mask_tgt, src = self.input_encoder(src)
        padded_src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True)
        padded_src = self.pos_encoder(padded_src.transpose(0,1), scale=self.scale_input)
        padded_tgt = torch.nn.utils.rnn.pad_sequence(self.label_encoder(tgt), batch_first=True).to(self.device)
        padded_tgt = self.pos_encoder(padded_tgt.transpose(0,1))
        # [ FALSE FALSE ... TRUE TRUE FALSE FALSE FALSE ... TRUE TRUE ...]
        memory = self.transformer.encoder(padded_src, src_mask, padding_mask_src)
        outs = self.transformer.decoder(padded_tgt, memory, tgt_mask, tgt_key_padding_mask=padding_mask_tgt)
        return self.generator(outs), self.glbl_generator(memory.transpose(0,1)[:,1,:])

    def get_v3_glbl_lbls(self, src, inp_seqs=None):
        if inp_seqs is not None:
            src = [src_[:len(s_)] for src_, s_ in zip(src, inp_seqs)]
        src_mask, tgt_mask, padding_mask_src, padding_mask_tgt, src = self.input_encoder(src, inp_seqs=inp_seqs)
        if self.add_lg_info:
            trim_ind_l , trim_ind_r = 2, 1
        else:
            trim_ind_l , trim_ind_r = 0, 0
        src_for_glbl_l = [src[i][trim_ind_l:-trim_ind_r, :] for  i in range(len(src))]
        padded_src_glbl = torch.nn.utils.rnn.pad_sequence(src_for_glbl_l, batch_first=True)
        return self.glbl_generator(padded_src_glbl.transpose(2, 1))

    def forward_only_decoder(self, src, tgt, inp_seqs, tgt_mask=None, padding_mask_src=None):
        if tgt_mask is None:
            src_mask, tgt_mask, padding_mask_src, padding_mask_tgt, src = self.input_encoder(src, inp_seqs=inp_seqs)
        else:
            if padding_mask_src is not None:
                src_mask, _, _, padding_mask_tgt, _ = self.input_encoder(src, inp_seqs=inp_seqs)
            else:
                src_mask, _, padding_mask_src, padding_mask_tgt, _ = self.input_encoder(src, inp_seqs=inp_seqs)
        # print(src[0].shape, self.add_lg_info)
        # for ind, is_ in enumerate(inp_seqs):
        #     if len(is_) < 70:
        #         print(len(is_), padding_mask_src[ind],)
        padded_src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True)
        no_pos_enc = False if self.add_bert_pe_from_dec_to_bert_out else self.no_pos_enc
        padded_src = self.pos_encoder(padded_src.transpose(0, 1), scale=self.scale_input, no_pos_enc=no_pos_enc, add_lg_info=self.add_lg_info)
        # [ FALSE FALSE ... TRUE TRUE FALSE FALSE FALSE ... TRUE TRUE ...]
        outs = self.decode(tgt, padded_src, tgt_mask, padding_mask_src)
        return self.generator(outs)

    def forward(self, src: Tensor, tgt: list, inp_seqs=None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            tgt: list of lists containing sequence labels
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        if self.train_only_decoder:
            return self.forward_only_decoder(src, tgt, inp_seqs)
        if self.use_glbl_lbls and self.glbl_lbl_version == 1:
            return self.forward_glb_lbls(src, tgt)
        if inp_seqs is not None:
            src = [src_[:len(s_)] for src_, s_ in zip(src, inp_seqs)]
        src_mask, tgt_mask, padding_mask_src, padding_mask_tgt, src = self.input_encoder(src, inp_seqs=inp_seqs)
        if self.glbl_lbl_version == 3:
            if self.add_lg_info:
                trim_ind_l , trim_ind_r = 2, 1
            else:
                trim_ind_l , trim_ind_r = 0, 0
            src_for_glbl_l = [src[i][trim_ind_l:-trim_ind_r, :] for  i in range(len(src))]
        padded_src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True)
        padded_src = self.pos_encoder(padded_src.transpose(0,1), scale=self.scale_input, no_pos_enc=self.no_pos_enc)
        padded_tgt = torch.nn.utils.rnn.pad_sequence(self.label_encoder(tgt), batch_first=True).to(self.device)
        padded_tgt = self.pos_encoder(padded_tgt.transpose(0,1))
        # [ FALSE FALSE ... TRUE TRUE FALSE FALSE FALSE ... TRUE TRUE ...]
        outs = self.transformer(padded_src, padded_tgt, src_mask, tgt_mask, None, padding_mask_src, padding_mask_tgt,
                                padding_mask_src)
        if self.glbl_lbl_version == 2 and self.use_glbl_lbls:
            if self.version2_agregation == "max":
                return self.generator(outs), self.glbl_generator(torch.max(outs.transpose(0,1), dim=1)[0])
            elif self.version2_agregation == "avg":
                return self.generator(outs), self.glbl_generator(torch.mean(outs.transpose(0,1), dim=1))
        elif self.glbl_lbl_version == 3 and self.use_glbl_lbls:
            padded_src_glbl = torch.nn.utils.rnn.pad_sequence(src_for_glbl_l, batch_first=True)
            return self.generator(outs), self.glbl_generator(padded_src_glbl.transpose(2,1))
        elif self.form_sp_reg_data:
            preds = self.generator(outs)
            return preds, self.glbl_generator(torch.mean(torch.sigmoid(preds.transpose(0,1)), dim=1))
        return self.generator(outs)

        # * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, no_pos_enc=False, linear_pos_enc=False, concat_pos_enc=False,pe_extra_dims=64,
                 residue_emb_extra_dims=32, use_blosum=False):
        super().__init__()
        residue_emb_extra_dims = 32 if use_blosum else residue_emb_extra_dims
        self.concat_pos_enc = concat_pos_enc
        self.pe_extra_dims = pe_extra_dims
        self.linear_pos_enc = linear_pos_enc
        self.dropout = nn.Dropout(p=dropout)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        d_model += residue_emb_extra_dims
        position = torch.arange(max_len).unsqueeze(1).to(self.device)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(self.device) \
            if not concat_pos_enc else torch.exp(torch.arange(0, pe_extra_dims, 2) * (-math.log(10000.0) / d_model)).to(self.device)
        pe = torch.zeros(max_len, 1, d_model).to(self.device) if not concat_pos_enc else torch.zeros(max_len, 1, pe_extra_dims).to(self.device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        if self.linear_pos_enc and not concat_pos_enc:
            self.pos_enc = torch.nn.Embedding(100, 1024).to(self.device)
        if self.linear_pos_enc and concat_pos_enc:
            a = torch.arange(0, pe_extra_dims, 1)
            self.pos_enc = torch.nn.functional.one_hot(a, num_classes=pe_extra_dims).reshape(pe_extra_dims, 1, pe_extra_dims).to(self.device)

    def update_pe(self, pe):
        self.linear_pos_enc = True
        self.pos_enc = torch.nn.Embedding(40000, 1024).to(self.device)
        self.pos_enc.load_state_dict(pe.state_dict())

    def forward(self, x: Tensor, scale=False, no_pos_enc=False, add_lg_info=False) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if self.linear_pos_enc and not no_pos_enc and self.concat_pos_enc:
            pe_ = self.pos_enc[:x.size(0)]
            # print(pe_, pe_.shape,x.shape,pe_.repeat(1,x.size(1),1).shape)
            # print(.shape)
            # exit(1)

            # print(pe_[0],pe_[1],x.shape)
            x = torch.cat([x, pe_.repeat(1, x.size(1), 1)], dim=-1)
            # print(x.shape,x[0,0,1024:],x[1,0,1024:],x[0,1,1024:],x[1,1,1024:])
            # exit(1)

            return x
        if self.linear_pos_enc and not no_pos_enc:
            pos_enc = self.pos_enc(torch.tensor(list(range(x.shape[0]))).to(self.device))
            pos_enc = pos_enc.repeat(1, x.shape[1]).reshape(x.shape[0], x.shape[1], x.shape[2])
            if scale:
                x = self.dropout(x * np.sqrt(1024) + self.pe[:x.size(0)] + pos_enc)
                return x
            else:
                return self.dropout(x + pos_enc)# + self.pe[:x.size(0)])
        if no_pos_enc:
            return self.dropout(x)
        if scale:
            x = self.dropout(x * np.sqrt(1024)+ self.pe[:x.size(0)])
        else:
            if add_lg_info:
                pe_ = self.pe[:x.size(0)-1]
                pe_ = torch.cat([pe_, torch.zeros(1,1,x.shape[-1]).to(self.device)],dim=0)
                x = self.dropout(x + pe_)
            else:
                if self.concat_pos_enc:
                    pe_ = self.pe[:x.size(0)] #* 0.1
                    # print(pe_, pe_.shape,x.shape,pe_.repeat(1,x.size(1),1).shape)
                    # print(.shape)
                    # exit(1)
                    # variance of the pe is about 10 times higher than BERT embs. Maybe getting all variances in line
                    # is good (multiply vectors with 0.1)
                    # print(torch.mean(torch.std(pe_, dim=2)))
                    x = torch.cat([x, pe_.repeat(1,x.size(1),1)],dim=-1)
                else:
                    x = self.dropout(x + self.pe[:x.size(0)])

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