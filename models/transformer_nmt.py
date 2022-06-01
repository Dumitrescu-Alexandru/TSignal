import os
import pickle
import numpy as np
import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset
from models.binary_sp_classifier import BinarySPClassifier, CNN3


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size, lbl2ind=None):
        """
        Module used in vector representations of the ouput labels or the organism group identifiers

        :param int vocab_size: number of unique tokens (e.g. 11 the output label embedding or 4 for organism group
                representation)
        :param int emb_size: dimensionality of the embedding layer (needs to match the ProtBERT dimension for the
                label encoding
        :param dict lbl2ind: dictionary of labels (output labels or organism group) to indices.
        """
        super(TokenEmbedding, self).__init__()
        self.emb_size = emb_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, emb_size).to(self.device)
        self.lbl2ind = lbl2ind

    def forward(self, tokens: Tensor):
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
    def __init__(self, partitions=[0, 1, 2], og2ind=None, use_glbl_lbls=False, aa2ind=None,
                 glbl_lbl_version=1, tuned_bert_embs_prefix="", tuning_bert=False,
                 residue_emb_extra_dims=0, use_blosum=False, use_extra_oh=False):
        """
        Module that processes the inputs given to TransformermModel (these inputs will be added/concatenate to
        PositionalEncoding outputs).

        :param list partitions: list of SignalP 6.0 partitions needed (when not tuning ProtBERT together with TSignal,
                these embeddings are pre-computed and fixed, and for efficiency they are loaded and used during
                training/testing)
        :param dict og2ind: dictionary of organism groups to index
        :param bool use_glbl_lbls: an extra token <CLS> may be added to the sequence and the SP type be predicted based
                on its representation
        :param dict aa2ind: residue label to index dictionary (used when extra embeddings, along with ProtBERT are used)
        :param int glbl_lbl_version: 1,2,3 or 4. Global label version (specify the way in which SP prediction will be
                conducted)
        :param str tuned_bert_embs_prefix: when tuning ProtBERT separately, the embeddings are again computed offline
                and loaded here from binary files (this specifies the prefix of the file names)
        :param bool tuning_bert: if true, ProtBERT is tuned together with TSignal and <partitions> argument is irrelevant
        :param int residue_emb_extra_dims: number of extra dimensions to use for the decoder's inputs
        :param bool use_blosum: residue_emb_extra_dims is fixed to 16, blosum extra (non-contextual) embeddings are used
        :type bool use_extra_oh: residue_emb_extra_dims is fixed to 32, oh extra (non-contextual) embeddings are used
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_folder = get_data_folder()
        self.use_extra_oh = use_extra_oh
        self.use_blosum = use_blosum
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
        self.og2ind = pickle.load(open("sp6_dicts.bin", "rb"))[1]
        seq2lg = {}
        self.use_lg = og2ind is not None
        if aa2ind is None:
            # if aa2ind is None, train on oh using an InputEmbeddingLayer. A single emb layer having everything (EOS, BOS,
            # CLS, LG) embs should be good
            for p in partitions:
                for t in ["test", "train"]:
                    part_dict = pickle.load(open(self.data_folder + tuned_bert_embs_prefix +
                                                 "sp6_partitioned_data_{}_{}.bin".format(t, p), "rb"))
                    if not tuning_bert:
                        seq2emb.update({seq: emb for seq, (emb, _, _, _) in part_dict.items()})
                    if og2ind is not None:
                        seq2lg.update({seq: og2ind[lg] for seq, (_, _, lg, _) in part_dict.items()})
            self.seq2lg = seq2lg
            self.seq2emb = seq2emb
            self.aa2ind = aa2ind
            # we do need the beginning of sequence and end of sequence tokens for the predictions, so we add a linear layer
            # 2x1024 for that. When using global classification, an additional <CLS> token will be added during training,
            # that will be used for global classification
            self.eos_bos_cls_embs = TokenEmbedding(3, 1024) if use_glbl_lbls else TokenEmbedding(2, 1024)
            self.lg_embs = TokenEmbedding(len(og2ind.keys()), 1024) if og2ind is not None else None
            self.use_glbl_lbls = use_glbl_lbls
        else:
            # add CLS, LG tokens:
            self.use_glbl_lbls = use_glbl_lbls
            self.use_lg = og2ind is not None
            self.aa2ind = aa2ind
            no_of_tokens = len(self.aa2ind.keys())
            self.aa2ind['CLS'], self.aa2ind['LG'] = no_of_tokens, no_of_tokens+1
            self.emb_layer = TokenEmbedding(len(self.aa2ind.items()), 1024)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_empty_mask(self, shape):
        """
        Function used to create esentially a "no-masking" matrix for the inputs (every label has access to all inputs)

        :param int shape: length (should be equal to some encoder layer's representation - coming from ProtBERT)
        """
        src_mask = torch.zeros((shape, shape), device=self.device).type(torch.bool)
        return src_mask

    def add_bos_eos_lg_glb_cls_tkns(self, strseq_or_inpemb, inp_seq=None):
        """
        Method used for:
            - adding {CLS} token in the case of type 1 global label training (predicting SP-type based on a {CLS} token
            - adding {BOS}/{EOS} tokens: as the sequences are of the same length, adding/leving the removed
                shouldn't affect in any way the final performance
            - concatenating blosum/oh/NN(ri) for each ri representation of the ProtBERT model. Used for the non-
                non-contextualized representation "injected" in the TSignal's decoder

        :param str_or_torch.tesnor strseq_or_inpemb: can be a string (when not tuning PortBERT model, the sequence embeddings are
            computed prior to TSignal's training and are simply loaded; as such, a dictionary {seq:embedding} is created
            and used here, and the parameter strseq_or_inpemb would retrieve the corresponding embedding; when seq is a tensor (i.e.
            ProtBERT is tuned along with the TSignal model and the embeddings are computed "online") inp_seqs will
            instead hold the actual sequence.

        :type strseq_or_inpemb:
        :param str_or_None inp_seq:
        :return torch.tensor: tensor containing the input to TSignal's decoder
        """
        aa_embedding = torch.tensor(self.seq2emb[strseq_or_inpemb], device=self.device) if inp_seq is None else strseq_or_inpemb

        # uncomment and use if you with to add to input_tensor the {BOS}/{EOS}; it most likely shouldn't affect the
        # final results
        # bos = self.eos_bos_cls_embs(torch.tensor(0, device=self.device)).reshape(1, -1)
        # eos = self.eos_bos_cls_embs(torch.tensor(1, device=self.device)).reshape(1, -1)
        inp_seq = strseq_or_inpemb if inp_seq is None else inp_seq
        input_tensor = [aa_embedding[:len(inp_seq)]]
        if self.use_glbl_lbls and self.glbl_lbl_version == 1:
            cls_emb = self.eos_bos_cls_embs(torch.tensor(2, device=self.device)).reshape(1,-1)
            input_tensor.append(cls_emb)
        if self.use_lg:
            lg_emb = self.lg_embs(torch.tensor([self.seq2lg[strseq_or_inpemb]], device=self.device)) \
                if inp_seq is None else self.lg_embs(torch.tensor([self.seq2lg[inp_seq]], device=self.device))
            input_tensor.append(lg_emb)
        # input_tensor.append(eos)
        # add a padding/no residue if lg is used at the end of the sequence
        if self.extra_embs_dec_input:
            if self.use_blosum:
                inp_extra_emb = inp_seq if not self.use_lg else inp_seq + "X"
                extra_emb_tensor = torch.cat([ torch.tensor([self.extra_embs_dec_input[r] for r in inp_extra_emb], device=self.device,dtype=torch.float32) ])
                return torch.cat([torch.cat(input_tensor, dim=0), extra_emb_tensor], dim=1)
            elif self.use_extra_oh:
                inp_extra_emb = inp_seq if not self.use_lg else inp_seq + "X"
                extra_emb_tensor = self.make_oh(inp_extra_emb)
                return torch.cat([torch.cat(input_tensor, dim=0), extra_emb_tensor], dim=1)

            else:
                inp_extra_emb = inp_seq if not self.use_lg else inp_seq + "X"
                extra_emb_tensor = self.extra_emb_layer_norm(self.extra_embs_dec_input(torch.tensor([self.aa2ind_extra[r] for r in inp_extra_emb], device=self.device)))
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
            tensor_inputs = [self.add_bos_eos_lg_glb_cls_tkns(s, inp_seq=is_) for (s, is_) in zip(seqs, inp_seqs)]
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
            self.og2ind = pickle.load(open("sp6_dicts.bin", "rb"))[1]
            dict_ = pickle.load(open(self.data_folder + emb_f_name, "rb"))
            seq2emb.update({seq: emb for seq, (emb, _, _, _) in dict_.items()})
            self.seq2lg = {seq: self.og2ind[lg] for seq, (_, _, lg, _) in dict_.items()}
        else:
            for p in partitions:
                for t in ["test", "train"]:
                    part_dict = pickle.load(open(self.data_folder + tuned_bert_embs_prefix +
                                             "sp6_partitioned_data_sublbls_{}_{}.bin".format(t, p), "rb"))
                    seq2emb.update({seq: emb for seq, (emb, _, _, _) in part_dict.items()})
        self.seq2emb = seq2emb



class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5,
                 lbl2ind={}, og2ind=None, use_glbl_lbls=False,
                 no_glbl_lbls=6, ff_dim=4096, aa2ind = None, train_oh=False, glbl_lbl_version=1, form_sp_reg_data=False,
                 version2_agregation="max", input_drop=False, no_pos_enc=False, linear_pos_enc=False, scale_input=False,
                 tuned_bert_embs_prefix="", tuning_bert=False,train_only_decoder=False, add_bert_pe_from_dec_to_bert_out=False,
                 concat_pos_enc=False, pe_extra_dims=64,residue_emb_extra_dims=0, use_blosum=False,use_extra_oh=False,
                 add_extra_embs2_decoder=False):
        """
        Transformer model (TSignal). When tuning ProtBERT together with TSignal model, this model will be the
        classification head of the ProtBERT model. If <train_only_decoder> parameter is set to true, the model
        will only have a Transformer Decoder module, and otherwise it is a full encoder-decoder architecture. When
        <tune_bert> parameter is true, the embeddings given to TransformerModel will be computed (on-line) by ProtBERT.
        Otherwise, for efficiency, pre-computed embeddings are required and will be read from sp_data/ fodler

        :param int ntoken: number of unique tokens (residues)
        :param int d_model: model dimensionality d (old, unused)
        :param int nhead: number of heads per layer
        :param int d_hid: dimensionality of the model d (without extra vectors)
        :param int nlayers: number of layers (both encoder and decoder)
        :param float dropout: dropout at each layer (excluding the Input Embedding)
        :param string data_folder: data path folder
        :param dict lbl2ind: token label dictionary (residues as characters to numbers r -> l)  
        :param dict og2ind: organism group dictionary (residues as characters to numbers og -> l)
        :param bool use_glbl_lbls: specify wether to use global labels asscociated to sequences or not
        :param int no_glbl_lbls: number of global labels (4 in this experiment, eukarya, archaea, gp and gn bacteria)
        :param int ff_dim: expanding dimension of Transformer layers (usually 4x d_hid)
        :param dict aa2ind: amino acid to index dictionary
        :param train_oh: train using one-hot vectors instead of ProtBERT
        :param int glbl_lbl_version: specify the global label approach to be used (1,2 or 3)
        :param bool form_sp_reg_data:
        :param str version2_agregation: old argument. Used when a global label was predicted based on residue embs
        :param float input_drop: amount of dropout from the input embeddings ( drop(E_0) )
        :param bool no_pos_enc: use to not add extra positional encoding added to the decoder
        :param bool linear_pos_enc: use a linear layer for positional encoding in the decoder (instead of fixed, sine-based)
        :param bool scale_input: scale bert embs (e.g. E_30) to have the same norm as the added positional encoding
        :param str tuned_bert_embs_prefix: if tune is tuned separately, load the (fixed) bert embeddings
        :param bool tuning_bert: tune ProtBert together with TSignal
        :param bool train_only_decoder: do not add any extra encoder layers (just nlayers decoder layers)
        :param bool add_bert_pe_from_dec_to_bert_out: initialize linear embedding for the decoder from the ProtBERT
                positional encoding layer
        :param bool concat_pos_enc: at the decoder, concatenate the positional encoding (instead of adding)
        :param int pe_extra_dims: number of extra dimensions to be used with concatenated positional encoding (has to be
                divisible by nhead)
        :param int residue_emb_extra_dims: number of extra dimensions the output layer W_O receives (which receives in
                this case concatenate(D_3[ri], NN(ri)) - i.e. concatenates the final decoder output D_3 and an
                additional, non-contextual representation of the corresponding residue ri from a neural-network.
        :param bool use_blosum: for the additional, non-contextual representation that the last layer W_O receives,
                add 16 fixed dimensions of Blosum62 encoding (trimmed down with PCA to 16, to be divisible by nhead
                we use)
        :param bool use_extra_oh: 32 extra dimensions are added to the last layer W_O which receives one-hot
                representations of the
        :param add_extra_embs2_decoder: the additional one-hot/blosum/NN(ri) (uncontextualized) representations will be
                added to the decoder (note this means predictions will still not have completely uncontextualized
                representations using this)
        """
        super().__init__()
        aa_dict = pickle.load(open("sp6_dicts.bin", "rb"))
        aa_dict = {k: v for k, v in aa_dict[-1].items() if v not in ['ES', 'PD', 'BS']}
        aa_dict['X'] = 20
        self.aa2ind_extra = aa_dict
        self.use_blosum = use_blosum
        self.use_extra_oh = use_extra_oh
        self.add_extra_embs2_decoder = add_extra_embs2_decoder
        if (use_blosum or use_extra_oh) and residue_emb_extra_dims:
            print("!!! WARNING !!!: you have used residue_emb_extra_dims specifying extra dims added to the output layer "
                  "W_O but use_blosum or use_extra_oh fix the extra dimensions to 16/32 respectively")
        if use_extra_oh:
            residue_emb_extra_dims = 32
        elif use_blosum:
            residue_emb_extra_dims = 16
            self.extra_embs_gen_input = pickle.load(open(get_data_folder()+"blosum.bin", "rb"))
        else:
            self.extra_embs_gen_input = None
        self.bert_pe_for_decoder = False
        self.pe_extra_dims = pe_extra_dims if concat_pos_enc else 0
        self.residue_emb_extra_dims = residue_emb_extra_dims

        self.concat_pos_enc = concat_pos_enc
        self.add_bert_pe_from_dec_to_bert_out=add_bert_pe_from_dec_to_bert_out
        self.train_only_decoder = train_only_decoder
        self.add_lg_info = og2ind is not None
        self.form_sp_reg_data = form_sp_reg_data
        self.model_type = 'Transformer'
        self.version2_agregation = version2_agregation
        self.no_pos_enc=no_pos_enc
        self.scale_input = scale_input
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout if input_drop else 0, linear_pos_enc=
                        linear_pos_enc, concat_pos_enc=concat_pos_enc,pe_extra_dims=pe_extra_dims,
                        residue_emb_extra_dims=residue_emb_extra_dims, use_blosum=use_blosum, use_extra_oh=use_extra_oh)
        extra_dims_decoder = residue_emb_extra_dims if add_extra_embs2_decoder else 0
        self.extra_dims_decoder = extra_dims_decoder
        if not train_only_decoder:
            self.transformer = Transformer(d_model=d_hid + self.pe_extra_dims,
                                           nhead=nhead,
                                           num_encoder_layers=nlayers,
                                           num_decoder_layers=nlayers,
                                           dim_feedforward=ff_dim,
                                           dropout=dropout).to(self.device)
        else:
            decoder_layer = TransformerDecoderLayer(d_model = d_hid + self.pe_extra_dims + extra_dims_decoder, nhead=nhead,
                                                    dim_feedforward=ff_dim, dropout=dropout)
            decoder_norm = LayerNorm(d_hid + self.pe_extra_dims + extra_dims_decoder)

            self.transformer = TransformerDecoder(decoder_layer, num_layers=nlayers, norm=decoder_norm).to(self.device)
        # input_encoder is just a dictionary with {sequence:embedding} with embeddings from bert LM
        aa2ind = None if not train_oh else aa2ind
        self.input_encoder = InputEmbeddingEncoder(partitions=[0, 1, 2], og2ind=og2ind,
                                                   use_glbl_lbls=use_glbl_lbls, aa2ind=aa2ind, glbl_lbl_version=glbl_lbl_version,
                                                   tuned_bert_embs_prefix=tuned_bert_embs_prefix, tuning_bert=tuning_bert,
                                                   residue_emb_extra_dims=extra_dims_decoder,
                                                   use_blosum=use_blosum, use_extra_oh=use_extra_oh)
        # the label encoder is an actualy encoder layer with dim (10 x 1000)
        self.label_encoder = TokenEmbedding(ntoken, d_hid + extra_dims_decoder, lbl2ind=lbl2ind)
        self.d_model = d_model
        # self.generator = nn.Sequential(nn.Linear(d_model + pe_extra_dims + residue_emb_extra_dims + self.add_extra_embs2_generator
        #                            if concat_pos_enc else d_model + residue_emb_extra_dims + self.add_extra_embs2_generator, 512).to(self.device),
        #                                    nn.LayerNorm(512).to(self.device), nn.ReLU(),nn.Linear(512, ntoken).to(self.device))
        self.generator = nn.Linear(d_model + self.pe_extra_dims + residue_emb_extra_dims, ntoken).to(self.device)
        self.use_glbl_lbls = use_glbl_lbls
        self.glbl_lbl_version = glbl_lbl_version
        if self.form_sp_reg_data and not use_glbl_lbls:
            self.glbl_generator = nn.Linear(ntoken, no_glbl_lbls).to(self.device)
        elif use_glbl_lbls and self.glbl_lbl_version != 3 and self.glbl_lbl_version != 4:
            self.glbl_generator = nn.Linear(d_model, no_glbl_lbls).to(self.device)
        elif self.use_glbl_lbls and self.glbl_lbl_version == 3 and self.glbl_lbl_version != 4:
            self.glbl_generator = CNN3(input_size=1024, output_size=no_glbl_lbls).to(self.device)
            # self.glbl_generator = BinarySPClassifier(input_size=1024, output_size=no_glbl_lbls).to(self.device)

    def make_oh(self, seq):
        inds = torch.tensor([self.aa2ind_extra[r_] for r_ in seq], device=self.device)
        return torch.nn.functional.one_hot(inds, num_classes=32)

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
        if self.train_only_decoder:
            return self.transformer(tgt, memory, tgt_mask, memory_key_padding_mask=padding_mask_src)
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
            trim_ind_l, trim_ind_r = 2, 1
            src_for_glbl_l = [src[i][trim_ind_l:-trim_ind_r, :] for i in range(len(src))]
        else:
            trim_ind_l, trim_ind_r = 0, 0
            src_for_glbl_l = [src[i] for i in range(len(src))]
        padded_src_glbl = torch.nn.utils.rnn.pad_sequence(src_for_glbl_l, batch_first=True)
        return self.glbl_generator(padded_src_glbl)

    def get_extra_tensor(self, inp_seqs, outs):
        extra_embs = []
        if self.use_blosum:
            for i_s in inp_seqs:
                inp_extra_emb = i_s + "X" * (outs.shape[0] - len(i_s))
                if outs.shape[0] < len(inp_extra_emb):
                    # during inference, not all outs are present at once (so inp_sequence_length > output_sequence_length)
                    inp_extra_emb = inp_extra_emb[:outs.shape[0]]
                extra_emb_tensor = torch.cat([torch.tensor([self.extra_embs_gen_input[r] for r in inp_extra_emb],
                                                           device=self.device, dtype=torch.float32)])
                extra_embs.append(extra_emb_tensor )
        elif self.use_extra_oh:
            for i_s in inp_seqs:
                inp_extra_emb = i_s + "X" * (outs.shape[0] - len(i_s))
                if outs.shape[0] < len(inp_extra_emb):
                    # during inference, not all outs are present at once
                    inp_extra_emb = inp_extra_emb[:outs.shape[0]]
                extra_emb_tensor = self.make_oh(inp_extra_emb)
                extra_embs.append(extra_emb_tensor )
        extra_embs = torch.stack(extra_embs).permute(1,0,2)
        # print(torch.stack(extra_embs).shape, outs.shape)
        # extra_embs = torch.stack(extra_embs).permute(1,2,0)
        return torch.cat([outs, extra_embs], dim=2)

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
        if self.residue_emb_extra_dims:
            outs = self.get_extra_tensor(inp_seqs, outs)
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
                src_for_glbl_l = [src[i][trim_ind_l:-trim_ind_r, :] for i in range(len(src))]
            else:
                trim_ind_l , trim_ind_r = 0, 0
                src_for_glbl_l = [src[i] for  i in range(len(src))]
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
            return self.generator(outs), self.glbl_generator(padded_src_glbl)
        elif self.form_sp_reg_data:
            preds = self.generator(outs)
            return preds, self.glbl_generator(torch.mean(torch.sigmoid(preds.transpose(0,1)), dim=1))
        return self.generator(outs)

        # * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, linear_pos_enc=False, concat_pos_enc=False,pe_extra_dims=64,
                 residue_emb_extra_dims=32, use_blosum=False, use_extra_oh=False):
        """
        Positional encoding module used both for the decoder's inputs, as well as the extra pe. added to the
        ProtBERT embeddings.

        :param int d_model: dimension of the model
        :param float dropout: dropout used for input encoding
        :param int max_len: the sine-based positional encoding (pe. - W_S in manuscript) is pre-computed, and will be
                able to encode positions for sequences of length at most <max_len>
        :param bool linear_pos_enc: if true, instead of sine-based W_S decoder pe., use a linear pe
        :param bool concat_pos_enc: instead of additive pe., concatenate W_S over embedding dimension
        :param int pe_extra_dims: (must be exactly divisible by nhead from TransformerModel); when concatenating the pe.
                specify how many extra dimensions to add to the vectors
        :param int residue_emb_extra_dims: Used only when <concat_pos_enc> is false. When using extra (oh/blosum/linear)
                residue embeddings (at the decoder stage) and <concat_pos_enc> is false (the pe. is additive), the pe.
                dimension needs to be the same as the full model dimension d, and thus pe will have:
                <residue_emb_extra_dims> + d_model
        :param bool use_blosum: <residue_emb_extra_dims> is fixed to 16
        :param bool use_extra_oh: <residue_emb_extra_dims> is fixed to 32
        """
        super().__init__()
        if use_blosum:
            residue_emb_extra_dims = 16
        elif use_blosum:
            residue_emb_extra_dims = 32
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