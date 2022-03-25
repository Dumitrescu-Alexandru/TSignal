from copy import deepcopy
from utils.swa_bn_update import update_bn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import logging
import sys

logging.getLogger('some_logger')
from sp_data.bert_tuning_tnmt import ProtBertClassifier, parse_arguments_and_retrieve_logger
import os
import numpy as np
import random
import pickle
import torch.nn as nn
import torch.optim as optim
import torch
sys.path.append(os.path.abspath(".."))
from misc.visualize_cs_pred_results import get_cs_and_sp_pred_results, get_summary_sp_acc, get_summary_cs_acc
from sp_data.data_utils import SPbinaryData, BinarySPDataset, SPCSpredictionData, CSPredsDataset, collate_fn, get_sp_type_loss_weights, get_residue_label_loss_weights
from models.transformer_nmt import TransformerModel

def init_model(ntoken, lbl2ind={}, lg2ind={}, dropout=0.5, use_glbl_lbls=False, no_glbl_lbls=6,
               ff_dim=1024 * 4, nlayers=3, nheads=8, aa2ind={}, train_oh=False, glbl_lbl_version=1,
               form_sp_reg_data=False, version2_agregation="max", input_drop=False, no_pos_enc=False,
               linear_pos_enc=False, scale_input=False, tuned_bert_embs_prefix="", tune_bert=False,train_only_decoder=False):
    model = TransformerModel(ntoken=ntoken, d_model=1024, nhead=nheads, d_hid=1024, nlayers=nlayers,
                             lbl2ind=lbl2ind, lg2ind=lg2ind, dropout=dropout, use_glbl_lbls=use_glbl_lbls,
                             no_glbl_lbls=no_glbl_lbls, ff_dim=ff_dim, aa2ind=aa2ind, train_oh=train_oh,
                             glbl_lbl_version=glbl_lbl_version, form_sp_reg_data=form_sp_reg_data,
                             version2_agregation=version2_agregation, input_drop=input_drop, no_pos_enc=no_pos_enc,
                             linear_pos_enc=linear_pos_enc, scale_input=scale_input, tuned_bert_embs_prefix=tuned_bert_embs_prefix,
                             tuning_bert=tune_bert, train_only_decoder=train_only_decoder)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def get_data_folder():
    if os.path.exists("/scratch/work/dumitra1"):
        return "/scratch/work/dumitra1/sp_data/"
    elif os.path.exists("/home/alex"):
        return "sp_data/"
    else:
        return "/scratch/project2003818/dumitra1/sp_data/"


def padd_add_eos_tkn(lbl_seqs, lbl2ind):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_len = max([len(l) for l in lbl_seqs])
    label_outpus_tensors = []
    for l_s in lbl_seqs:
        tokenized_seq = []
        tokenized_seq.extend(l_s)
        tokenized_seq.append(lbl2ind["ES"])
        tokenized_seq.extend([lbl2ind["PD"]] * (1 + max_len - len(tokenized_seq)))
        label_outpus_tensors.append(torch.tensor(tokenized_seq, device=device))
    return torch.vstack(label_outpus_tensors)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def greedy_decode(model, src, start_symbol, lbl2ind, tgt=None, form_sp_reg_data=False, second_model=None,
                  test_only_cs=False, glbl_lbls=None, tune_bert=False, train_oh=False, saliency_map=False,
                  hook_layer="bert"):
    # model.ProtBertBFD.requires_grad=True
    # onelasttime
    if saliency_map:
        model.requires_grad=True
        model.unfreeze_encoder()
    ind2glbl_lbl = {0: 'NO_SP', 1: 'SP', 2: 'TATLIPO', 3: 'LIPO', 4: 'TAT', 5: 'PILIN'}
    ind2lbl = {v: k for k, v in lbl2ind.items()}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    src = src
    seq_lens = [len(src_) for src_ in src]
    sp_probs = []
    sp_logits = []
    all_seq_sp_probs = []
    all_seq_sp_logits = []
    # used for glbl label version 2
    all_outs = []
    all_outs_2nd_mdl = []
    glbl_labels = None
    retain_grads = []
    sp_predicted_batch_elements = []
    sp_predicted_batch_elements_extracated_cs = []
    sp_pred_inds_CS_spType = []
    if saliency_map:
        def hook_(self, grad_inp, grad_out):
            retain_grads.append((grad_out[0].cpu()))
        # for n, p in model.ProtBertBFD.named_parameters():
        #     print(n)
        # exit(1)
        # model.ProtBertBFD.embeddings.word_embeddings.register_backward_hook(hook_)
        if hook_layer == "bert":
            handle = model.ProtBertBFD.encoder.register_backward_hook(hook_)
        elif hook_layer == "word_embs":
            handle = model.ProtBertBFD.embeddings.word_embeddings.register_backward_hook(hook_)
        elif hook_layer == "full_emb":
            handle = model.ProtBertBFD.embeddings.register_backward_hook(hook_)
        elif hook_layer == "pos_enc":
            handle = model.ProtBertBFD.embeddings.position_embeddings.register_backward_hook(hook_)
        elif hook_layer == "classification_layer":
            handle = model.classification_head.transformer.decoder[-2].register_backward_hook(hook_)
        else:
            handle = model.ProtBertBFD.embeddings.LayerNorm.register_backward_hook(hook_)
        # for n, p in model.ProtBertBFD.named_modules():
        #     print(n)
        # exit(1)
        # model.ProtBertBFD.embeddings.word_embeddings.register_forward_hook(hook_)
        if tune_bert:
            seq_lengths = [len(s) for s in src]
            seqs = [" ".join(r_ for r_ in s) for s in src]
            inputs = model.tokenizer.batch_encode_plus(seqs,
                                                       add_special_tokens=model.hparams.special_tokens,
                                                       padding=True,
                                                       truncation=True,
                                                       max_length=model.hparams.max_length)
            input_ids = torch.tensor(inputs['input_ids'], device=model.device)
            attention_mask = torch.tensor(inputs['attention_mask'], device=model.device)
            memory_bfd = model.ProtBertBFD(input_ids=input_ids, attention_mask=attention_mask)[0]
            if not model.classification_head.train_only_decoder:
                memory = model.classification_head.encode(memory_bfd, inp_seqs=src)
            else:
                _, _, _, _, memory = model.classification_head.input_encoder(memory_bfd, inp_seqs=[s.replace(" ","") for s in seqs])
                memory = torch.nn.utils.rnn.pad_sequence(memory, batch_first=True)
        else:
            memory = model.encode(src)
        if second_model is not None:
            memory_2nd_mdl = model.encode(src)
        else:
            memory_2nd_mdl = None
        if tune_bert and model.classification_head.glbl_lbl_version and model.classification_head.use_glbl_lbls or \
                not tune_bert and model.glbl_lbl_version == 3 and model.use_glbl_lbls:
            # when tuning BERT model + TSignal and having a separate classifier for ths SP type which (should) be based
            # on the embeddings comming from the BERT model
            if test_only_cs:
                batch_size = len(src)
                glbl_labels = torch.zeros(batch_size, 6)
                glbl_labels[list(range(batch_size)), glbl_lbls] = 1
                _, glbl_preds = torch.max(glbl_labels, dim=1)
            else:
                if tune_bert:
                    glbl_labels = model.classification_head.get_v3_glbl_lbls(memory_bfd, inp_seqs=src)
                    _, glbl_preds = torch.max(glbl_labels, dim=1)
                else:
                    glbl_labels = model.get_v3_glbl_lbls(src)
                    _, glbl_preds = torch.max(glbl_labels, dim=1)
    else:
        with torch.no_grad():
            if tune_bert:
                seq_lengths = [len(s) for s in src]
                seqs = [" ".join(r_ for r_ in s) for s in src]
                inputs = model.tokenizer.batch_encode_plus(seqs,
                                                           add_special_tokens=model.hparams.special_tokens,
                                                           padding=True,
                                                           truncation=True,
                                                           max_length=model.hparams.max_length)
                input_ids = torch.tensor(inputs['input_ids'], device=model.device)
                attention_mask = torch.tensor(inputs['attention_mask'], device=model.device)
                memory_bfd = model.ProtBertBFD(input_ids=input_ids, attention_mask=attention_mask)[0]
                seqs = src
                # memory_bfd = model(input_ids=input_ids, attention_mask=attention_mask,
                #                    token_type_ids=inputs['token_type_ids'],return_embeddings=True)[0]
                if not model.classification_head.train_only_decoder:
                    memory = model.classification_head.encode(memory_bfd, inp_seqs=src)
                else:
                    _, _, padding_mask_src, _, memory = model.classification_head.input_encoder(memory_bfd,
                                                                                 inp_seqs=[s.replace(" ", "") for s in
                                                                                           seqs])
                    memory = torch.nn.utils.rnn.pad_sequence(memory, batch_first=True)
            else:
                memory = model.encode(src)
            if second_model is not None:
                memory_2nd_mdl = model.encode(src)
            else:
                memory_2nd_mdl = None
            if tune_bert and model.classification_head.glbl_lbl_version and model.classification_head.use_glbl_lbls or \
                    not tune_bert and model.glbl_lbl_version == 3 and model.use_glbl_lbls:
                # when tuning BERT model + TSignal and having a separate classifier for ths SP type which (should) be based
                # on the embeddings comming from the BERT model
                if test_only_cs:
                    batch_size = len(src)
                    glbl_labels = torch.zeros(batch_size, 6)
                    glbl_labels[list(range(batch_size)), glbl_lbls] = 1
                    _, glbl_preds = torch.max(glbl_labels, dim=1)
                else:
                    if tune_bert:
                        glbl_labels = model.classification_head.get_v3_glbl_lbls(memory_bfd, inp_seqs=src)
                        _, glbl_preds = torch.max(glbl_labels, dim=1)
                    else:
                        glbl_labels = model.get_v3_glbl_lbls(src)
                        _, glbl_preds = torch.max(glbl_labels, dim=1)

    # ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    ys = []

    if not ys:
        for _ in range(len(src)):
            ys.append([])
    start_ind = 0
    # if below condition, then global labels are computed with a separate model. This also affects the first label pred
    # of the model predicting the sequence label. Because of this, compute the first predictions first, and take care
    # of the glbl label model and sequence-model consistency (e.g. one predicts SP other NO-SP - take care of that)
    if glbl_labels is not None:
        tgt_mask = (generate_square_subsequent_mask(1))
        if tune_bert:
            if model.classification_head.train_only_decoder:
                out = model.classification_head.forward_only_decoder(ys, memory.to(device), tgt_mask.to(device), padding_mask_src=padding_mask_src)
            else:
                out = model.classification_head.decode(ys, memory.to(device), tgt_mask.to(device))
        else:
            out = model.decode(ys, memory.to(device), tgt_mask.to(device))

        if tune_bert:
            if model.classification_head.train_only_decoder:
                prob = out
            else:
                out = out.transpose(0, 1)
                prob = model.classification_head.generator(out[:, -1])
        else:
            prob = model.generator(out[:, -1])
        _, next_words = torch.max(prob, dim=1)
        next_word = [nw.item() for nw in next_words]
        current_ys = []
        # ordered indices of no-sp aa labels O, M, I
        ordered_no_sp = [lbl2ind['O'], lbl2ind['M'], lbl2ind['I']]
        for bach_ind in range(len(src)):
            if ind2glbl_lbl[glbl_preds[bach_ind].item()] == "NO_SP" and ind2lbl[next_word[bach_ind]] == "S":
                max_no_sp = np.argmax([prob[bach_ind][lbl2ind['O']].item(), prob[bach_ind][lbl2ind['M']].item(), prob[bach_ind][lbl2ind['I']].item()])
                current_ys.append(ys[bach_ind])
                current_ys[-1].append(ordered_no_sp[max_no_sp])
            elif ind2glbl_lbl[glbl_preds[bach_ind].item()] in ['SP', 'TATLIPO', 'LIPO', 'TAT', 'PILIN'] \
                    and ind2lbl[next_word[bach_ind]] != "S":
                current_ys.append(ys[bach_ind])
                current_ys[-1].append(lbl2ind["S"])
            else:
                current_ys.append(ys[bach_ind])
                current_ys[-1].append(next_word[bach_ind])
        ys = current_ys
        start_ind = 1
    if not tune_bert and not train_oh:
        model.glbl_generator.eval()
    all_probs = []
    for i in range(start_ind, max(seq_lens) + 1):
        if saliency_map:
            tgt_mask = (generate_square_subsequent_mask(len(ys[0]) + 1))
            if second_model is not None:
                out_2nd_mdl = second_model.decode(ys, memory_2nd_mdl.to(device), tgt_mask.to(device))
                out_2nd_mdl = out_2nd_mdl.transpose(0, 1)
                prob_2nd_mdl = second_model.generator(out_2nd_mdl[:, -1])
                all_outs_2nd_mdl.append(out_2nd_mdl[:, -1])
            if tune_bert:
                # for n, p in model.ProtBertBFD.named_parameters():
                #     print(n)
                # exit(1)

                # memory.requires_grad=True
                if model.classification_head.train_only_decoder:
                    prob = model.classification_head.forward_only_decoder(memory.to(device), ys, seqs,
                                                                          tgt_mask.to(device))
                    prob = prob[-1]
                else:
                    out = model.classification_head.decode(ys, memory.to(device), tgt_mask.to(device))
                    out = out.transpose(0, 1)
                    # TODO what if for TAT I use something like p(TAT)/(p(SP)+p(LIPO)+p(PILIN)+p(TAT/SPII))
                    prob = model.classification_head.generator(out[:, -1])
                    all_outs.append(out[:, -1])
                # print("doing the backward pass...")
                for batch_ind in range(prob.shape[0]):
                    max_ind = torch.argmax(prob[batch_ind]).item()
                    if ind2lbl[max_ind] in ["S", "T", "L"] :
                        if i == start_ind:
                            sp_predicted_batch_elements.append(batch_ind)
                            model.zero_grad()
                            model.ProtBertBFD.zero_grad()
                            model.classification_head.zero_grad()
                            # if ind2lbl[max_ind] == 'T':
                            #     prob[batch_ind,max_ind]/(prob[batch_ind,lbl2ind['S']] +
                            #                              prob[batch_ind,lbl2ind['L']] +
                            #                              prob[batch_ind,lbl2ind['W']] +
                            #                              prob[batch_ind,lbl2ind['P']]).backward(retain_graph=True)
                            # else:
                            prob[batch_ind, max_ind].backward(retain_graph=True)
                            sp_pred_inds_CS_spType.append(str(batch_ind) + "_spType")
                    elif ind2lbl[max_ind] not in ["S", "T", "L"] and batch_ind in sp_predicted_batch_elements \
                            and batch_ind not in sp_predicted_batch_elements_extracated_cs:
                        model.zero_grad()
                        model.ProtBertBFD.zero_grad()
                        model.classification_head.zero_grad()
                        prob[batch_ind, max_ind].backward(retain_graph=True)
                        sp_pred_inds_CS_spType.append(str(batch_ind) + "_csPred")
                        sp_predicted_batch_elements_extracated_cs.append(batch_ind)
                # print("did the backward pass")
            else:
                out = model.decode(ys, memory.to(device), tgt_mask.to(device))
                out = out.transpose(0, 1)
                prob = model.generator(out[:, -1])
                all_outs.append(out[:, -1])
        else:
            with torch.no_grad():
                tgt_mask = (generate_square_subsequent_mask(len(ys[0]) + 1))
                if second_model is not None:
                    out_2nd_mdl = second_model.decode(ys, memory_2nd_mdl.to(device), tgt_mask.to(device))
                    out_2nd_mdl = out_2nd_mdl.transpose(0, 1)
                    prob_2nd_mdl = second_model.generator(out_2nd_mdl[:, -1])
                    all_outs_2nd_mdl.append(out_2nd_mdl[:, -1])
                if tune_bert:
                    if model.classification_head.train_only_decoder:
                        prob = model.classification_head.forward_only_decoder(memory.to(device), ys, seqs, tgt_mask.to(device))
                        prob = prob[-1]
                    else:
                        out = model.classification_head.decode(ys, memory.to(device), tgt_mask.to(device))
                        out = out.transpose(0, 1)
                        prob = model.classification_head.generator(out[:, -1])
                        all_outs.append(out[:, -1])
                else:
                    out = model.decode(ys, memory.to(device), tgt_mask.to(device))
                    out = out.transpose(0, 1)
                    prob = model.generator(out[:, -1])
                    all_outs.append(out[:, -1])

        if i == 0 and not form_sp_reg_data:
            # extract the sp-presence probabilities
            sp_probs = [sp_prb.item() for sp_prb in torch.nn.functional.softmax(prob, dim=-1)[:, lbl2ind['S']]]
            all_seq_sp_probs = [[sp_prob.item()] for sp_prob in
                                torch.nn.functional.softmax(prob, dim=-1)[:, lbl2ind['S']]]
            all_seq_sp_logits = [[sp_prob.item()] for sp_prob in prob[:, lbl2ind['S']]]
        elif not form_sp_reg_data:
            # used to update the sequences of probabilities
            softmax_probs = torch.nn.functional.softmax(prob, dim=-1)
            next_sp_probs = softmax_probs[:, lbl2ind['S']]
            next_sp_logits = prob[:, lbl2ind['S']]
            for seq_prb_ind in range(len(all_seq_sp_probs)):
                all_seq_sp_probs[seq_prb_ind].append(next_sp_probs[seq_prb_ind].item())
                all_seq_sp_logits[seq_prb_ind].append(next_sp_logits[seq_prb_ind].item())
        all_probs.append(prob)
        if second_model is not None:
            probs_fm, next_words_fm = torch.max(torch.nn.functional.softmax(prob, dim=-1), dim=1)
            probs_sm, next_words_sm = torch.max(torch.nn.functional.softmax(prob_2nd_mdl, dim=-1), dim=1)
            all_probs_mdls = torch.stack([probs_fm, probs_sm])
            all_next_w_mdls = torch.stack([next_words_fm, next_words_sm])
            if i == 0:
                _, inds = torch.max(all_probs_mdls, dim=0)
            next_words = all_next_w_mdls[inds, torch.tensor(list(range(inds.shape[0])))]
        else:
            _, next_words = torch.max(prob, dim=1)
        next_word = [nw.item() for nw in next_words]
        current_ys = []
        for bach_ind in range(len(src)):
            current_ys.append(ys[bach_ind])
            current_ys[-1].append(next_word[bach_ind])
        ys = current_ys
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if form_sp_reg_data:
        glbl_lbl_version, use_glbl_lbls = (model.glbl_lbl_version, model.use_glbl_lbls) if not tune_bert else \
            (model.classification_head.glbl_lbl_version, model.classification_head.use_glbl_lbls)
        if glbl_lbl_version == 2 and use_glbl_lbls:
            if model.version2_agregation == "max":
                glbl_labels = model.glbl_generator(torch.max(torch.stack(all_outs).transpose(0, 1), dim=1)[0])
                if second_model is not None:
                    glbl_labels = torch.stack([torch.nn.functional.softmax(glbl_labels, dim=1),
                                               torch.nn.functional.softmax(second_model.glbl_generator(
                                                   torch.max(torch.stack(all_outs_2nd_mdl).transpose(0, 1), dim=1)[0]),
                                                   dim=1)])
                    glbl_labels = glbl_labels[inds, torch.tensor(list(range(inds.shape[0]))), :]


            elif model.version2_agregation == "avg":
                glbl_labels = model.glbl_generator(torch.mean(torch.stack(all_outs).transpose(0, 1), dim=1))
                if second_model is not None:
                    glbl_labels = torch.nn.functional.softmax(glbl_labels) + \
                                  torch.nn.functional.softmax(second_model.glbl_generator(
                                      torch.mean(torch.stack(all_outs_2nd_mdl).transpose(0, 1), dim=1)), -1)
        elif glbl_lbl_version == 1 and use_glbl_lbls:
            glbl_labels = model.glbl_generator(memory.transpose(0, 1)[:, 1, :])
            if second_model is not None:
                glbl_labels = torch.nn.functional.softmax(glbl_labels, dim=-1) + \
                              torch.nn.functional.softmax(
                                  second_model.glbl_generator(memory_2nd_mdl.transpose(0, 1)[:, 1, :]), dim=-1)
        elif glbl_lbl_version != 3:
            glbl_labels = model.glbl_generator(torch.mean(torch.sigmoid(torch.stack(all_probs)).transpose(0, 1), dim=1))
            if second_model is not None:
                glbl_labels = torch.nn.functional.softmax(glbl_labels, dim=-1) + \
                              second_model.glbl_generator(
                                  torch.mean(torch.sigmoid(torch.stack(all_probs)).transpose(0, 1), dim=1), dim=-1)
        if saliency_map:
            handle.remove()
            return (ys, torch.stack(all_probs).transpose(0, 1), sp_probs, all_seq_sp_probs, all_seq_sp_logits,
                    glbl_labels), retain_grads, sp_pred_inds_CS_spType
        return ys, torch.stack(all_probs).transpose(0, 1), sp_probs, all_seq_sp_probs, all_seq_sp_logits, \
               glbl_labels
    if saliency_map:
        handle.remove()
        return (ys, torch.stack(all_probs).transpose(0, 1), sp_probs, all_seq_sp_probs, all_seq_sp_logits), retain_grads, sp_pred_inds_CS_spType
    return ys, torch.stack(all_probs).transpose(0, 1), sp_probs, all_seq_sp_probs, all_seq_sp_logits


def beam_decode(model, src, start_symbol, lbl2ind, tgt=None, form_sp_reg_data=False, second_model=None,
                  test_only_cs=False, glbl_lbls=None, tune_bert=False, beam_width=2):
    ind2lbl = {v: k for k, v in lbl2ind.items()}
    ind2glbl_lbl = {0: 'NO_SP', 1: 'SP', 2: 'TATLIPO', 3: 'LIPO', 4: 'TAT', 5: 'PILIN'}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    src = src
    all_outs = []
    model.eval()
    seq_lens = [len(src_) for src_ in src]
    with torch.no_grad():
        if tune_bert:
            seq_lengths = [len(s) for s in src]
            seqs = [" ".join(r_ for r_ in s) for s in src]
            inputs = model.tokenizer.batch_encode_plus(seqs,
                                                       add_special_tokens=model.hparams.special_tokens,
                                                       padding=True,
                                                       truncation=True,
                                                       max_length=model.hparams.max_length)
            input_ids = torch.tensor(inputs['input_ids'], device=model.device)
            attention_mask = torch.tensor(inputs['attention_mask'], device=model.device)
            memory_bfd = model.ProtBertBFD(input_ids=input_ids, attention_mask=attention_mask)[0]
            memory = model.classification_head.encode(memory_bfd, inp_seqs=src)
        else:
            memory = model.encode(src)
        if second_model is not None:
            memory_2nd_mdl = model.encode(src)
        else:
            memory_2nd_mdl = None
        if tune_bert and model.classification_head.glbl_lbl_version and model.classification_head.use_glbl_lbls or \
                not tune_bert and model.glbl_lbl_version == 3 and model.use_glbl_lbls:
            if test_only_cs:
                batch_size = len(src)
                glbl_labels = torch.zeros(batch_size, 6)
                glbl_labels[list(range(batch_size)), glbl_lbls] = 1
                _, glbl_preds = torch.max(glbl_labels, dim=1)
            else:
                if tune_bert:
                    glbl_labels = model.classification_head.get_v3_glbl_lbls(memory_bfd, inp_seqs=src)
                    _, glbl_preds = torch.max(glbl_labels, dim=1)
                else:
                    glbl_labels = model.get_v3_glbl_lbls(src)
                    _, glbl_preds = torch.max(glbl_labels, dim=1)
    # ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    ys = []
    if not ys:
        for _ in range(len(src)):
            ys.append([])
    start_ind = 0
    all_probs, sp_probs, sp_logits = [], [], []
    current_batch_size = len(src)
    log_probs = torch.zeros(current_batch_size, beam_width)
    for i in range(start_ind, max(seq_lens) + 1):
        with torch.no_grad():
            tgt_mask = (generate_square_subsequent_mask(len(ys[0]) + 1))
            # seq len, batch size, emb dimensions
            # 5,5,3 -> 5,15,3 -> 5,3,15 ->
            if i == 1:
                memory = memory.repeat(1, 1, beam_width).reshape(memory.shape[0], current_batch_size * beam_width, -1)
            if tune_bert:
                out = model.classification_head.decode(ys, memory.to(device), tgt_mask.to(device))
                out = out.transpose(0, 1)
                prob = model.classification_head.generator(out[:, -1])
                all_outs.append(out[:, -1])
            else:
                out = model.decode(ys, memory.to(device), tgt_mask.to(device))
                out = out.transpose(0, 1)
                prob = model.generator(out[:, -1])
                all_outs.append(out[:, -1])
        softmax_probs = torch.nn.functional.softmax(prob, dim=-1)
        beam_probs, next_words = torch.topk(torch.nn.functional.softmax(prob, dim=-1), beam_width, dim=1)
        if i == 0:
            # will be used to determine cs prediction probabilities
            # all_probs = beam_probs.reshape(-1)
            # # will be used to visualize and calibrate probabilities of sp-presence predictions
            # sp_probs = [softmax_probs[bi, lbl2ind['S']].item() for bi in range(current_batch_size)]
            # sp_logits = [[prob[bi, bwi].item() for bwi in range(beam_width)] for bi in range(current_batch_size)]
            log_probs = torch.log(beam_probs).reshape(current_batch_size, beam_width)
        else:
            # correctly broadcast (e.g. second element prediction for each sequence)
            # [[s1_pred_1 + s1_pred_1.1, s1_pred_1 + s1_pred_1.2... ] [s1_pred_2 + s1_pred_2.1 s1_pred_2+ s1_pred_2.2 ...], ... ]]
            log_probs = log_probs.reshape(*log_probs.shape, 1)
            log_probs = log_probs + torch.log(beam_probs.reshape(current_batch_size, beam_width, beam_width))
            log_probs = log_probs.reshape(current_batch_size, beam_width * beam_width)
            if i == max(seq_lens):
                log_probs, next_word_seq_col_indices = torch.topk(log_probs, 1, dim=1)
                next_word_col_indices = next_word_seq_col_indices.reshape(-1)
            else:
                log_probs, next_word_seq_col_indices = torch.topk(log_probs, beam_width, dim=1)
                next_word_col_indices = next_word_seq_col_indices.reshape(-1)
                # current_probs = []
                # if 0 < i < max(seq_lens):
                #     for seq_ind in range(len(src)):
                #         for j in range(beam_width):
                #             next_prob_chosen = next_word_col_indices[seq_ind * beam_width + j]
                #             # choose the prev sequence to which this maximal probability corresponds
                #             previous_prob_seq = all_probs[seq_ind * beam_width + next_prob_chosen // beam_width]
                #             chosen_seq_probs = torch.cat([previous_prob_seq.reshape(-1),
                #                                           beam_probs.reshape(-1, beam_width*beam_width)
                #                                           [seq_ind, next_prob_chosen].reshape(-1) ] )
                #             current_probs.append(chosen_seq_probs)
                #     all_probs = torch.vstack(current_probs)
                # elif i == max(seq_lens):
                #     for seq_ind in range(len(src)):
                #         next_prob_chosen = next_word_col_indices[seq_ind]
                #         previous_prob_seq = all_probs[seq_ind * beam_width + next_prob_chosen // beam_width]
                #         chosen_seq_probs = torch.cat([previous_prob_seq.reshape(-1),
                #                                       beam_probs[seq_ind, next_prob_chosen].reshape(-1)])
                #         current_probs.append(chosen_seq_probs)
                #     all_probs = torch.vstack(current_probs).reshape(-1)

        current_ys = []
        if i == 0:
            next_word = [nw.item() for nw in next_words.reshape(-1)]
            for _ in range(len(src) * (beam_width - 1)):
                ys.append([])
            for bach_ind in range(len(src) * beam_width):
                current_ys.append(ys[bach_ind])
                current_ys[-1].append(next_word[bach_ind])
        elif i == max(seq_lens):
            for seq_ind in range(len(src)):
                next_chosen_seq = next_word_col_indices[seq_ind]
                previous_word_seq = ys[seq_ind * beam_width + next_chosen_seq // beam_width].copy()
                current_ys.append(previous_word_seq)
                current_ys[-1].append(next_words[
                                          seq_ind * beam_width + next_chosen_seq // beam_width, next_chosen_seq % beam_width].item())
        else:
            for seq_ind in range(len(src)):
                for j in range(beam_width):
                    next_chosen_seq = next_word_col_indices[seq_ind * beam_width + j]
                    # choose the prev sequence to which this maximal probability corresponds
                    previous_word_seq = ys[seq_ind * beam_width + next_chosen_seq // beam_width].copy()
                    current_ys.append(previous_word_seq)
                    current_ys[-1].append(next_words[
                                              seq_ind * beam_width + next_chosen_seq // beam_width, next_chosen_seq % beam_width].item())
        ys = current_ys

    return ys, torch.tensor([0.1]), sp_probs, torch.tensor([1])


def translate(model: torch.nn.Module, src: str, bos_id, lbl2ind, tgt=None, use_beams_search=False,
              form_sp_reg_data=False, second_model=None, test_only_cs=False, glbl_lbls=None,
              tune_bert=False, train_oh=False):
    model.eval()
    if form_sp_reg_data:
        tgt_tokens, probs, sp_probs, \
        all_sp_probs, all_seq_sp_logits, sp_type_probs = greedy_decode(model, src, start_symbol=bos_id, lbl2ind=lbl2ind,
                                                                       tgt=tgt, form_sp_reg_data=form_sp_reg_data,
                                                                       second_model=second_model, test_only_cs=test_only_cs,
                                                                       glbl_lbls=glbl_lbls, tune_bert=tune_bert, train_oh=train_oh)
        return tgt_tokens, probs, sp_probs, \
               all_sp_probs, all_seq_sp_logits, sp_type_probs
    if use_beams_search:
        tgt_tokens, probs, sp_probs, all_sp_probs = beam_decode(model, src, start_symbol=bos_id, lbl2ind=lbl2ind,
                                           tgt=tgt, form_sp_reg_data=form_sp_reg_data,
                                           second_model=second_model, test_only_cs=test_only_cs,
                                           glbl_lbls=glbl_lbls, tune_bert=tune_bert, beam_width=3)
        all_seq_sp_logits = None, None, None
    else:
        tgt_tokens, probs, sp_probs, all_sp_probs, all_seq_sp_logits = greedy_decode(model, src, start_symbol=bos_id,
                                                                                     lbl2ind=lbl2ind, tgt=tgt, tune_bert=tune_bert,
                                                                                     train_oh=train_oh)
    return tgt_tokens, probs, sp_probs, all_sp_probs, all_seq_sp_logits


def eval_trainlike_loss(model, lbl2ind, run_name="", test_batch_size=50, partitions=[0, 1], sets=["train"],
                        form_sp_reg_data=False, simplified=False, very_simplified=False, tuned_bert_embs_prefix="",
                        tune_bert=False,extended_sublbls=False, random_folds_prefix=""):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=lbl2ind["PD"])
    model.eval()
    sp_data = SPCSpredictionData(form_sp_reg_data=form_sp_reg_data, simplified=simplified, very_simplified=very_simplified,
                                 extended_sublbls=extended_sublbls)
    sp_dataset = CSPredsDataset(sp_data.lbl2ind, partitions=partitions, data_folder=sp_data.data_folder,
                                glbl_lbl_2ind=sp_data.glbl_lbl_2ind, sets=sets, form_sp_reg_data=form_sp_reg_data,
                                tuned_bert_embs_prefix=tuned_bert_embs_prefix, extended_sublbls=extended_sublbls,
                                random_folds_prefix=random_folds_prefix)
    dataset_loader = torch.utils.data.DataLoader(sp_dataset,
                                                 batch_size=test_batch_size, shuffle=False,
                                                 num_workers=4, collate_fn=collate_fn)
    ind2lbl = {v: k for k, v in lbl2ind.items()}
    total_loss = 0
    for ind, (src, tgt, _, _) in enumerate(dataset_loader):
        with torch.no_grad():
            if tune_bert:
                seq_lengths = [len(s) for s in src]
                src = [" ".join(r_ for r_ in s) for s in src]
                inputs = model.tokenizer.batch_encode_plus(src,
                                                           add_special_tokens=model.hparams.special_tokens,
                                                           padding=True,
                                                           truncation=True,
                                                           max_length=model.hparams.max_length)
                inputs['targets'] = tgt
                inputs['seq_lengths'] = seq_lengths
                if model.classification_head.use_glbl_lbls:
                    logits, _ = model(**inputs)
                elif form_sp_reg_data and not extended_sublbls:
                    logits, _ = model(**inputs)
                else:
                    logits = model(**inputs)
            else:
                if model.use_glbl_lbls:
                    logits, _ = model(src, tgt)
                elif form_sp_reg_data:
                    logits, _ = model(src, tgt)
                else:
                    logits = model(src, tgt)
        targets = padd_add_eos_tkn(tgt, sp_data.lbl2ind)
        loss = loss_fn(logits.transpose(0, 1).reshape(-1, logits.shape[-1]), targets.reshape(-1))
        total_loss += loss.item()
        # print("Number of sequences tested: {}".format(ind * test_batch_size))
    return total_loss / len(dataset_loader)

def modify_sp_subregion_preds_and_retrieve_sptype_pred(predicted_lbls):
    glbl_lbl_2ind = {'NO_SP': 0, 'SP': 1, 'TATLIPO': 2, 'LIPO': 3, 'TAT': 4, 'PILIN': 5}
    if predicted_lbls[0] == "S" or predicted_lbls[0] == "P":
        if "RR" in predicted_lbls:
            if "C" in predicted_lbls:
                # not necessarely correct <- should replace C with the I/O/M that comes right after
                return glbl_lbl_2ind["TATLIPO"], predicted_lbls.replace("RR", "SS").replace("C", "I").replace("S", "T")
            else:
                return glbl_lbl_2ind["TAT"], predicted_lbls.replace("RR", "SS").replace("S", "T")
        elif "C" in predicted_lbls:
            # not necessarely correct <- should replace C with the I/O/M that comes right after (same as above)
            return glbl_lbl_2ind["LIPO"], predicted_lbls.replace("C", "I").replace("S", "L")
        elif "P" in predicted_lbls:
            return glbl_lbl_2ind["PILIN"], predicted_lbls
        else:
            return glbl_lbl_2ind["SP"], predicted_lbls

    else:
        return glbl_lbl_2ind["NO_SP"], predicted_lbls

def modify_sp_subregion_preds(pred_tokens, sp_type):
    pred_sptype = torch.argmax(sp_type).item()
    sp_type2inds = {'NO_SP': 0, 'SP': 1, 'TATLIPO': 2, 'LIPO': 3, 'TAT': 4, 'PILIN': 5}
    sp_inds2type = {v: k for k, v in sp_type2inds.items()}
    sptype2letter = {'NO_SP': "W", 'SP': "S", 'TATLIPO': "T", 'LIPO': "L", 'TAT': "T", 'PILIN': "P"}
    if sp_inds2type[pred_sptype] != "NO_SP":
        non_sp_positions = [pred_tokens.find("I"), pred_tokens.find("O"), pred_tokens.find("M")]
        non_sp_positions = [non_sp_position if non_sp_position >= 0 else len(pred_tokens) for non_sp_position in
                            non_sp_positions]
        first_non_sp_pred = min(non_sp_positions)
        # if a +1 cysteine is predicted AFTER the cs, then move behind 1 position
        if pred_tokens[first_non_sp_pred - 1] == "C":
            first_non_sp_pred = first_non_sp_pred - 1
        new_pred_tokens_string = sptype2letter[sp_inds2type[pred_sptype]] * first_non_sp_pred
        new_pred_tokens_string += pred_tokens[first_non_sp_pred:]
    else:
        new_pred_tokens_string = pred_tokens

    return new_pred_tokens_string


def clean_sec_sp2_preds(seq, preds, sp_type, ind2glbl_lbl):
    glbl_lbl2inds = {v:k for k,v in ind2glbl_lbl.items()}
    if ind2glbl_lbl[torch.argmax(sp_type).item()] in ["LIPO", "TATLIPO"]:
        last_l_ind = preds.replace("ES", "W").rfind("S")
        if last_l_ind < len(seq) and seq[last_l_ind + 1] == "C":
            return preds, sp_type
        elif last_l_ind < len(seq) - 3:
            min_i = 10
            for i in range(-2, 3):
                if seq[last_l_ind + i + 1] == "C":
                    if np.abs(i) < np.abs(min_i):
                        best_ind = i
                        min_i = i
            if min_i == 10:
                new_probs = torch.zeros(sp_type.shape[0])
                non_secSP2_inds = [glbl_lbl2inds['SP'], glbl_lbl2inds['TAT'], glbl_lbl2inds['PILIN']]
                non_secSP2_probs = [sp_type[i] for i in [glbl_lbl2inds['SP'], glbl_lbl2inds['TAT'], glbl_lbl2inds['PILIN']]]
                # if the cleavage site of a predicted LIPO/TATLIPO is not found within 3 aas of a cystein amino acid,
                # replace that LIPO/TATLIPO prediction with the highest non-*/SPII SP
                new_probs[non_secSP2_inds[np.argmax(non_secSP2_probs)]] = 1
                return preds, new_probs
            elif min_i > 0:
                return preds[:last_l_ind] + min_i * "S" + preds[last_l_ind + min_i:], sp_type
            elif min_i < 0:
                return preds[:last_l_ind + min_i] + np.abs(min_i) * preds[last_l_ind + 1] + preds[last_l_ind + np.abs(min_i):], sp_type

    return preds, sp_type

def evaluate(model, lbl2ind, run_name="", test_batch_size=50, partitions=[0, 1], sets=["train"], epoch=-1,
             dataset_loader=None,use_beams_search=False, form_sp_reg_data=False, simplified=False, second_model=None,
             very_simplified=False, test_only_cs=False, glbl_lbl_2ind=None, account_lipos=False,
             tuned_bert_embs_prefix="", tune_bert=False, extended_sublbls=False, random_folds_prefix="",
             train_oh=False):
    if glbl_lbl_2ind is not None:
        ind2glbl_lbl = {v:k for k,v in glbl_lbl_2ind.items()}
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=lbl2ind["PD"])
    eval_dict = {}
    sp_type_dict = {}
    seqs2probs = {}
    if "P" in lbl2ind and not extended_sublbls:
        pred_aa_lbl2glbl_ind = {lbl2ind['P']: glbl_lbl_2ind['PILIN'], lbl2ind['S']: glbl_lbl_2ind['SP'], lbl2ind['O']:glbl_lbl_2ind['NO_SP'],
                                lbl2ind['M']:glbl_lbl_2ind['NO_SP'], lbl2ind['I']:glbl_lbl_2ind['NO_SP'], lbl2ind['PD']:glbl_lbl_2ind['NO_SP'],
                                lbl2ind['BS']:glbl_lbl_2ind['NO_SP'], lbl2ind['ES']:glbl_lbl_2ind['NO_SP'], lbl2ind['L']:glbl_lbl_2ind['LIPO'],
                                lbl2ind['T']:glbl_lbl_2ind['TAT'], lbl2ind['W']:glbl_lbl_2ind['TATLIPO']}
    else:
        pred_aa_lbl2glbl_ind = {}
    model.eval()
    if second_model is not None:
        second_model.eval()
    val_or_test = "test" if len(sets) == 2 else "validation"
    if dataset_loader is None:
        sp_data = SPCSpredictionData(form_sp_reg_data=form_sp_reg_data, simplified=simplified,
                                     very_simplified=very_simplified,
                                     extended_sublbls=extended_sublbls)
        sp_dataset = CSPredsDataset(sp_data.lbl2ind, partitions=partitions, data_folder=sp_data.data_folder,
                                    glbl_lbl_2ind=sp_data.glbl_lbl_2ind, sets=sets, form_sp_reg_data=form_sp_reg_data,
                                    tuned_bert_embs_prefix=tuned_bert_embs_prefix, extended_sublbls=extended_sublbls,
                                    random_folds_prefix=random_folds_prefix)

        dataset_loader = torch.utils.data.DataLoader(sp_dataset,
                                                     batch_size=test_batch_size, shuffle=False,
                                                     num_workers=4, collate_fn=collate_fn)

    ind2lbl = {v: k for k, v in lbl2ind.items()}
    total_loss = 0
    for ind, (src, tgt, _, glbl_lbls) in tqdm(enumerate(dataset_loader), "Epoch {} {}".format(epoch, val_or_test),
                                              total=len(dataset_loader)):
        # print("Number of sequences tested: {}".format(ind * test_batch_size))
        src = src
        tgt = tgt
        if form_sp_reg_data and not extended_sublbls:
            predicted_tokens, probs, sp_probs, all_sp_probs, all_seq_sp_logits, sp_type_probs = \
                translate(model, src, lbl2ind['BS'], lbl2ind, tgt=tgt, use_beams_search=use_beams_search,
                          form_sp_reg_data=form_sp_reg_data if not extended_sublbls else False, second_model=second_model,
                          test_only_cs=test_only_cs, glbl_lbls=glbl_lbls, tune_bert=tune_bert, train_oh=train_oh)
        else:
            predicted_tokens, probs, sp_probs, all_sp_probs, all_seq_sp_logits = \
                translate(model, src, lbl2ind['BS'], lbl2ind, tgt=tgt, use_beams_search=use_beams_search,
                          form_sp_reg_data=form_sp_reg_data if not extended_sublbls else False,
                          second_model=second_model, tune_bert=tune_bert, train_oh=train_oh)
            sp_type_probs = [""] * len(predicted_tokens)
        true_targets = padd_add_eos_tkn(tgt, lbl2ind)
        # if not use_beams_search:
        #     total_loss += loss_fn(probs.reshape(-1, 10), true_targets.reshape(-1)).item()
        for s, t, pt, sp_type in zip(src, tgt, predicted_tokens, sp_type_probs):
            predicted_lbls = "".join([ind2lbl[i] for i in pt])
            if account_lipos:
                predicted_lbls, sp_type = clean_sec_sp2_preds(s, predicted_lbls, sp_type, ind2glbl_lbl)
            if type(sp_type) == str:
                # in this case, global label is not used for the model <- glbl label is decided by the first aa label
                # prediction; however, if extended_sublbls is True, the sp-type will be predicted based on the presence
                # of RR/C/P labels
                if extended_sublbls:
                    sp_type_dict[s], predicted_lbls = modify_sp_subregion_preds_and_retrieve_sptype_pred(predicted_lbls)
                else:
                    sp_type_dict[s] = pred_aa_lbl2glbl_ind[pt[0]]
                    # also replace W with T Tat/TATLIPO <- it will already be accounted in the sptype dict
                    predicted_lbls = predicted_lbls.replace("W", "T")
            else:
                sp_type_dict[s] = torch.argmax(sp_type).item()
            if form_sp_reg_data and not extended_sublbls:
                new_predicted_lbls = modify_sp_subregion_preds(predicted_lbls, sp_type)
                predicted_lbls = new_predicted_lbls
            eval_dict[s] = predicted_lbls[:len(t)]
        if sp_probs is not None:
            for s, sp_prob, all_sp_probs, all_sp_logits in zip(src, sp_probs, all_sp_probs, all_seq_sp_logits):
                seqs2probs[s] = (sp_prob, all_sp_probs, all_sp_logits)
    pickle.dump(eval_dict, open(run_name + ".bin", "wb"))
    pickle.dump(sp_type_dict, open(run_name + "_sptype.bin", "wb"))
    if sp_probs is not None and len(sets) > 1:
        # retrieve the dictionary of calibration only for the test set (not for validation) - for now it doesn't
        # make sense to do prob calibration since like 98% of predictions have >0.99 and are correct. See with weight decay
        pickle.dump(seqs2probs, open(run_name + "_sp_probs.bin", "wb"))
    return total_loss / len(dataset_loader)

def load_sptype_model(model_path):
    folder = get_data_folder()
    model = torch.load(folder + model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return model.to(device)

def load_model(model_path, dict_file=None, tuned_bert_embs_prefix="", tune_bert=False, testing=False, opt=False):
    folder = get_data_folder()
    model = torch.load(folder + model_path)
    if not tune_bert:
        model.input_encoder.update(emb_f_name=dict_file,tuned_bert_embs_prefix=tuned_bert_embs_prefix)
    elif tune_bert and testing:
        model.classification_head.input_encoder.update(emb_f_name=dict_file,tuned_bert_embs_prefix=tuned_bert_embs_prefix)
    if opt:
        if os.path.exists(folder + model_path.replace("_best_eval.pth","_best_eval_only_opt_state_dict_cls_head.pth")):
            opt_cls_h_dict = torch.load(folder + model_path.replace("_best_eval.pth","_best_eval_only_opt_state_dict_cls_head.pth"))
            opt_bert_dict = torch.load(folder + model_path.replace("_best_eval.pth","_best_eval_only_opt_state_dict_bert.pth"))
            return model, [opt_cls_h_dict, opt_bert_dict]
        optimizer_ = torch.load(folder + model_path.replace("_best_eval.pth", "_best_eval_only_opt_state_dict.pth"))
        return model, optimizer_
    return model

def save_model(model, model_name="", tuned_bert_embs_prefix="", tune_bert=False, optimizer=None):
    folder = get_data_folder()
    if not tune_bert:
        model.input_encoder.seq2emb = {}
    torch.save(model, folder + model_name + "_best_eval.pth")
    if not tune_bert:
        model.input_encoder.update(tuned_bert_embs_prefix=tuned_bert_embs_prefix)
    if optimizer is not None:
        if type(optimizer) == list:
            torch.save(optimizer[0].state_dict(), folder + model_name + "_best_eval_only_opt_state_dict_cls_head.pth")
            torch.save(optimizer[1].state_dict(), folder + model_name + "_best_eval_only_opt_state_dict_bert.pth")
        else:
            torch.save(optimizer.state_dict(), folder + model_name + "_best_eval_only_opt_state_dict.pth")

def save_sptype_model(model, model_name="", best=False, optimizer=None):
    folder = get_data_folder()
    if os.path.exists(folder + model_name + "_best_sptye_eval.pth"):
        os.remove(folder + model_name + "_best_sptye_eval.pth")
    torch.save(model, folder + model_name + "_best_sptye_eval.pth" if best else folder + model_name + "_current_sptype.pth")


def other_fold_mdl_finished(model_name="", tr_f=0, val_f=1):
    if val_f is None:
        return False
    folder = get_data_folder()
    current_fold_mdl_name = model_name + "_best_eval.pth"
    other_fold_mdl_name = "_".join(current_fold_mdl_name.split("_")[:-6]) + "_t_{}_v_{}".format(val_f,
                                                                                                tr_f) + "_best_eval.pth"
    if os.path.exists(folder + other_fold_mdl_name):
        return other_fold_mdl_name
    return False




def log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="VALIDATION", ep=-1,
                                     beam_txt="Greedy", all_f1_scores=None,sptype_f1=None, only_cs=False):
    print(beam_txt, test_on + "_ONLY_CS" if only_cs else test_on, ep, all_recalls)
    print(
        "{}_{}, epoch {} Mean sp_pred mcc for life groups: {}, {}, {}, {}".format(beam_txt, test_on + "_ONLY_CS" if only_cs else test_on, ep, *sp_pred_mccs))
    print("{}_{}, epoch {} Mean cs recall: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
        beam_txt,
        test_on + "_ONLY_CS" if only_cs else test_on, ep, *all_recalls))
    print("{}_{}, epoch {} Mean cs precision: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
        beam_txt,
        test_on + "_ONLY_CS" if only_cs else test_on, ep, *all_precisions))
    print("{}_{}, epoch {} Mean cs f1-score: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
        beam_txt,
        test_on + "_ONLY_CS" if only_cs else test_on, ep, *np.concatenate(all_f1_scores)))
    print("{}_{}, epoch {} Mean class preds F1Score: {}, {}, {}, {}".format(
        beam_txt,
        test_on + "_ONLY_CS" if only_cs else test_on, ep, *sptype_f1))
    logging.info("{}_{}, epoch {}: Mean sp_pred mcc for life groups: {}, {}, {}, {}".format(beam_txt, test_on + "_ONLY_CS" if only_cs else test_on, ep,
                                                                                            *sp_pred_mccs))
    logging.info(
        "{}_{}, epoch {}: Mean cs recall: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
            beam_txt, test_on + "_ONLY_CS" if only_cs else test_on, ep, *all_recalls))
    logging.info(
        "{}_{}, epoch {}: Mean cs precision: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
            beam_txt, test_on + "_ONLY_CS" if only_cs else test_on, ep, *all_precisions))
    logging.info(
        "{}_{}, epoch {}: Mean cs f1: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
            beam_txt, test_on + "_ONLY_CS" if only_cs else test_on, ep, *np.concatenate(all_f1_scores)))
    logging.info("{}_{}, epoch {} Mean class preds F1Score: {}, {}, {}, {}".format(
        beam_txt,
        test_on + "_ONLY_CS" if only_cs else test_on, ep, *sptype_f1))


def get_lr_scheduler_swa(opt, lr_scheduler_swa=False, lr_sched_warmup=0, use_swa=False):
    def get_schduler_type(op, lr_sched):
        if use_swa:
            return CosineAnnealingWarmRestarts(optimizer=op, T_0=10)
        if lr_sched == "expo":
            return ExponentialLR(optimizer=op, gamma=0.98)
        elif lr_sched == "step":
            return StepLR(optimizer=op, gamma=0.1, step_size=20)
        elif lr_sched == "cos":
            return CosineAnnealingWarmRestarts(optimizer=op)

    if lr_scheduler_swa == "none":
        return None, None
    elif lr_sched_warmup >= 2:
        scheduler = get_schduler_type(opt, lr_scheduler_swa)
        # in lr_sched_warmup epochs, the learning rate will be increased by 10. 1e-5 is the stable lr found. This
        # lr will increase to 1e-4 after lr_sched_warmup steps
        warmup_scheduler = ExponentialLR(opt, gamma=10 ** (1 / lr_sched_warmup))
        return warmup_scheduler, scheduler
    else:
        return None, get_schduler_type(opt, lr_scheduler_swa)


def euk_importance_avg(cs_mcc):
    return (3 / 4) * cs_mcc[0] + (1 / 4) * np.mean(cs_mcc[1:])


def train_cs_predictors(bs=16, eps=20, run_name="", use_lg_info=False, lr=0.0001, dropout=0.5,
                        test_freq=1, use_glbl_lbls=False, partitions=[0, 1], ff_d=4096, nlayers=3, nheads=8,
                        patience=30, train_oh=False, deployment_model=False, lr_scheduler_swa=False, lr_sched_warmup=0,
                        test_beam=False, wd=0., glbl_lbl_weight=1, glbl_lbl_version=1, validate_on_test=False,
                        validate_on_mcc=True, form_sp_reg_data=False, simplified=False, version2_agregation="max",
                        validate_partition=None, very_simplified=False, tune_cs=5, input_drop=False,use_swa=False,
                        separate_save_sptype_preds=False, no_pos_enc=False, linear_pos_enc=False,scale_input=False,
                        test_only_cs=False, weight_class_loss=False, weight_lbl_loss=False, account_lipos=False,
                        tuned_bert_embs=False, warmup_epochs=20, tune_bert=False, frozen_epochs=3, extended_sublbls=False,
                        random_folds=False, train_on_subset=1., train_only_decoder=False, remove_bert_layers=0, augment_trimmed_seqs=False,
                        high_lr=False, cycle_length=5, lr_multiplier_swa=20,change_swa_decoder_optimizer=False,add_val_data_on_swa=False,
                        reinint_swa_decoder=False,anneal_start=-1,anneal_epochs=20,annealed_lr=0.00002):
    if validate_partition is not None:
        test_partition = {0, 1, 2} - {partitions[0], validate_partition}
    else:
        test_partition = set() if deployment_model else {0, 1, 2} - set(partitions)
    partitions = [0, 1, 2] if deployment_model else partitions
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sp_data = SPCSpredictionData(form_sp_reg_data=form_sp_reg_data, simplified=simplified, very_simplified=very_simplified,
                                 extended_sublbls=extended_sublbls)
    train_sets = ['test', 'train'] if validate_on_test or validate_partition is not None else ['train']
    if len(partitions) == 3:
        tuned_bert_embs_prefix = "bert_tuned_{}_{}_{}_".format(*partitions) if tuned_bert_embs else ""
    else:
        tuned_bert_embs_prefix = "bert_tuned_{}_{}_".format(*partitions) if tuned_bert_embs else ""
    random_folds_prefix = "random_folds_" if random_folds else ""
    sp_dataset = CSPredsDataset(sp_data.lbl2ind, partitions=partitions, data_folder=sp_data.data_folder,
                                glbl_lbl_2ind=sp_data.glbl_lbl_2ind, sets=train_sets, form_sp_reg_data=form_sp_reg_data,
                                tuned_bert_embs_prefix=tuned_bert_embs_prefix, extended_sublbls=extended_sublbls,
                                random_folds_prefix=random_folds_prefix, train_on_subset=train_on_subset)
    dataset_loader = torch.utils.data.DataLoader(sp_dataset,
                                                 batch_size=bs, shuffle=True,
                                                 num_workers=4, collate_fn=collate_fn)
    swa_start = 60
    if len(sp_data.lg2ind.keys()) <= 1 or not use_lg_info:
        lg2ind = None
    elif len(sp_data.lg2ind.keys()) > 1 and use_lg_info:
        lg2ind = sp_data.lg2ind
    aa2ind = sp_data.aa2ind if train_oh else None
    if tune_bert:
        hparams, logger = parse_arguments_and_retrieve_logger(save_dir="experiments")
        hparams.train_enc_dec_sp6 = True
        hparams.use_glbl_lbls = use_glbl_lbls
        # form_sp_reg_data=form_sp_reg_data if not extended_sublbls else False
        # the form_sp_reg_data param is used to both denote teh RR/C... usage and usually had a mandatory glbl label
        # in the SP-cs. The current experiment however tests no-glbl-cs tuning
        classification_head = init_model(len(sp_data.lbl2ind.keys()), lbl2ind=sp_data.lbl2ind, lg2ind=lg2ind,
                           dropout=dropout, use_glbl_lbls=use_glbl_lbls, no_glbl_lbls=len(sp_data.glbl_lbl_2ind.keys()),
                           ff_dim=ff_d, nlayers=nlayers, nheads=nheads, train_oh=train_oh, aa2ind=aa2ind,
                           glbl_lbl_version=glbl_lbl_version, form_sp_reg_data=form_sp_reg_data if not extended_sublbls else False,
                           version2_agregation=version2_agregation, input_drop=input_drop, no_pos_enc=no_pos_enc,
                           linear_pos_enc=linear_pos_enc, scale_input=scale_input, tuned_bert_embs_prefix=tuned_bert_embs_prefix,
                                         tune_bert=tune_bert,train_only_decoder=train_only_decoder)
        model = ProtBertClassifier(hparams)
        if remove_bert_layers != 0:
            model.ProtBertBFD.encoder.layer = model.ProtBertBFD.encoder.layer[:-remove_bert_layers]
        model.classification_head = classification_head
        model.to(device)
        if frozen_epochs > 0:
            model.freeze_encoder()
    else:
        model = init_model(len(sp_data.lbl2ind.keys()), lbl2ind=sp_data.lbl2ind, lg2ind=lg2ind,
                           dropout=dropout, use_glbl_lbls=use_glbl_lbls, no_glbl_lbls=len(sp_data.glbl_lbl_2ind.keys()),
                           ff_dim=ff_d, nlayers=nlayers, nheads=nheads, train_oh=train_oh, aa2ind=aa2ind,
                           glbl_lbl_version=glbl_lbl_version, form_sp_reg_data=form_sp_reg_data,
                           version2_agregation=version2_agregation, input_drop=input_drop, no_pos_enc=no_pos_enc,
                           linear_pos_enc=linear_pos_enc, scale_input=scale_input, tuned_bert_embs_prefix=tuned_bert_embs_prefix)
    if weight_class_loss:
        glbl_lbl_2weights = get_sp_type_loss_weights()
        ind2glbl_lbl = {v:k for k,v in sp_data.glbl_lbl_2ind.items()}
        glbl_lbl_weights = [glbl_lbl_2weights[ind2glbl_lbl[i]] for i in range(6)]
        loss_fn_glbl = torch.nn.CrossEntropyLoss(weight=torch.tensor(glbl_lbl_weights, device=device))
    else:
        loss_fn_glbl = torch.nn.CrossEntropyLoss()

    if weight_lbl_loss:
        ind2lbl = {v:k for k,v in sp_data.lbl2ind.items()}
        lbl2weights = get_residue_label_loss_weights()
        label_weights = [lbl2weights[ind2lbl[i]] for i in range(len(sp_data.lbl2ind.values()))]
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(label_weights, device=device), ignore_index=sp_data.lbl2ind["PD"])
    else:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=sp_data.lbl2ind["PD"])

    loss_fn_tune = torch.nn.CrossEntropyLoss(ignore_index=sp_data.lbl2ind["PD"], reduction='none')


    if tune_bert:
        if use_swa:
            # in this case, we will use a cyclic scheduler on the best model found based on early stopping. few problems:
            # 1. the BERT lr is much more sensitive than classification_head optimizer
            # 2. We need to load the optimizer(best_model) found in early stopping to resume training
            # 3. when adding the scheduler, we want to increase (cyclically) the lr only for classification_head parameters; That we
            # cannot do with a scheduler on an optimizer on the full model; therefore we need to use 2 optimizers and
            # attach a schduler only on classification_head_optimizer
            classification_head_optimizer = optim.Adam(model.classification_head.parameters(),  lr=lr * 10 if high_lr
                                                                else lr,  eps=1e-9, weight_decay=wd, betas=(0.9, 0.98),)
            bert_optimizer = optim.Adam(model.ProtBertBFD.parameters(),  lr=0.00001,  eps=1e-9, weight_decay=wd, betas=(0.9, 0.98),)
            optimizer = [classification_head_optimizer, bert_optimizer]
        else:
            # BERT model always has this LR, as any higher lr worsens the results
            parameters = [
                {"params": model.classification_head.parameters()},
                {
                    "params": model.ProtBertBFD.parameters(),
                    "lr": 0.00001,
                },
            ]
            # optimizer = Lamb(parameters, lr=self.hparams.learning_rate, weight_decay=0.01)
            optimizer = optim.Adam(parameters,  lr=lr * 10 if high_lr else lr,  eps=1e-9, weight_decay=wd, betas=(0.9, 0.98),)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=wd)
    if anneal_start != -1:
        if type(optimizer) != list:
            print("WARNING!!!: You tried to use annealing lr on both BERT and class_head; Only the classification_head "
                         "may be annealed, as BERT parameters are much more sensitive. Check implementation; Currently this only works with SWA (--use_swa)")
            logging.info("WARNING!!!: You tried to use annealing lr on both BERT and class_head; Only the classification_head "
                         "may be annealed, as BERT parameters are much more sensitive. Check implementation; Currently this only works with SWA (--use_swa)")
            anneal_scheduler = None
        else:
            anneal_scheduler = SWALR(optimizer[0], swa_lr=annealed_lr, anneal_epochs=anneal_epochs, anneal_strategy='linear')
            print("Using annealing LR ({}-{}) from epochs ({}-{})".format(lr, annealed_lr, anneal_start,anneal_epochs+anneal_start))
            logging.info("Using annealing LR ({}-{}) from epochs ({}-{})".format(lr, annealed_lr, anneal_start,anneal_epochs+anneal_start))
    else:
        anneal_scheduler = None
    # if not use_swa and lr_scheduler_swa != "none":
    #     warmup_scheduler, scheduler = get_lr_scheduler_swa(optimizer, lr_scheduler_swa, lr_sched_warmup, use_swa)
    # else:
    warmup_scheduler = None

    best_valid_loss = 5 ** 10
    best_valid_mcc_and_recall = -1
    best_epoch = 0
    bestf1_sp_type = 0
    current_sptype_f1 = 0
    warmup_epochs = warmup_epochs if validate_on_mcc else 0
    e = -1
    iter_no = 0
    while patience != 0:
        if type(optimizer) == list:
            print("LR:", optimizer[0].param_groups[0]['lr'], "LR_bert:", optimizer[1].param_groups[0]['lr'])
        if tune_bert and frozen_epochs > e:
            model.eval()
            model.classification_head.train()
        elif tune_bert and frozen_epochs == e:
            model.unfreeze_encoder()
            model.train()
        elif frozen_epochs > e:
            model.train()
        else:
            model.train()
        e += 1
        losses = 0
        losses_glbl = 0
        if anneal_scheduler is not None and anneal_start < e <= swa_start:
            anneal_scheduler.step()
        for ind, batch in tqdm(enumerate(dataset_loader), "Epoch {} train:".format(e), total=len(dataset_loader)):
            if lr_scheduler_swa != "none" and e >= swa_start:
                scheduler.step(iter_no % cycle_length)
                if iter_no % (cycle_length+1) == 1:
                    # if the last training step was the one corresponding to training with the lowest lr, update swa
                    swa_model.to("cuda:0")
                    swa_model.update_parameters(model)
                    swa_model.to("cpu")
                iter_no += 1
            seqs, lbl_seqs, _, glbl_lbls = batch
            # if augment_trimmed_seqs:
            cuts = np.random.randint(0,10,len(seqs))
            if augment_trimmed_seqs:
                cut_seqs = [s[:-cuts[cut_ind]]  if cut_ind != 0 else s for cut_ind, s in enumerate(seqs)]
                lbl_seqs = [l[:-cuts[cut_ind]] if cut_ind != 0  else l for cut_ind, l in enumerate(lbl_seqs)]
            else:
                cut_seqs = None
            if use_glbl_lbls:
                if tune_bert:
                    seq_lengths = [len(s) for s in seqs]
                    seqs = [" ".join(r_ for r_ in s) for s in seqs]
                    inputs = model.tokenizer.batch_encode_plus(seqs,
                                                               add_special_tokens=model.hparams.special_tokens,
                                                               padding=True,
                                                               truncation=True,
                                                               max_length=model.hparams.max_length)
                    inputs['targets'] = lbl_seqs
                    inputs['seq_lengths'] = seq_lengths
                    logits, glbl_logits = model(**inputs)
                else:
                    logits, glbl_logits = model(seqs, lbl_seqs)
                if type(optimizer) == list:
                    # for separate classification_head/BERT optimizers
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()
                targets = padd_add_eos_tkn(lbl_seqs, sp_data.lbl2ind)
                if tune_cs > 0 and e > 35:
                    seq_indices = []
                    seq_dim = logits.shape[0]
                    for ind_, (gl, t) in enumerate(zip(glbl_lbls, targets)):
                        if gl != 0:
                            t = list(t.cpu().numpy())
                            t.reverse()
                            last_sp_ind = len(t) - t.index(0) - 1
                            sp_inds = list(range(last_sp_ind - tune_cs, last_sp_ind + tune_cs))
                            seq_indices.extend([si + seq_dim * ind_ for si in sp_inds])

                    loss = loss_fn_tune(logits.transpose(0, 1).reshape(-1, logits.shape[-1]), targets.reshape(-1))
                    loss *= 0.1
                    loss[seq_indices] *= 10
                    loss = torch.mean(loss)
                    losses += loss.item()

                    loss_glbl = loss_fn_glbl(glbl_logits, torch.tensor(glbl_lbls, device=device))
                    losses_glbl += loss_glbl.item()
                    loss += loss_glbl * glbl_lbl_weight
                else:
                    # if "MKIFFAVLVILVLFSMLIWTAYGTPYPVNCKTDRDCVMCGLGISCKNGYCQGCTR" in seqs:
                    #     datruind = -1
                    #     for ind__, s in enumerate(seqs):
                    #         if s == "MKIFFAVLVILVLFSMLIWTAYGTPYPVNCKTDRDCVMCGLGISCKNGYCQGCTR":
                    #             datruind = ind__
                    #     print(lbl_seqs[datruind])
                    #     print(targets[datruind])
                    loss = loss_fn(logits.transpose(0, 1).reshape(-1, logits.shape[-1]), targets.reshape(-1))
                    losses += loss.item()
                    loss_glbl = loss_fn_glbl(glbl_logits, torch.tensor(glbl_lbls, device=device))
                    losses_glbl += loss_glbl.item()
                    loss += loss_glbl * glbl_lbl_weight
            elif form_sp_reg_data and not extended_sublbls:
                logits, glbl_logits = model(seqs, lbl_seqs)
                if type(optimizer) == "list":
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()
                targets = padd_add_eos_tkn(lbl_seqs, sp_data.lbl2ind)
                loss = loss_fn(logits.transpose(0, 1).reshape(-1, logits.shape[-1]), targets.reshape(-1))
                losses += loss.item()

                loss_glbl = loss_fn_glbl(glbl_logits, torch.tensor(glbl_lbls, device=device))
                losses_glbl += loss_glbl.item()
                loss += loss_glbl * glbl_lbl_weight
            else:
                if tune_bert:
                    seq_lengths = [len(s) for s in seqs]
                    seqs = [" ".join(r_ for r_ in s) for s in seqs]
                    cut_seqs = [" ".join(r_ for r_ in s) for s in cut_seqs] if cut_seqs is not None else None
                    inputs = model.tokenizer.batch_encode_plus(cut_seqs if augment_trimmed_seqs else seqs,
                                                               add_special_tokens=model.hparams.special_tokens,
                                                               padding=True,
                                                               truncation=True,
                                                               max_length=model.hparams.max_length)
                    inputs['targets'] = lbl_seqs
                    inputs['seq_lengths'] = seq_lengths
                    if  augment_trimmed_seqs:
                        inputs['sequences'] = seqs
                    logits = model(**inputs)
                else:
                    logits = model(seqs, lbl_seqs)
                if type(optimizer) == list:
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()
                targets = padd_add_eos_tkn(lbl_seqs, sp_data.lbl2ind)
                loss = loss_fn(logits.transpose(0, 1).reshape(-1, logits.shape[-1]), targets.reshape(-1))
                losses += loss.item()

            loss.backward()
            if type(optimizer) == list:
                optimizer[0].step()
                optimizer[1].step()
            else:
                optimizer.step()
        if use_swa and e >= swa_start and lr_scheduler_swa == "none":
            swa_model.to("cuda:0")
            swa_model.update_parameters(model)
            swa_model.to("cpu")

        if validate_on_test:
            validate_partitions = list(test_partition)
            _ = evaluate(swa_model.module.to(device) if use_swa and e + 1>= swa_start else model , sp_data.lbl2ind, run_name=run_name, partitions=validate_partitions, sets=["test"],
                         epoch=e, form_sp_reg_data=form_sp_reg_data, simplified=simplified,very_simplified=very_simplified, glbl_lbl_2ind=sp_data.glbl_lbl_2ind,
                         tuned_bert_embs_prefix=tuned_bert_embs_prefix, extended_sublbls=extended_sublbls, random_folds_prefix=random_folds_prefix, train_oh=train_oh)
            sp_pred_mccs, sp_pred_mccs2, lipo_pred_mccs, lipo_pred_mccs2, tat_pred_mccs, tat_pred_mccs2, \
            all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat, all_f1_scores_lipo, all_f1_scores_tat, \
            all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores, sptype_f1 = \
                get_cs_and_sp_pred_results(filename=run_name + ".bin", v=False, return_everything=True, return_class_prec_rec=True)
            valid_loss = eval_trainlike_loss(model, sp_data.lbl2ind, run_name=run_name, partitions=validate_partitions,
                                             sets=["test"], form_sp_reg_data=form_sp_reg_data, simplified=simplified,
                                             very_simplified=very_simplified, tune_bert=tune_bert, extended_sublbls=extended_sublbls,
                                             random_folds_prefix=random_folds_prefix)
            # revert valid_loss to not change the loss condition next ( this won't be a loss
            # but it's the quickest way to test performance when validation with the test set
        else:
            valid_sets = ["train", "test"] if validate_partition is not None else ["test"]
            validate_partitions = [validate_partition] if validate_partition is not None else partitions
            valid_loss = eval_trainlike_loss(model, sp_data.lbl2ind, run_name=run_name, partitions=validate_partitions,
                                             sets=valid_sets, form_sp_reg_data=form_sp_reg_data, simplified=simplified,
                                             very_simplified=very_simplified, tune_bert=tune_bert, extended_sublbls=extended_sublbls,
                                             random_folds_prefix=random_folds_prefix)
            _ = evaluate(swa_model.module.to(device) if use_swa and e >= swa_start else model, sp_data.lbl2ind, run_name=run_name,
                         partitions=validate_partitions, sets=valid_sets, epoch=e, form_sp_reg_data=form_sp_reg_data,
                         simplified=simplified, very_simplified=very_simplified, glbl_lbl_2ind=sp_data.glbl_lbl_2ind,
                         tuned_bert_embs_prefix=tuned_bert_embs_prefix, tune_bert=tune_bert, extended_sublbls=extended_sublbls,
                         random_folds_prefix=random_folds_prefix, train_oh=train_oh)
            sp_pred_mccs, sp_pred_mccs2, lipo_pred_mccs, lipo_pred_mccs2, tat_pred_mccs, tat_pred_mccs2, \
            all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat, all_f1_scores_lipo, all_f1_scores_tat, \
            all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores, sptype_f1 = \
                get_cs_and_sp_pred_results(filename=run_name + ".bin", v=False, return_everything=True, return_class_prec_rec=True)

        if validate_on_mcc:
            no_of_seqs_sp1 = list(np.array([2040, 44, 142, 356]).repeat(4))
            no_of_seqs_sp2 = list(np.array([1087, 516, 12]).repeat(4))
            no_of_seqs_tat = list(np.array([313, 39, 13]).repeat(4))
            no_of_tested_sp_seqs = sum([2040, 44, 142, 356]) + sum([1087, 516, 12]) + sum([313, 39, 13])
            summary = np.nansum(
                (np.array(all_f1_scores).reshape(-1) * np.array(no_of_seqs_sp1))) / no_of_tested_sp_seqs + \
                      np.nansum((np.array(all_f1_scores_lipo).reshape(-1) * np.array(
                          no_of_seqs_sp2))) / no_of_tested_sp_seqs + \
                      np.nansum(
                          (np.array(all_f1_scores_tat).reshape(-1) * np.array(no_of_seqs_tat))) / no_of_tested_sp_seqs
            # print(sp_pred_mccs, sp_pred_mccs2)
            # print(all_f1_scores, all_f1_scores_tat, all_f1_scores_lipo)
            # print(summary)
            # patiente_metric = np.mean(sp_pred_mccs2)
            patiente_metric = summary
            if separate_save_sptype_preds:
                current_sptype_f1 = sptype_f1[0] * 0.5 + sptype_f1[1] * 0.2 + sptype_f1[2] * 0.2  + sptype_f1[3] * 0.1
        else:
            patiente_metric = valid_loss
            if separate_save_sptype_preds:
                current_sptype_f1 = sptype_f1[0] * 0.5 + sptype_f1[1] * 0.2 + sptype_f1[2] * 0.2  + sptype_f1[3] * 0.1
        # sp_pred_mccs
        if use_glbl_lbls or form_sp_reg_data:
            print("On epoch {} total train/validation loss and glbl loss: {}/{}, {}".format(e, losses / len(
                dataset_loader),
                                                                                            valid_loss,
                                                                                            losses_glbl / len(
                                                                                                dataset_loader)))
            logging.info("On epoch {} total train/validation loss and glbl loss: {}/{}, {}".format(e, losses / len(
                dataset_loader),
                                                                                                   valid_loss,
                                                                                                   losses_glbl / len(
                                                                                                       dataset_loader)))
        else:
            print("On epoch {} total train/validation loss: {}/{}".format(e, losses / len(dataset_loader), valid_loss))
            logging.info(
                "On epoch {} total train/validation loss: {}/{}".format(e, losses / len(dataset_loader), valid_loss))
        if current_sptype_f1 > bestf1_sp_type and separate_save_sptype_preds:
            bestf1_sp_type = current_sptype_f1
            save_sptype_model(model.glbl_generator, run_name, best=True, optimizer=optimizer if use_swa else None)
            print("Best SP type has been saved with score {}".format(current_sptype_f1))
            all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), \
                                                           list(np.array(all_precisions).flatten()), list(
                np.array(total_positives).flatten())

            log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="VALIDATION", ep=e,
                                             all_f1_scores=all_f1_scores, sptype_f1=sptype_f1)
        elif separate_save_sptype_preds:
            print("SP type has not been saved with score {}".format(current_sptype_f1))
            eval_model = deepcopy(model)
            eval_model.glbl_generator = load_sptype_model(run_name + "_best_sptye_eval.pth")
            _ = evaluate(eval_model, sp_data.lbl2ind, run_name=run_name,
                         partitions=validate_partitions, sets=valid_sets, epoch=e, form_sp_reg_data=form_sp_reg_data,
                         simplified=simplified, very_simplified=very_simplified, glbl_lbl_2ind=sp_data.glbl_lbl_2ind,
                         tuned_bert_embs_prefix=tuned_bert_embs_prefix, tune_bert=tune_bert, extended_sublbls=extended_sublbls,
                         random_folds_prefix=random_folds_prefix, train_oh=train_oh)
            sp_pred_mccs, sp_pred_mccs2, lipo_pred_mccs, lipo_pred_mccs2, tat_pred_mccs, tat_pred_mccs2, \
            all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat, all_f1_scores_lipo, all_f1_scores_tat, \
            all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores, sptype_f1 = \
                get_cs_and_sp_pred_results(filename=run_name + ".bin", v=False, return_everything=True, return_class_prec_rec=True)
            all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), \
                                                           list(np.array(all_precisions).flatten()), list(
                np.array(total_positives).flatten())
            log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="VALIDATION", ep=e,
                                             all_f1_scores=all_f1_scores, sptype_f1=sptype_f1)
        else:
            all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), \
                                                           list(np.array(all_precisions).flatten()), list(
                np.array(total_positives).flatten())

            log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="VALIDATION", ep=e,
                                             all_f1_scores=all_f1_scores, sptype_f1=sptype_f1)
        print("VALIDATION: avg mcc on epoch {}: {}".format(e, np.mean(sp_pred_mccs2)))
        if validate_on_mcc:
            print("VALIDATION: avg f1 score on epoch {}: {}".format(e,patiente_metric))
            logging.info("VALIDATION: avg f1 score on epoch {}: {}".format(e,patiente_metric))
        if e == eps - 1:
            patience = 0
        if (valid_loss < best_valid_loss and eps == -1 and not validate_on_mcc) or (eps != -1 and e == eps - 1) or \
                (patiente_metric > best_valid_mcc_and_recall and eps == -1 and validate_on_mcc):
            best_epoch = e
            best_valid_loss = valid_loss
            best_valid_mcc_and_recall = patiente_metric
            save_model(swa_model.module if use_swa and swa_start <= e else model, run_name, tuned_bert_embs_prefix=tuned_bert_embs_prefix, tune_bert=tune_bert, optimizer=optimizer if use_swa else None)
        elif (e > warmup_epochs and valid_loss > best_valid_loss and eps == -1 and not validate_on_mcc) or \
                (e > warmup_epochs and best_valid_mcc_and_recall > patiente_metric and eps == -1 and validate_on_mcc):
            if validate_on_mcc:
                best_val_metrics, val_metric = best_valid_mcc_and_recall, patiente_metric
            else:
                best_val_metrics, val_metric = best_valid_loss, valid_loss
            print("On epoch {} dropped patience to {} because on valid result {} from epoch {} compared to best {}.".
                  format(e, patience, val_metric, best_epoch, best_val_metrics))
            logging.info("On epoch {} dropped patience to {} because on valid result {} from epoch {} compared to best {}.".
                         format(e, patience, val_metric, best_epoch, best_val_metrics))
            patience -= 1
        if use_swa and swa_start == e + 1:
            model, optimizer_state_d = load_model(run_name + "_best_eval.pth", tuned_bert_embs_prefix=tuned_bert_embs_prefix,
                               tune_bert=tune_bert, opt=True)
            if type(optimizer) == list:
                classification_head_optimizer = optim.Adam(model.classification_head.parameters(), lr=0.00001 * lr_multiplier_swa,
                                                           eps=1e-9, weight_decay=wd, betas=(0.9, 0.98), )
                bert_optimizer = optim.Adam(model.ProtBertBFD.parameters(), lr=0.00001, eps=1e-9, weight_decay=wd, betas=(0.9, 0.98), )
                if change_swa_decoder_optimizer:
                    classification_head_optimizer = optim.SGD(model.classification_head.parameters(), lr=0.0001)
                # dont know how to set the optimizer for StepLR to start with 0.0002 lr but have the rest of the states
                # the same (momentum...
                # classification_head_optimizer.load_state_dict(optimizer_state_d[0])
                if not reinint_swa_decoder:
                    bert_optimizer.load_state_dict(optimizer_state_d[1])
                if not change_swa_decoder_optimizer and not reinint_swa_decoder:
                    # if lr scheduler is none, simply load the optimizer and continue training...
                    classification_head_optimizer.load_state_dict(optimizer_state_d[0])
                optimizer = [classification_head_optimizer, bert_optimizer]
                # set the cycle s.t. after cycle_length steps, the lr will be decreased at 10^-5 from 2*10*-4
                if lr_scheduler_swa != "none":
                    scheduler = torch.optim.lr_scheduler_swa.StepLR(optimizer[0], step_size=1, gamma=np.exp(np.log(1/lr_multiplier_swa)/cycle_length))
                swa_model = AveragedModel(model)
                swa_model.module.to("cpu")
                eps = e + int(best_epoch * 0.5)
                # set 0 dropout when swa starts to move along all directions
                for dec_layer in model.classification_head.transformer.decoder.layers:
                    dec_layer.p = 0
                # eps = e + int(best_epoch * 0.5)
                print("Started SWA training for {} more epochs".format(int(best_epoch* 0.5)))
                logging.info("Started SWA training for {} more epochs".format(int(best_epoch * 0.5)))
            else:
                parameters = [
                    {"params": model.classification_head.parameters()},
                    {
                        "params": model.ProtBertBFD.parameters(),
                        "lr": lr,
                    },
                ]
                optimizer = optim.Adam(parameters, lr=lr * 10 if high_lr else lr, eps=1e-9, weight_decay=wd,
                                       betas=(0.9, 0.98), )
                optimizer.load_state_dict(optimizer_state_d)
                scheduler = None
                warmup_scheduler = None
            # since we dont use early stopping anymore, add these seqs
            if add_val_data_on_swa:
                sp_dataset.add_test_seqs()
                dataset_loader = torch.utils.data.DataLoader(sp_dataset,
                                                             batch_size=bs, shuffle=True,
                                                             num_workers=4, collate_fn=collate_fn)
    if use_swa:
        update_bn(dataset_loader, swa_model.to(device))
        save_model(swa_model.module, run_name, tuned_bert_embs_prefix=tuned_bert_embs_prefix, tune_bert=tune_bert)

    other_mdl_name = other_fold_mdl_finished(run_name, partitions[0], validate_partition)
    model = load_model(run_name + "_best_eval.pth", tuned_bert_embs_prefix=tuned_bert_embs_prefix, tune_bert=tune_bert)
    if not deployment_model and not validate_partition is not None or (
            validate_partition is not None and other_mdl_name):
        if separate_save_sptype_preds:
            model.glbl_generator = load_sptype_model(run_name + "_best_sptye_eval.pth")
        # other model is used for the D1 train, D2 validate, D3 test CV (sp6 cv method)
        second_model = load_model(other_mdl_name, tuned_bert_embs_prefix=tuned_bert_embs_prefix) if other_mdl_name else None
        evaluate(model, sp_data.lbl2ind, run_name=run_name + "_best", partitions=test_partition, sets=["train", "test"],
                 form_sp_reg_data=form_sp_reg_data, simplified=simplified, second_model=second_model, very_simplified=very_simplified,
                 glbl_lbl_2ind=sp_data.glbl_lbl_2ind, tuned_bert_embs_prefix=tuned_bert_embs_prefix,
                 tune_bert=tune_bert, extended_sublbls=extended_sublbls,random_folds_prefix=random_folds_prefix,
                 train_oh=train_oh)
        sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores, sptype_f1 = \
            get_cs_and_sp_pred_results(filename=run_name + "_best.bin".format(e), v=False, return_class_prec_rec=True)
        all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), list(
            np.array(all_precisions).flatten()), list(np.array(total_positives).flatten())
        log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="TEST", ep=best_epoch,
                                         all_f1_scores=all_f1_scores, sptype_f1=sptype_f1)
        pos_fp_info = total_positives
        pos_fp_info.extend(false_positives)
        if account_lipos:
            evaluate(model, sp_data.lbl2ind, run_name=run_name + "_lippos_best", partitions=test_partition,
                     sets=["train", "test"], form_sp_reg_data=form_sp_reg_data, simplified=simplified, second_model=second_model,
                     very_simplified=very_simplified, glbl_lbl_2ind=sp_data.glbl_lbl_2ind, account_lipos=account_lipos,
                     tuned_bert_embs_prefix=tuned_bert_embs_prefix, tune_bert=tune_bert, extended_sublbls=extended_sublbls,
                     random_folds_prefix=random_folds_prefix, train_oh=train_oh)
        if test_only_cs:
            evaluate(model, sp_data.lbl2ind, run_name=run_name + "_onlycs_best", partitions=test_partition,
                     sets=["train", "test"], form_sp_reg_data=form_sp_reg_data, simplified=simplified,
                     second_model=second_model, very_simplified=very_simplified, test_only_cs=test_only_cs, glbl_lbl_2ind=sp_data.glbl_lbl_2ind,
                     tuned_bert_embs_prefix=tuned_bert_embs_prefix, tune_bert=tune_bert, extended_sublbls=extended_sublbls,
                     random_folds_prefix=random_folds_prefix, train_oh=train_oh)
            sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores, sptype_f1 = \
                get_cs_and_sp_pred_results(filename=run_name + "_onlycs_best.bin".format(e), v=False,
                                           return_class_prec_rec=True)
            all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), list(
                np.array(all_precisions).flatten()), list(np.array(total_positives).flatten())
            log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="TEST", ep=best_epoch,
                                             all_f1_scores=all_f1_scores, sptype_f1=sptype_f1, only_cs=True)

        if test_beam:
            model = load_model(run_name + "_best_eval.pth", tuned_bert_embs_prefix=tuned_bert_embs_prefix)
            evaluate(model, sp_data.lbl2ind, run_name="best_beam_" + run_name + "_best", partitions=test_partition,
                     sets=["train", "test"],
                     form_sp_reg_data=form_sp_reg_data, simplified=simplified, second_model=second_model,
                     very_simplified=very_simplified,
                     glbl_lbl_2ind=sp_data.glbl_lbl_2ind, tuned_bert_embs_prefix=tuned_bert_embs_prefix,
                     tune_bert=tune_bert,use_beams_search=True, random_folds_prefix=random_folds_prefix,
                     train_oh=train_oh)
            sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores = \
                get_cs_and_sp_pred_results(filename="best_beam_" + run_name + ".bin".format(e), v=False)
            all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), list(
                np.array(all_precisions).flatten()), list(np.array(total_positives).flatten())
            log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="TEST", ep=best_epoch,
                                             beam_txt="using_bm_search", all_f1_scores=all_f1_scores)
            pos_fp_info = total_positives
            pos_fp_info.extend(false_positives)

def test_seqs_w_pretrained_mdl(model_f_name="", test_file="", verbouse=True, tune_bert=False, saliency_map_save_fn="save.bin",hook_layer="bert"):
    # model = nn.Sequential(nn.Linear(100,10), nn.Linear(10,5))
    # opt = torch.optim.Adam(model.parameters(), lr=0.1)
    # torch.save(opt.state_dict(), "asd.txt")
    # model2, model3 = nn.Linear(100, 10), nn.Linear(10, 5)
    # loaded_opt = torch.load("asd.txt")
    # print(loaded_opt)
    # opt1, opt2 = torch.optim.Adam(model2.parameters(), lr=0.1), torch.optim.Adam(model3.parameters(), lr=0.1)
    # opt1.load_state_dict(loaded_opt)
    # exit(1)
    # def a(epoch):
    #
    #     swa_start = 70
    #     swa_lr=0.05
    #     lr_init=0.1
    #     t = epoch / swa_start
    #     lr_ratio = swa_lr / lr_init
    #     swa_lr = 0.05
    #     lr_init = 0.1
    #     if t <= 0.5:
    #         factor = 1.0
    #     elif t <= 0.9:
    #         factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    #     else:
    #         factor = lr_ratio
    #     return lr_init * factor
    #
    # lrs =[]
    # for e in range(150):
    #     lrs.append(a(e))
    # print(lrs)
    # j = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    # print(len(j))
    # exit(1)
    lr_multiplier_swa = 20
    model = nn.Linear(100,10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001 * lr_multiplier_swa)
    c_l = 5
    # lr_scheduler_swa = torch.optim.lr_scheduler_swa.StepLR(optimizer, step_size=1, gamma=np.exp(np.log(1/lr_multiplier_swa)/c_l))
    lr_scheduler_swa = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=np.exp(np.log(1/lr_multiplier_swa)/c_l))
    swa_scheduler = SWALR(optimizer, swa_lr=0.00002, anneal_epochs=30, anneal_strategy='linear')
    # sched = torch.optim.lr_scheduler_swa.SWALR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=1, step_size_down=10, mode='triangular')
    # ot_sched.step()
    # ot_sched.step()
    # sched = SWALR(optimizer, swa_lr=10e-5, anneal_strategy="linear", anneal_epochs=20)
    # lr_scheduler_swa = torch.optim.lr_scheduler_swa.StepLR(optimizer, step_size=1, gamma=10)
    # lr_scheduler_swa = torch.optim.lr_scheduler_swa.MultiStepLR(optimizer, )
    # schedd_ = torch.optim.lr_scheduler_swa.CyclicLR(optimizer,)
    lrs = []
    for i in range(50):
        # sched.step(i%5)
        # lr_scheduler_swa.step(i % (c_l+1))
        lrs.append([i, optimizer.param_groups[0]['lr']])
        if i > 10:
            swa_scheduler.step()
        # ot_sched.step()
        # lr_scheduler_swa.step(i%5)
        # sched.step(i%5)
    print(lrs)
    exit(1)

    folder = get_data_folder()
    sp_data = SPCSpredictionData(form_sp_reg_data=False)
    # hard-code this for now to check some sequences
    test_file = "sp6_partitioned_data_train_1.bin"
    sp_dataset = CSPredsDataset(sp_data.lbl2ind, partitions=None, data_folder=sp_data.data_folder,
                                glbl_lbl_2ind=sp_data.glbl_lbl_2ind, test_f_name=test_file)
    gather_10 = [0, 0, 0]
    sp_type_letters = ["S","L","T"]
    def visualize_importance(outs, grads, seqs_, ind2lbl_, batch_index_, sp_pred_inds_CS_spType_):
        corresponding_grads = {}
        seq_preds_grad_CSgrad = []
        predicted_SPs = []
        for ind_, elem in enumerate(sp_pred_inds_CS_spType_):
            predicted_SPs.append(int(elem.split("_")[0]))
            corresponding_grads[elem] = ind_
        # pred_sp_ind selects the grad from batch_size dimension [batch_size, 70,1024] (gradients contain all batch
        # elements, gradients wrt o other batch elems are 0 for the corresp seq)
        for pred_sp_ind in predicted_SPs:
            # one single case with SP but no CS prediction was found
            if str(pred_sp_ind) + "_spType" in corresponding_grads and str(
                    pred_sp_ind) + "_csPred" in corresponding_grads:
                seq_, pred_ = seqs_[pred_sp_ind], outs[1][pred_sp_ind]
                pred_string = "".join([ind2lbl_[torch.argmax(out_wrd).item()] for out_wrd in pred_])
                grad_ind_spT = corresponding_grads[str(pred_sp_ind)+"_spType"]
                grad_ind_CS = corresponding_grads[str(pred_sp_ind)+"_csPred"]
                seq_preds_grad_CSgrad.append((seq_, pred_string, torch.sum(torch.abs(grads[grad_ind_spT][pred_sp_ind]), dim=-1).detach().cpu().numpy(),
                                             torch.sum(torch.abs(grads[grad_ind_CS][pred_sp_ind]), dim=-1).detach().cpu().numpy()))
        return seq_preds_grad_CSgrad
    hparams, logger = parse_arguments_and_retrieve_logger(save_dir="experiments")

    hparams.train_enc_dec_sp6 = True
    hparams.use_glbl_lbls = False
    lg2ind = sp_data.lg2ind
    aa2ind = None
    tuned_bert_embs_prefix = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # form_sp_reg_data=form_sp_reg_data if not extended_sublbls else False
    # the form_sp_reg_data param is used to both denote teh RR/C... usage and usually had a mandatory glbl label
    # in the SP-cs. The current experiment however tests no-glbl-cs tuning
    model = load_model(model_f_name, dict_file=test_file, tune_bert=tune_bert, testing=True)
    dataset_loader = torch.utils.data.DataLoader(sp_dataset,
                                                 batch_size=10, shuffle=False,
                                                 num_workers=4, collate_fn=collate_fn)
    seqs, some_output = [], []
    ind2lbl = {v:k for k,v in sp_data.lbl2ind.items()}
    all_seq_preds_grad_CSgrad = []
    save_index = 0
    for ind, batch in enumerate(dataset_loader):
        print("{} number of seqs out of {} tested".format(ind, len(dataset_loader)))
        seqs, lbl_seqs, _, glbl_lbls = batch
        some_output, input_gradients, sp_pred_inds_CS_spType= greedy_decode(model, seqs, sp_data.lbl2ind['BS'], sp_data.lbl2ind, tgt=None,
                                            form_sp_reg_data=False, second_model=None, test_only_cs=False,
                                                     glbl_lbls=None, tune_bert=tune_bert, saliency_map=True,
                                                                            hook_layer=hook_layer)
        all_seq_preds_grad_CSgrad.extend(visualize_importance(some_output, input_gradients, seqs, ind2lbl, ind, sp_pred_inds_CS_spType))
    pickle.dump(all_seq_preds_grad_CSgrad,
                open(folder+saliency_map_save_fn, "wb"))
    for seq, pred in zip(seqs, some_output[1]):
        print(seq)
        print("".join([ind2lbl[torch.argmax(out_wrd).item()] for out_wrd in pred]))
        print()
    exit(1)
    # sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions = \
    #     get_cs_and_sp_pred_results(filename=test_file.replace(".bin", "") + "_results.bin", v=False,
    #                                probabilities_file=test_file.replace(".bin", "") + "_results_sp_probs.bin")
    # evaluate(model, sp_data.lbl2ind, test_file.replace(".bin", "") + "_results", epoch=-1,
    #          dataset_loader=dataset_loader,use_beams_search=False, partitions=[1], sets=['test'])
    # exit(1)
    # if verbouse:
    #     sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions = \
    #         get_cs_and_sp_pred_results(filename=test_file.replace(".bin", "") + "_results.bin", v=True, return_everything=False)

    # evaluate(model, sp_data.lbl2ind, test_file.replace(".bin", "") + "_results_beam", epoch=-1, dataset_loader=dataset_loader,
    #          use_beams_search=True)
    # if verbouse:
    #     sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions = \
    #         get_cs_and_sp_pred_results(filename=test_file.replace(".bin", "") + "_results_beam.bin", v=False, return_everything=True)
    #     print(sp_pred_mccs, all_recalls, all_precisions)

    sp_pred_mccs, sp_pred_mccs2, lipo_pred_mccs, lipo_pred_mccs2, tat_pred_mccs, tat_pred_mccs2, \
    all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat = \
        get_cs_and_sp_pred_results(filename=test_file.replace(".bin", "") + "_results.bin", v=False,
                                   return_everything=True)
    beam_sp_pred_mccs, beam_sp_pred_mccs2, beam_lipo_pred_mccs, beam_lipo_pred_mccs2, beam_tat_pred_mccs, beam_tat_pred_mccs2, \
    beam_all_recalls_lipo, beam_all_precisions_lipo, beam_all_recalls_tat, beam_all_precisions_tat = \
        get_cs_and_sp_pred_results(filename=test_file.replace(".bin", "") + "_results_beam.bin", v=False,
                                   return_everything=True)
    print(sp_pred_mccs2, beam_sp_pred_mccs2)
    print("MCC1/MCC2 SEC/SPII")
    print(lipo_pred_mccs, lipo_pred_mccs2)
    print(beam_lipo_pred_mccs, beam_lipo_pred_mccs2)

    print("Recalls SEC/SPII")
    print(all_recalls_lipo)
    print(beam_all_recalls_lipo)

    print(sp_pred_mccs2, beam_sp_pred_mccs2)
    print("MCC1/MCC2 TAT/SPI")
    print(tat_pred_mccs, tat_pred_mccs2)
    print(beam_tat_pred_mccs, beam_tat_pred_mccs2)

    print("Recalls TAT/SPI")
    print(all_recalls_tat)
    print(beam_all_recalls_tat)
