import time
import logging
import sys

logging.getLogger('some_logger')

import os
import numpy as np
import random
import pickle
import torch.nn as nn
import torch.optim as optim
import torch
sys.path.append(os.path.abspath(".."))
from misc.visualize_cs_pred_results import get_cs_and_sp_pred_results, get_summary_sp_acc, get_summary_cs_acc
from sp_data.data_utils import SPbinaryData, BinarySPDataset, SPCSpredictionData, CSPredsDataset, collate_fn
from models.transformer_nmt import TransformerModel


def init_model(ntoken, partitions, lbl2ind={}, lg2ind={}, dropout=0.5, use_glbl_lbls=False,no_glbl_lbls=6, ff_dim=1024*4, nlayers=3, nheads=8,
               aa2ind = {}, train_oh = False):
    model = TransformerModel(ntoken=ntoken, d_model=1024, nhead=nheads, d_hid=1024, nlayers=nlayers, partitions=partitions,
                             lbl2ind=lbl2ind, lg2ind=lg2ind, dropout=dropout, use_glbl_lbls=use_glbl_lbls,
                             no_glbl_lbls=no_glbl_lbls, ff_dim=ff_dim, aa2ind=aa2ind, train_oh=train_oh)
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


def greedy_decode(model, src, start_symbol, lbl2ind, tgt=None):
    ind2lbl = {v: k for k, v in lbl2ind.items()}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    src = src
    seq_lens = [len(src_) for src_ in src]
    with torch.no_grad():
        memory = model.encode(src)
    # ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    ys = []
    if not ys:
        for _ in range(len(src)):
            ys.append([])
    model.eval()
    all_probs = []
    for i in range(max(seq_lens) + 1):
        with torch.no_grad():
            tgt_mask = (generate_square_subsequent_mask(len(ys[0]) + 1))
            out = model.decode(ys, memory.to(device), tgt_mask.to(device))
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
        all_probs.append(prob)
        _, next_words = torch.max(prob, dim=1)
        next_word = [nw.item() for nw in next_words]
        current_ys = []
        for bach_ind in range(len(src)):
            current_ys.append(ys[bach_ind])
            current_ys[-1].append(next_word[bach_ind])
        ys = current_ys
    return ys, torch.stack(all_probs).transpose(0,1)


def translate(model: torch.nn.Module, src: str, bos_id, lbl2ind, tgt=None):
    model.eval()
    tgt_tokens, probs = greedy_decode(model, src, start_symbol=bos_id, lbl2ind=lbl2ind, tgt=tgt)
    return tgt_tokens, probs


def eval_trainlike_loss(model, lbl2ind, run_name="", test_batch_size=50, partitions=[0,1], sets=["train"]):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=lbl2ind["PD"])
    model.eval()
    sp_data = SPCSpredictionData()
    sp_dataset = CSPredsDataset(sp_data.lbl2ind, partitions=partitions, data_folder=sp_data.data_folder,
                                glbl_lbl_2ind=sp_data.glbl_lbl_2ind, sets=sets)

    dataset_loader = torch.utils.data.DataLoader(sp_dataset,
                                                 batch_size=test_batch_size, shuffle=False,
                                                 num_workers=4, collate_fn=collate_fn)
    ind2lbl = {v: k for k, v in lbl2ind.items()}
    total_loss = 0
    for ind, (src, tgt, _, _) in enumerate(dataset_loader):
        with torch.no_grad():
            if model.use_glbl_lbls:
                logits, _  = model(src, tgt)
            else:
                logits = model(src, tgt)
        targets = padd_add_eos_tkn(tgt, sp_data.lbl2ind)
        loss = loss_fn(logits.transpose(0, 1).reshape(-1, logits.shape[-1]), targets.reshape(-1))
        total_loss += loss.item()
        # print("Number of sequences tested: {}".format(ind * test_batch_size))
    return total_loss / len(dataset_loader)


def evaluate(model, lbl2ind, run_name="", test_batch_size=50, partitions=[0,1], sets=["train"]):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=lbl2ind["PD"])
    eval_dict = {}
    model.eval()
    sp_data = SPCSpredictionData()
    sp_dataset = CSPredsDataset(sp_data.lbl2ind, partitions=partitions, data_folder=sp_data.data_folder,
                                glbl_lbl_2ind=sp_data.glbl_lbl_2ind, sets=sets)

    dataset_loader = torch.utils.data.DataLoader(sp_dataset,
                                                 batch_size=test_batch_size, shuffle=False,
                                                 num_workers=4, collate_fn=collate_fn)
    ind2lbl = {v: k for k, v in lbl2ind.items()}
    total_loss = 0
    for ind, (src, tgt, _, _) in enumerate(dataset_loader):
        # print("Number of sequences tested: {}".format(ind * test_batch_size))
        src = src
        tgt = tgt
        predicted_tokens, probs = translate(model, src, lbl2ind['BS'], lbl2ind, tgt=tgt)
        true_targets = padd_add_eos_tkn(tgt, sp_data.lbl2ind)
        total_loss += loss_fn(probs.reshape(-1,10), true_targets.reshape(-1)).item()
        for s, t, pt in zip(src, tgt, predicted_tokens):
            predicted_lbls = "".join([ind2lbl[i] for i in pt])
            eval_dict[s] = predicted_lbls[:len(t)]
    pickle.dump(eval_dict, open(run_name + ".bin", "wb"))
    return total_loss/len(dataset_loader)

def save_model(model, model_name=""):
    folder = get_data_folder()
    model.input_encoder.seq2emb = {}
    torch.save(model, folder + model_name+"_best_eval.pth")
    model.input_encoder.update()

def load_model(model_path, ntoken, partitions, lbl2ind, lg2ind, dropout=0.5, use_glbl_lbls=False,no_glbl_lbls=6, ff_dim=1024*4):
    folder = get_data_folder()
    model= torch.load(folder + model_path)
    model.input_encoder.update()
    return model

def log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="VALIDATION", ep=-1):
    print("{}, epoch {} Mean sp_pred mcc for life groups: {}, {}, {}, {}".format(test_on, ep, *sp_pred_mccs))
    print("{}, epoch {} Mean cs recall: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
        test_on, ep, *all_recalls))
    print("{}, epoch {} Mean cs precision: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
        test_on, ep, *all_precisions))
    logging.info("{}, epoch {}: Mean sp_pred mcc for life groups: {}, {}, {}, {}".format(test_on, ep, *sp_pred_mccs))
    logging.info("{}, epoch {}: Mean cs recall: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
        test_on, ep, *all_recalls))
    logging.info("{}, epoch {} Mean cs precision: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
        test_on, ep, *all_precisions))

def euk_importance_avg(cs_mcc):
    return (3/4) * cs_mcc[0] + (1/4) * np.mean(cs_mcc[1:])

def train_cs_predictors(bs=16, eps=20, run_name="", use_lg_info=False, lr=0.0001, dropout=0.5,
                        test_freq=1, use_glbl_lbls=False, partitions=[0, 1], ff_d=4096, nlayers=3, nheads=8, patience=30,
                        train_oh=False):
    logging.info("Log from here...")
    test_partition = list({0,1,2} - set(partitions))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sp_data = SPCSpredictionData()
    sp_dataset = CSPredsDataset(sp_data.lbl2ind, partitions=partitions, data_folder=sp_data.data_folder,
                                glbl_lbl_2ind=sp_data.glbl_lbl_2ind)
    dataset_loader = torch.utils.data.DataLoader(sp_dataset,
                                                 batch_size=bs, shuffle=True,
                                                 num_workers=4, collate_fn=collate_fn)
    if len(sp_data.lg2ind.keys()) <= 1 or not use_lg_info:
        lg2ind = None
    elif len(sp_data.lg2ind.keys()) > 1 and use_lg_info:
        lg2ind = sp_data.lg2ind
    aa2ind = sp_data.aa2ind if train_oh else None
    model = init_model(len(sp_data.lbl2ind.keys()), partitions=[0, 1], lbl2ind=sp_data.lbl2ind, lg2ind=lg2ind,
                       dropout=dropout, use_glbl_lbls=use_glbl_lbls, no_glbl_lbls=len(sp_data.glbl_lbl_2ind.keys()),
                       ff_dim=ff_d, nlayers=nlayers, nheads=nheads, train_oh=train_oh, aa2ind=aa2ind)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=sp_data.lbl2ind["PD"])
    loss_fn_glbl = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    ind2lbl = {ind: lbl for lbl, ind in sp_data.lbl2ind.items()}

    best_avg_mcc = -1
    best_valid_loss = 5**10
    best_epoch = 0
    e = -1
    while patience != 0:
        model.train()
        e += 1
        losses = 0
        losses_glbl = 0
        for ind, batch in enumerate(dataset_loader):
            seqs, lbl_seqs, _, glbl_lbls = batch
            if use_glbl_lbls:
                logits, glbl_logits = model(seqs, lbl_seqs)
                optimizer.zero_grad()
                targets = padd_add_eos_tkn(lbl_seqs, sp_data.lbl2ind)
                loss = loss_fn(logits.transpose(0, 1).reshape(-1, logits.shape[-1]), targets.reshape(-1))
                losses += loss.item()

                loss_glbl = loss_fn_glbl(glbl_logits, torch.tensor(glbl_lbls, device=device))
                losses_glbl += loss_glbl.item()
                loss += loss_glbl
            else:
                logits = model(seqs, lbl_seqs)
                optimizer.zero_grad()
                targets = padd_add_eos_tkn(lbl_seqs, sp_data.lbl2ind)
                loss = loss_fn(logits.transpose(0, 1).reshape(-1, logits.shape[-1]), targets.reshape(-1))
                losses += loss.item()

            loss.backward()
            optimizer.step()
        _ = evaluate(model, sp_data.lbl2ind, run_name=run_name, partitions=partitions, sets=["test"])
        valid_loss = eval_trainlike_loss(model, sp_data.lbl2ind, run_name=run_name, partitions=partitions, sets=["test"])
        sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions\
            = get_cs_and_sp_pred_results(filename=run_name + ".bin", v=False)
        print(euk_importance_avg(sp_pred_mccs), sp_pred_mccs)
        all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), \
                              list(np.array(all_precisions).flatten()), list(np.array(total_positives).flatten())
        if use_glbl_lbls:
            print("On epoch {} total train/validation loss and glbl loss: {}/{}, {}".format(e, losses / len(dataset_loader),
                                                                                valid_loss, losses_glbl / len(dataset_loader)))
            logging.info("On epoch {} total train/validation loss and glbl loss: {}/{}, {}".format(e, losses / len(dataset_loader),
                                                                                   valid_loss, losses_glbl / len(dataset_loader)))
        else:
            print("On epoch {} total train/validation loss: {}/{}".format(e, losses / len(dataset_loader), valid_loss))
            logging.info("On epoch {} total train/validation loss: {}/{}".format(e, losses / len(dataset_loader), valid_loss))
        log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="VALIDATION", ep=e)

        print("VALIDATION: avg mcc on epoch {}: {}".format(e, euk_importance_avg(sp_pred_mccs)))
        if (valid_loss < best_valid_loss and eps == -1) or (eps != -1 and e == eps-1):
            best_epoch = e
            best_valid_loss = valid_loss
            save_model(model, run_name)
            if e == eps - 1:
                patience = 0
        elif e > 20 and valid_loss > best_valid_loss and eps == -1:
            print("On epoch {} dropped patience to {} because on valid result {} compared to best {}.".
                  format(e, patience, valid_loss, best_valid_loss))
            logging.info("On epoch {} dropped patience to {} because on valid result {} compared to best {}.".
                  format(e, patience, valid_loss, best_valid_loss))
            patience -= 1
    model = load_model(run_name + "_best_eval.pth", len(sp_data.lbl2ind.keys()), partitions=[0, 1], lbl2ind=sp_data.lbl2ind, lg2ind=lg2ind,
                       dropout=dropout, use_glbl_lbls=use_glbl_lbls, no_glbl_lbls=len(sp_data.glbl_lbl_2ind.keys()))
    
    evaluate(model, sp_data.lbl2ind, run_name=run_name+"_best", partitions=test_partition, sets=["train", "test"])
    sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions = \
        get_cs_and_sp_pred_results(filename=run_name + "_best.bin".format(e), v=False)
    all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), list(
        np.array(all_precisions).flatten()), list(np.array(total_positives).flatten())
    log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="TEST", ep=best_epoch)
    print("TEST: Total positives and false positives: ", total_positives, false_positives)
    print("TEST: True positive predictions {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "
          "{}, {}, {}, {}, {},".format(*np.concatenate(predictions)))
    pos_fp_info  = total_positives
    pos_fp_info.extend(false_positives)
    logging.info("TEST: Total positives and false positives: {}, {}, {}, {}, {}, {}, {}, {}", pos_fp_info)
    logging.info("TEST: True positive predictions {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "
          "{}, {}, {}, {}, {},".format(*np.concatenate(predictions)))

