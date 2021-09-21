from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from tqdm import tqdm
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


def init_model(ntoken, lbl2ind={}, lg2ind={}, dropout=0.5, use_glbl_lbls=False, no_glbl_lbls=6,
               ff_dim=1024 * 4, nlayers=3, nheads=8, aa2ind={}, train_oh=False, glbl_lbl_version=1):
    model = TransformerModel(ntoken=ntoken, d_model=1024, nhead=nheads, d_hid=1024, nlayers=nlayers,
                             lbl2ind=lbl2ind, lg2ind=lg2ind, dropout=dropout, use_glbl_lbls=use_glbl_lbls,
                             no_glbl_lbls=no_glbl_lbls, ff_dim=ff_dim, aa2ind=aa2ind, train_oh=train_oh,
                             glbl_lbl_version=glbl_lbl_version)
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
    sp_probs = []
    sp_logits = []
    all_seq_sp_probs = []
    all_seq_sp_logits = []

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
        if i == 0:
            # extract the sp-presence probabilities
            sp_probs = [sp_prb.item() for sp_prb in torch.nn.functional.softmax(prob, dim=-1)[:, lbl2ind['S']]]
            all_seq_sp_probs = [[sp_prob.item()] for sp_prob in torch.nn.functional.softmax(prob, dim=-1)[:, lbl2ind['S']]]
            all_seq_sp_logits = [[sp_prob.item()] for sp_prob in prob[:, lbl2ind['S']]]
        else:
            # used to update the sequences of probabilities
            softmax_probs = torch.nn.functional.softmax(prob, dim=-1)
            next_sp_probs = softmax_probs[:, lbl2ind['S']]
            next_sp_logits = prob[:, lbl2ind['S']]
            for seq_prb_ind in range(len(all_seq_sp_probs)):
                all_seq_sp_probs[seq_prb_ind].append(next_sp_probs[seq_prb_ind].item())
                all_seq_sp_logits[seq_prb_ind].append(next_sp_logits[seq_prb_ind].item())
        all_probs.append(prob)
        _, next_words = torch.max(prob, dim=1)
        next_word = [nw.item() for nw in next_words]
        current_ys = []
        for bach_ind in range(len(src)):
            current_ys.append(ys[bach_ind])
            current_ys[-1].append(next_word[bach_ind])
        ys = current_ys
    return ys, torch.stack(all_probs).transpose(0, 1), sp_probs, all_seq_sp_probs, all_seq_sp_logits

def beam_decode(model, src, start_symbol, lbl2ind, tgt=None, beam_width=3):
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
    all_probs, sp_probs, sp_logits = [], [], []
    current_batch_size = len(src)
    log_probs = torch.zeros(current_batch_size, beam_width)
    for i in range(max(seq_lens) + 1):
        with torch.no_grad():
            tgt_mask = (generate_square_subsequent_mask(len(ys[0]) + 1))
            # seq len, batch size, emb dimensions
            # 5,5,3 -> 5,15,3 -> 5,3,15 ->
            if i == 1:
                memory = memory.repeat(1, 1, beam_width).reshape(memory.shape[0], current_batch_size * beam_width, -1)
            out = model.decode(ys, memory.to(device), tgt_mask.to(device))
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
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
            for _ in range(len(src) * (beam_width-1)):
                ys.append([])
            for bach_ind in range(len(src) * beam_width):
                current_ys.append(ys[bach_ind])
                current_ys[-1].append(next_word[bach_ind])
        elif i == max(seq_lens):
            for seq_ind in range(len(src)):
                next_chosen_seq = next_word_col_indices[seq_ind]
                previous_word_seq = ys[seq_ind * beam_width + next_chosen_seq // beam_width].copy()
                current_ys.append(previous_word_seq)
                current_ys[-1].append(next_words[seq_ind * beam_width + next_chosen_seq // beam_width, next_chosen_seq % beam_width].item())
        else:
            for seq_ind in range(len(src)):
                for j in range(beam_width):
                    next_chosen_seq = next_word_col_indices[seq_ind * beam_width + j]
                    # choose the prev sequence to which this maximal probability corresponds
                    previous_word_seq = ys[seq_ind * beam_width + next_chosen_seq // beam_width].copy()
                    current_ys.append(previous_word_seq)
                    current_ys[-1].append(next_words[seq_ind * beam_width + next_chosen_seq // beam_width, next_chosen_seq % beam_width].item())
        ys = current_ys

    return ys, torch.tensor([0.1])

def translate(model: torch.nn.Module, src: str, bos_id, lbl2ind, tgt=None, use_beams_search=False):
    model.eval()
    if use_beams_search:
        tgt_tokens, probs  = beam_decode(model, src, start_symbol=bos_id, lbl2ind=lbl2ind, tgt=tgt)
        sp_probs, all_sp_probs, all_seq_sp_logits = None, None, None
    else:
        tgt_tokens, probs, sp_probs, all_sp_probs, all_seq_sp_logits = greedy_decode(model, src, start_symbol=bos_id, lbl2ind=lbl2ind, tgt=tgt)
    return tgt_tokens, probs, sp_probs, all_sp_probs, all_seq_sp_logits


def eval_trainlike_loss(model, lbl2ind, run_name="", test_batch_size=50, partitions=[0, 1], sets=["train"]):
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
                logits, _ = model(src, tgt)
            else:
                logits = model(src, tgt)
        targets = padd_add_eos_tkn(tgt, sp_data.lbl2ind)
        loss = loss_fn(logits.transpose(0, 1).reshape(-1, logits.shape[-1]), targets.reshape(-1))
        total_loss += loss.item()
        # print("Number of sequences tested: {}".format(ind * test_batch_size))
    return total_loss / len(dataset_loader)


def evaluate(model, lbl2ind, run_name="", test_batch_size=50, partitions=[0, 1], sets=["train"], epoch=-1, dataset_loader=None,
             use_beams_search=False):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=lbl2ind["PD"])
    eval_dict = {}
    seqs2probs = {}
    model.eval()
    val_or_test = "test" if len(sets) == 2 else "validation"
    if dataset_loader is None:
        sp_data = SPCSpredictionData()
        sp_dataset = CSPredsDataset(sp_data.lbl2ind, partitions=partitions, data_folder=sp_data.data_folder,
                                    glbl_lbl_2ind=sp_data.glbl_lbl_2ind, sets=sets)

        dataset_loader = torch.utils.data.DataLoader(sp_dataset,
                                                     batch_size=test_batch_size, shuffle=False,
                                                     num_workers=4, collate_fn=collate_fn)


    ind2lbl = {v: k for k, v in lbl2ind.items()}
    total_loss = 0
    for ind, (src, tgt, _, _) in tqdm(enumerate(dataset_loader), "Epoch {} {}".format(epoch, val_or_test), total=len(dataset_loader)):
        # print("Number of sequences tested: {}".format(ind * test_batch_size))
        src = src
        tgt = tgt
        predicted_tokens, probs, sp_probs, all_sp_probs, all_seq_sp_logits = \
            translate(model, src, lbl2ind['BS'], lbl2ind, tgt=tgt, use_beams_search=use_beams_search)
        true_targets = padd_add_eos_tkn(tgt, lbl2ind)
        if not use_beams_search:
            total_loss += loss_fn(probs.reshape(-1, 10), true_targets.reshape(-1)).item()
        for s, t, pt in zip(src, tgt, predicted_tokens):
            predicted_lbls = "".join([ind2lbl[i] for i in pt])
            eval_dict[s] = predicted_lbls[:len(t)]
        if sp_probs is not None:
            for s, sp_prob, all_sp_probs, all_sp_logits in zip(src, sp_probs, all_sp_probs, all_seq_sp_logits):
                seqs2probs[s] = (sp_prob, all_sp_probs, all_sp_logits)
    pickle.dump(eval_dict, open(run_name + ".bin", "wb"))
    if sp_probs is not None and len(sets) > 1:
        # retrieve the dictionary of calibration only for the test set (not for validation) - for now it doesn't
        # make sense to do prob calibration since like 98% of predictions have >0.99 and are correct. See with weight decay
        pickle.dump(seqs2probs, open(run_name + "_sp_probs.bin", "wb"))

    return total_loss / len(dataset_loader)


def save_model(model, model_name=""):
    folder = get_data_folder()
    model.input_encoder.seq2emb = {}
    torch.save(model, folder + model_name + "_best_eval.pth")
    model.input_encoder.update()


def load_model(model_path, dict_file=None):
    folder = get_data_folder()
    model = torch.load(folder + model_path)
    model.input_encoder.update(emb_f_name=dict_file)
    return model


def log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="VALIDATION", ep=-1, beam_txt="Greedy"):
    print("{}_{}, epoch {} Mean sp_pred mcc for life groups: {}, {}, {}, {}".format(beam_txt, test_on, ep, *sp_pred_mccs))
    print("{}_{}, epoch {} Mean cs recall: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(beam_txt,
        test_on, ep, *all_recalls))
    print("{}_{}, epoch {} Mean cs precision: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(beam_txt,
        test_on, ep, *all_precisions))
    logging.info("{}_{}, epoch {}: Mean sp_pred mcc for life groups: {}, {}, {}, {}".format(beam_txt, test_on, ep, *sp_pred_mccs))
    logging.info("{}_{}, epoch {}: Mean cs recall: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
        beam_txt,test_on, ep, *all_recalls))
    logging.info(
        "{}_{}, epoch {}: Mean cs precision: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
            beam_txt, test_on, ep, *all_precisions))

def get_lr_scheduler(opt, lr_scheduler=False, lr_sched_warmup=0):
    def get_schduler_type(op, lr_sched):
        if lr_sched == "expo":
            return ExponentialLR(optimizer=op, gamma=0.98)
        elif lr_sched == "step":
            return StepLR(optimizer=op, gamma=0.1, step_size=20)

    if lr_scheduler == "none":
        return None, None
    elif lr_sched_warmup >= 2:
        scheduler = get_schduler_type(opt, lr_scheduler)
        # in lr_sched_warmup epochs, the learning rate will be increased by 10. 1e-5 is the stable lr found. This
        # lr will increase to 1e-4 after lr_sched_warmup steps
        warmup_scheduler = ExponentialLR(opt, gamma=10 ** (1/lr_sched_warmup))
        return warmup_scheduler, scheduler
    else:
        return None, get_schduler_type(opt, lr_scheduler)

def euk_importance_avg(cs_mcc):
    return (3 / 4) * cs_mcc[0] + (1 / 4) * np.mean(cs_mcc[1:])


def train_cs_predictors(bs=16, eps=20, run_name="", use_lg_info=False, lr=0.0001, dropout=0.5,
                        test_freq=1, use_glbl_lbls=False, partitions=[0, 1], ff_d=4096, nlayers=3, nheads=8,
                        patience=30, train_oh=False, deployment_model=False, lr_scheduler=False, lr_sched_warmup=0,
                        test_beam=False, wd=0., glbl_lbl_weight=1, glbl_lbl_version=1, validate_on_test=False):
    logging.info("Log from here...")
    test_partition = set() if deployment_model else {0, 1, 2} - set(partitions)
    partitions = [0,1,2] if deployment_model else partitions
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
    model = init_model(len(sp_data.lbl2ind.keys()), lbl2ind=sp_data.lbl2ind, lg2ind=lg2ind,
                       dropout=dropout, use_glbl_lbls=use_glbl_lbls, no_glbl_lbls=len(sp_data.glbl_lbl_2ind.keys()),
                       ff_dim=ff_d, nlayers=nlayers, nheads=nheads, train_oh=train_oh, aa2ind=aa2ind,
                       glbl_lbl_version=glbl_lbl_version)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=sp_data.lbl2ind["PD"])
    loss_fn_glbl = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=wd)
    warmup_scheduler, scheduler = get_lr_scheduler(optimizer, lr_scheduler, lr_sched_warmup)
    best_avg_mcc = -1
    best_valid_loss = 5 ** 10
    best_epoch = 0
    e = -1
    while patience != 0:
        print("\n\nLR:",optimizer.param_groups[0]['lr'],"\n\n")
        model.train()
        e += 1
        losses = 0
        losses_glbl = 0
        for ind, batch in tqdm(enumerate(dataset_loader), "Epoch {} train:".format(e), total=len(dataset_loader)):
            seqs, lbl_seqs, _, glbl_lbls = batch
            if use_glbl_lbls:
                logits, glbl_logits = model(seqs, lbl_seqs)
                optimizer.zero_grad()
                targets = padd_add_eos_tkn(lbl_seqs, sp_data.lbl2ind)
                loss = loss_fn(logits.transpose(0, 1).reshape(-1, logits.shape[-1]), targets.reshape(-1))
                losses += loss.item()

                loss_glbl = loss_fn_glbl(glbl_logits, torch.tensor(glbl_lbls, device=device))
                losses_glbl += loss_glbl.item()
                loss += loss_glbl * glbl_lbl_weight
            else:
                logits = model(seqs, lbl_seqs)
                optimizer.zero_grad()
                targets = padd_add_eos_tkn(lbl_seqs, sp_data.lbl2ind)
                loss = loss_fn(logits.transpose(0, 1).reshape(-1, logits.shape[-1]), targets.reshape(-1))
                losses += loss.item()

            loss.backward()
            optimizer.step()
        if scheduler is not None:
            if e < lr_sched_warmup and lr_sched_warmup > 2:
                warmup_scheduler.step()
            else:
                scheduler.step()
        if validate_on_test:
            validate_partitions = list(test_partition)
            _ = evaluate(model, sp_data.lbl2ind, run_name=run_name, partitions=validate_partitions, sets=["test"],
                         epoch=e)
            sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions \
                = get_cs_and_sp_pred_results(filename=run_name + ".bin", v=False)
            valid_loss = -np.mean(sp_pred_mccs)
            # revert valid_loss to not change the loss condition next ( this won't be a loss
            # but it's the quickest way to test performance when validation with the test set
        else:
            valid_loss = eval_trainlike_loss(model, sp_data.lbl2ind, run_name=run_name, partitions=partitions,
                                             sets=["test"])
            _ = evaluate(model, sp_data.lbl2ind, run_name=run_name, partitions=partitions, sets=["test"],
                         epoch=e)
            sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions \
                = get_cs_and_sp_pred_results(filename=run_name + ".bin", v=False)
        # sp_pred_mccs
        all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), \
                                                       list(np.array(all_precisions).flatten()), list(
            np.array(total_positives).flatten())
        if use_glbl_lbls:
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
        log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="VALIDATION", ep=e)

        print("VALIDATION: avg mcc on epoch {}: {}".format(e, euk_importance_avg(sp_pred_mccs)))
        if (valid_loss < best_valid_loss and eps == -1) or (eps != -1 and e == eps - 1):
            best_epoch = e
            best_valid_loss = valid_loss
            save_model(model, run_name)
            if e == eps - 1:
                patience = 0
        elif e > 10 and valid_loss > best_valid_loss and eps == -1:
            print("On epoch {} dropped patience to {} because on valid result {} compared to best {}.".
                  format(e, patience, valid_loss, best_valid_loss))
            logging.info("On epoch {} dropped patience to {} because on valid result {} compared to best {}.".
                         format(e, patience, valid_loss, best_valid_loss))
            patience -= 1
    if not deployment_model:
        model = load_model(run_name + "_best_eval.pth")
        evaluate(model, sp_data.lbl2ind, run_name=run_name + "_best", partitions=test_partition, sets=["train", "test"])
        sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions = \
            get_cs_and_sp_pred_results(filename=run_name + "_best.bin".format(e), v=False)
        all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), list(
            np.array(all_precisions).flatten()), list(np.array(total_positives).flatten())
        log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="TEST", ep=best_epoch)
        pos_fp_info = total_positives
        pos_fp_info.extend(false_positives)
        if test_beam:
            model = load_model(run_name + "_best_eval.pth")
            evaluate(model, sp_data.lbl2ind, run_name="best_beam_" + run_name , partitions=test_partition,
                     sets=["train", "test"], use_beams_search=True)
            sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions = \
                get_cs_and_sp_pred_results(filename="best_beam_" + run_name +".bin".format(e), v=False)
            all_recalls, all_precisions, total_positives = list(np.array(all_recalls).flatten()), list(
                np.array(all_precisions).flatten()), list(np.array(total_positives).flatten())
            log_and_print_mcc_and_cs_results(sp_pred_mccs, all_recalls, all_precisions, test_on="TEST", ep=best_epoch, beam_txt="using_bm_search")
            pos_fp_info = total_positives
            pos_fp_info.extend(false_positives)

def test_seqs_w_pretrained_mdl(model_f_name="", test_file="", verbouse=True):
    # model = load_model(model_f_name, dict_file=None)
    model = load_model(model_f_name, dict_file=test_file)
    sp_data = SPCSpredictionData()
    sp_dataset = CSPredsDataset(sp_data.lbl2ind, partitions=None, data_folder=sp_data.data_folder,
                                glbl_lbl_2ind=sp_data.glbl_lbl_2ind, test_f_name=test_file)

    dataset_loader = torch.utils.data.DataLoader(sp_dataset,
                                                 batch_size=50, shuffle=True,
                                                 num_workers=4, collate_fn=collate_fn)
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
    print(beam_tat_pred_mccs,beam_tat_pred_mccs2)


    print("Recalls TAT/SPI")
    print(all_recalls_tat)
    print(beam_all_recalls_tat)

