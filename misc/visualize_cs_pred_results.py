import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances as euclidian_distance
import random
from io import StringIO
import requests as r
from matplotlib.collections import LineCollection
import torch.nn
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef as compute_mcc
import os
import pickle
from Bio import SeqIO
import numpy as np

def clean_sec_sp2_preds(seq, preds):
    last_l_ind = preds.rfind("L")
    min_i = 10
    for i in range(-2,3):
        if seq[last_l_ind + i + 1] == "C":
            if np.abs(i) < np.abs(min_i):
                best_ind = i
                min_i = i
    if min_i == 0:
        return preds
    elif min_i == 10:
        return preds.replace("L", "T")
    elif min_i > 0:
        return preds[:last_l_ind] + min_i * "L" + preds[last_l_ind + min_i:]
    elif min_i < 0:
        return preds[:last_l_ind + min_i] + np.abs(min_i) * preds[last_l_ind+1] + preds[last_l_ind+np.abs(min_i):]


def reassign_cs(s, p):
    residues = s[p.rfind("L")-3:p.rfind("L")+3]
    ind = residues.find("C")
    if ind != -1:
        return "L" * (p.rfind("L")-3+ind) + (70-p.rfind("L") +3 -ind) * "O"
    else:
        return -1

def get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="SP", sptype_preds=None):
    def get_acc_for_tolerence(ind, t_lbl, sp_letter):
        true_cs = 0
        while t_lbl[true_cs] == sp_letter and true_cs < len(t_lbl):
            true_cs += 1
        if np.abs(true_cs - ind) == 0:
            return np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        elif np.abs(true_cs - ind) == 1:
            return np.array([0, 1, 1, 1, 1, 0, 1, 0, 0, 0])
        elif np.abs(true_cs - ind) == 2:
            return np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])
        elif np.abs(true_cs - ind) == 3:
            return np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 0])
        elif ind != 0:
            # if ind==0, SP was predicted, but CS prediction is off for all tolerence levels, meaning it's a false positive
            # if ind != 0, and teh corresponding SP was predicted, this becomes a false positive on CS predictions for all
            # tolerance levels
            return np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
        else:
            # if ind==0, SP was not even predicted (so there is no CS prediction) and this affects precision metric
            # tp/(tp+fp). It means this a fn
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    ind2glbl_lbl = {0: 'NO_SP', 1: 'SP', 2: 'TATLIPO', 3: 'LIPO', 4: 'TAT', 5: 'PILIN'}
    glbllbl2_ind = {v:k for k,v in ind2glbl_lbl.items()}
    sptype2letter = {'TAT': 'T', 'LIPO': 'L', 'PILIN': 'P', 'TATLIPO': 'T', 'SP': 'S'}
    sp_types = ["S", "T", "L", "P"]
    ind2sptype = {0: 'NO_SP', 1: 'SP', 2: 'TATLIPO', 3: 'LIPO', 4: 'TAT', 5: 'PILIN'}
    sptype2ind = {v:k for k,v in ind2sptype.items()}
    # S = signal_peptide; T = Tat/SPI or Tat/SPII SP; L = Sec/SPII SP; P = SEC/SPIII SP; I = cytoplasm; M = transmembrane; O = extracellular;
    # order of elemnts in below list:
    # (eukaria_correct_tollerence_0, eukaria_correct_tollerence_1, eukaria_correct_tollerence_2, ..3, eukaria_total, eukaria_all_pos_preds)
    # (negative_correct_tollerence_0, negative_correct_tollerence_1, negative_correct_tollerence_2, ..3, negative_total, negative_all_pos_preds)
    # (positive_correct_tollerence_0, positive_correct_tollerence_1, positive_correct_tollerence_2, ..3, positive_total, positive_all_pos_preds)
    # (archaea_correct_tollerence_0, archaea_correct_tollerence_1, archaea_correct_tollerence_2, ..3, archaea_total, archae_all_pos_preds)
    # We used precision and recall to assess CS predictions, where precision is defined as the fraction of CS predictions
    # that are correct, and recall is the fraction of real SPs that are predicted as the correct SP type and have the correct CS assigned.
    grp2_ind = {"EUKARYA": 0, "NEGATIVE": 1, "POSITIVE": 2, "ARCHAEA": 3}
    predictions = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    predictions = [np.array(p) for p in predictions]
    count, count2 = 0, 0
    count_tol_fn, count_complete_fn, count_otherSPpred = 0, 0, 0
    sp_letter = sptype2letter[sp_type]
    cnt1, cnt2, ttl = 0,0,0 # tp, total, fp
    cnt4, cnt5 = 0, 0
    for l, s, t, p in zip(life_grp, seqs, true_lbls, pred_lbls):
        # if ind2glbl_lbl[sptype_preds[s]] == "LIPO" and s[p.rfind("L")+1]!="C":
        lg, sp_info = l.split("|")

        # I previously re-replace Tatlipo seqs to have lbls W instead of T; correct prediction of type is based on
        # dictionary sptype_preds anyways
        t = t.replace("W", "T")
        #     print(p)
        #     print(t)
        #     print(s)
        #     print("\n")
            # sptype_preds[s] = glbllbl2_ind["NO_SP"]
            # print(glbllbl2_ind["NO_SP"], sptype2ind["NO_SP"], ind2glbl_lbl[sptype_preds[s]])
            # p = "I"*len(p)
        # elif ind2glbl_lbl[sptype_preds[s]] =="TATLIPO" and p.rfind("T") < len(s)-4 and s[p.rfind("T")+1]!="C":
        #     sptype_preds[s] = glbllbl2_ind["NO_SP"]
        #     p = "I"*len(p)
        # if ind2sptype[sptype_preds[s]] == "LIPO" and p.rfind("L") >= 69:
        #     sptype_preds[s] = sptype2ind["NO_SP"]
        #     p = "I" * len(s)
        # elif ind2sptype[sptype_preds[s]] == "LIPO" and s[p.rfind("L") + 1] != "C":
        #     # if the sp type is also LIPO, print the result
        #     if sp_info == "LIPO":
        #         print(p)
        #         print(t)
        #         print(s)
        #         print("\n")
        #     if "C" in s[p.rfind("L")-3:p.rfind("L")+3]:
        #         new_res = reassign_cs(s,p)
        #         if new_res == -1:
        #             sptype_preds[s] = glbllbl2_ind["NO_SP"]
        #             p = "I" * len(p)
        #         else:
        #             p = new_res
        #
        # elif ind2sptype[sptype_preds[s]] == "TATLIPO" and s[p.rfind("L") + 1] != "C":
        #     sptype_preds[s] = sptype2ind["NO_SP"]
        # if p == "L" and s[p.rfind("L")+1] != "C":
        #     sp_info = "NO_SP"
        #     p = "I" * len(p)
        ind = 0
        predicted_sp = p[0]
        is_sp = predicted_sp in sp_types

        if sp_info == sp_type:
            if p[0] == t[0]:
                count += 1
            else:
                count2 += 1
            #     # the precision was defined as the fraction of correct CS predictions over the number of predicted
            #     # CS, recall as the fraction of correct CS predictions over the number of true CS. In both cases,
            #     # a CS was only considered correct if it was predicted in the correct SP class (e.g. when the
            #     # model predicts a CS in a Sec/SPI sequence, but predicts Sec/SPII as the sequence label, the
            #     # sample is considered as no CS predicted).

            # SO ONLY CONSIDER IT AS A FALSE NEGATIVE FOR THE APPROPRIATE correct-SP class?
            if (sptype_preds is not None and ind2glbl_lbl[sptype_preds[s]] == sp_type) or (sptype_preds is None and sp_letter == p[ind]):
                while (p[ind] == sp_letter or (p[ind] == predicted_sp and is_sp and only_cs_position)) and ind < len(p) - 1:
                    # when only_cs_position=True, the cleavage site positions will be taken into account irrespective of
                    # whether the predicted SP is the correct kind
                    ind += 1
            else:
                ind = 0
            predictions[grp2_ind[lg]] += get_acc_for_tolerence(ind, t, sp_letter)
        # elif sp_info != sp_type and   p[ind] == sp_letter:
        elif (sptype_preds is not None and sp_info != sp_type and ind2glbl_lbl[sptype_preds[s]] == sp_type) or (sptype_preds is None and p[ind] == sp_letter):
            predictions[grp2_ind[lg]] += np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    if v:
        print(" count_tol_fn, count_complete_fn, count_otherSPpred", count_tol_fn, count_complete_fn, count_otherSPpred)
    # prc,rc = cnt1/(cnt1+cnt2), cnt1/ttl
    # print(2*prc*rc/(prc+rc),cnt1,cnt2,ttl)
    # exit(1)
    # print(sp_type, "count, count2", count, count2 )
    # print(cnt1, cnt2)
    all_recalls = []
    all_precisions = []
    all_f1_scores = []
    total_positives = []
    false_positives = []
    for life_grp, ind in grp2_ind.items():
        if sp_type == "SP" or life_grp != "EUKARYA":
            # eukaryotes do not have SEC/SPI or SEC/SPII
            current_preds = predictions[grp2_ind[life_grp]]
            if v:
                print("Recall {}: {}".format(life_grp, [current_preds[i] / current_preds[4] for i in range(4)]))
                print("Prec {}: {}".format(life_grp,
                                           [current_preds[i] / (current_preds[i] + current_preds[5]) for i in
                                            range(4)]))
            all_recalls.append([current_preds[i] / current_preds[4] for i in range(4)])
            all_precisions.append([])
            all_f1_scores.append([])
            for i in range(4):
                if current_preds[5] + current_preds[i] == 0:
                    all_precisions[-1].append(0.)
                else:
                    all_precisions[-1].append(
                        current_preds[i] / (current_preds[i] + current_preds[i + 6]))
            current_recs, current_precs = all_recalls[-1], all_precisions[-1]
            all_f1_scores[-1].extend([0 if current_recs[i] * current_precs[i] == 0 else 2 * current_recs[i] *
                                                                                        current_precs[i] / (
                                                                                                    current_recs[i] +
                                                                                                    current_precs[i])
                                      for i in range(4)])
            total_positives.append(current_preds[4])
            false_positives.append(current_preds[5])
    return all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores


def get_class_sp_accs(life_grp, seqs, true_lbls, pred_lbls):
    groups_tp_tn_fp_fn = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    grp2_ind = {"EUKARYA": 0, "NEGATIVE": 1, "POSITIVE": 2, "ARCHAEA": 3}
    for lg, s, tl, pl in zip(life_grp, seqs, true_lbls, pred_lbls):
        lg = lg.split("|")[0]
        if tl[0] == pl[0] and tl[0] == "S":
            groups_tp_tn_fp_fn[grp2_ind[lg]][0] += 1
        elif tl[0] != "S" and pl[0] != "S":
            groups_tp_tn_fp_fn[grp2_ind[lg]][1] += 1
        elif tl[0] == "S" and pl[0] != "S":
            groups_tp_tn_fp_fn[grp2_ind[lg]][3] += 1
        elif tl[0] != "S" and pl[0] == "S":
            groups_tp_tn_fp_fn[grp2_ind[lg]][2] += 1
    recs = [groups_tp_tn_fp_fn[i][0] / (groups_tp_tn_fp_fn[i][0] + groups_tp_tn_fp_fn[i][3]) if
            groups_tp_tn_fp_fn[i][0] + groups_tp_tn_fp_fn[i][3] != 0 else 0 for i in range(4)]
    precs = [groups_tp_tn_fp_fn[i][0] / (groups_tp_tn_fp_fn[i][0] + groups_tp_tn_fp_fn[i][2]) if
             groups_tp_tn_fp_fn[i][0] + groups_tp_tn_fp_fn[i][2] != 0 else 0 for i in range(4)]
    return [ (2 * recs[i] * precs[i]) / (precs[i] + recs[i]) if precs[i] + recs[i] != 0 else 0 for i in range(4)]


def get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=False, sp_type="SP", sptype_preds=None):
    # S = signal_peptide; T = Tat/SPI or Tat/SPII SP; L = Sec/SPII SP; P = SEC/SPIII SP; I = cytoplasm; M = transmembrane; O = extracellular;
    # order of elemnts in below list:
    # (eukaria_tp, eukaria_tn, eukaria_fp, eukaria_fn)
    # (negative_tp, negative_tn, negative_fp, negative_fn)
    # (positive_tp, positive_tn, positive_fp, positive_fn)
    # (archaea_correct, archaea_total)
    # Matthews correlation coefficient (MCC) both true and false positive and negative predictions are counted at
    # the sequence level
    grp2_ind = {"EUKARYA": 0, "NEGATIVE": 1, "POSITIVE": 2, "ARCHAEA": 3}
    lg2sp_letter = {'TAT': 'T', 'LIPO': 'L', 'PILIN': 'P', 'TATLIPO': 'W', 'SP': 'S'}
    ind2sptype = {0:'NO_SP', 1:'SP', 2:'TATLIPO', 3:'LIPO', 4:'TAT', 5:'PILIN'}
    sptype2ind = {v:k for k,v in ind2sptype.items()}

    sp_type_letter = lg2sp_letter[sp_type]
    predictions = [[[], []], [[], []], [[], []], [[], []]]
    predictions_mcc2 = [[[], []], [[], []], [[], []], [[], []]]
    zv = 0
    for l, s, t, p in zip(life_grp, seqs, true_lbls, pred_lbls):
        zv += 1
        lg, sp_info = l.split("|")
        # if ind2sptype[sptype_preds[s]] == "LIPO" and p.rfind("L") >= 69:
        #     print(p)
        #     print(t)
        #     print(s)
        #     print("\n")
        # elif ind2sptype[sptype_preds[s]] == "LIPO" and s[p.rfind("L") + 1] != "C":
        #     print(p)
        #     print(t)
        #     print(s)
        #     print("\n")


            # if "C" in s[p.rfind("L")-3:p.rfind("L")+3]:
        #         print(p)
        #         print(t)
        #         print(s)
                # new_res = reassign_cs(s,p)
                # if new_res == -1:
                #     sptype_preds[s] = sptype2ind["NO_SP"]
                #     p = "I" * len(p)
                # else:
                #     p = new_res

        # elif ind2sptype[sptype_preds[s]] == "TATLIPO" and s[p.rfind("L") + 1] != "C":
        #     sptype_preds[s] = sptype2ind["NO_SP"]
        # if ind2sptype[sptype_preds[s]] == "LIPO" and s[t.rfind("L")+1] != "C":
        #     sptype_preds[s] = sptype2ind["NO_SP"]
        # elif ind2sptype[sptype_preds[s]] == "TATLIPO" and s[t.rfind("L")+1] != "C":
        #     sptype_preds[s] = sptype2ind["NO_SP"]

        if sp_info == sp_type or sp_info == "NO_SP":
            # if sp_info == "LIPO" and s[p.rfind("L")+1] != "C":
            #     sptype_preds[s] = sptype2ind[sp_info]
            # elif sp_info == "TATLIPO"  and s[p.rfind("L")+1] != "C":
            #     sptype_preds[s] = sptype2ind[sp_info]
            p = p.replace("ES", "J")
            len_ = min(len(p), len(t))
            t, p = t[:len_], p[:len_]
            # for ind in range(len(t[:1])):
            # if t[0] == sp_type_letter and p[0] == sp_type_letter:
            if sp_info == sp_type and ind2sptype[sptype_preds[s]] == sp_info:
                predictions[grp2_ind[lg]][1].append(1)
                predictions[grp2_ind[lg]][0].append(1)
            elif sp_info == sp_type and ind2sptype[sptype_preds[s]] != sp_type:
                # elif t[0] == sp_type_letter and p[0] != sp_type_letter:
                predictions[grp2_ind[lg]][1].append(1)
                predictions[grp2_ind[lg]][0].append(-1)
            # elif t[0] != sp_type_letter and p[0] == sp_type_letter:
            elif sp_info != sp_type and ind2sptype[sptype_preds[s]] == sp_type:
                predictions[grp2_ind[lg]][1].append(-1)
                predictions[grp2_ind[lg]][0].append(1)
            # elif t[0] != sp_type_letter and p[0] != sp_type_letter:
            elif sp_info != sp_type and ind2sptype[sptype_preds[s]] != sp_type:
                predictions[grp2_ind[lg]][1].append(-1)
                predictions[grp2_ind[lg]][0].append(-1)
        if return_mcc2:
            p = p.replace("ES", "J")
            len_ = min(len(p), len(t))
            t, p = t[:len_], p[:len_]
            # for ind in range(len(t[:1])):
            # if t[0] == sp_type_letter and p[0] == sp_type_letter:
            if sp_info == sp_type and ind2sptype[sptype_preds[s]] == sp_info:
                predictions_mcc2[grp2_ind[lg]][1].append(1)
                predictions_mcc2[grp2_ind[lg]][0].append(1)
            # elif t[0] == sp_type_letter and p[0] != sp_type_letter:
            elif sp_info == sp_type and ind2sptype[sptype_preds[s]] != sp_type:
                predictions_mcc2[grp2_ind[lg]][1].append(1)
                predictions_mcc2[grp2_ind[lg]][0].append(-1)
            # elif t[0] != sp_type_letter and p[0] == sp_type_letter:
            elif sp_info != sp_type and ind2sptype[sptype_preds[s]] == sp_type:
                predictions_mcc2[grp2_ind[lg]][1].append(-1)
                predictions_mcc2[grp2_ind[lg]][0].append(1)
            elif sp_info != sp_type and ind2sptype[sptype_preds[s]] != sp_type:
                # elif t[0] != sp_type_letter and p[0] != sp_type_letter:
                predictions_mcc2[grp2_ind[lg]][1].append(-1)
                predictions_mcc2[grp2_ind[lg]][0].append(-1)
    mccs, mccs2 = [], []
    for grp, id in grp2_ind.items():
        if sp_type == "SP" or grp != "EUKARYA":
            if sum(predictions[grp2_ind[grp]][0]) == -len(predictions[grp2_ind[grp]][0]) or \
                    sum(predictions[grp2_ind[grp]][0]) == len(predictions[grp2_ind[grp]][0]):
                mccs.append(-1)
            else:
                mccs.append(compute_mcc(predictions[grp2_ind[grp]][0]
                                        , predictions[grp2_ind[grp]][1]))
            if v:
                print("{}: {}".format(grp, mccs[-1]))
    if return_mcc2:
        for grp, id in grp2_ind.items():
            if sp_type == "SP" or grp != "EUKARYA":

                if sum(predictions_mcc2[grp2_ind[grp]][0]) == -len(predictions_mcc2[grp2_ind[grp]][0]) or \
                        sum(predictions_mcc2[grp2_ind[grp]][0]) == len(predictions_mcc2[grp2_ind[grp]][0]):
                    mccs2.append(-1)
                else:
                    mccs2.append(compute_mcc(predictions_mcc2[grp2_ind[grp]][0]
                                             , predictions_mcc2[grp2_ind[grp]][1]))
                if v:
                    print("{}: {}".format(grp, mccs2[-1]))
        return mccs, mccs2
    return mccs


def get_bin(p, bins):
    for i in range(len(bins)):
        if bins[i] < p <= bins[i + 1]:
            return i


def get_cs_preds_by_tol(tl, pl):
    pl = pl.replace("ES", "")
    correct_by_tol = [0, 0, 0, 0]
    for tol in range(4):
        correct_by_tol[tol] = int(tl.rfind("S") - tol <= pl.rfind("S") <= tl.rfind("S") + tol)
    return correct_by_tol


def plot_all_reliability_diagrams(resulted_perc_by_acc, name, total_counts_per_acc, ece):
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"
    fig, ax = plt.subplots(2, 2)
    import pylab as pl
    from matplotlib.ticker import FixedLocator, FormatStrFormatter
    for i in range(4):
        accs = [acc_to_perc[0] for acc_to_perc in resulted_perc_by_acc[i]]
        total_counts_per_acc_ = list(total_counts_per_acc[i])
        percs = [acc_to_perc[1] for acc_to_perc in resulted_perc_by_acc[i]]
        bars_width = accs[0] - accs[1]
        # ax[i//2, i%2].set_title(name[i], fontsize=12.5)
        if i == 0:
            box = ax[i//2, i%2].get_position()
            ax[i//2, i%2].set_position([box.x0, box.y0 + box.height * 0.5, box.width * 1.15, box.height * 0.65])
            ax[i//2, i%2].bar(accs, accs, width=bars_width, alpha=0.5, linewidth=2, edgecolor="black", color='blue',
                    label='Perfect calibration')
            ax[i//2, i%2].bar(accs, percs, width=bars_width, alpha=0.5, color='red', label="Model's calibration")
            ax[i//2, i%2].set_xticks([round(accs[j],2) for j in range(len(accs))])
            ax[i//2, i%2].set_xticklabels(["{}/{}".format(str(round(accs[j], 2)), str(total_counts_per_acc_[j])) for j in
               range(len(accs))],fontsize=6, rotation=300)
            ax[i // 2, i % 2].set_xlim([0,1])
            ax[i // 2, i % 2].set_title("tol {}, ece: {}".format(i, ece[i]), fontsize=12.5)
            # ax[i//2, i%2].set_xticklabels(["{}\n{}".format(str(round(accs[i], 2)), str(total_counts_per_acc[i])) for i in
            #             range(len(accs)//2 - 2)], fontsize=12.5)
            ax[i//2, i%2].set_yticks([0, 0.5, 1])
            ax[i//2, i%2].set_yticklabels([0, 0.5, 1], fontsize=6)
        else:
            box = ax[i//2, i%2].get_position()
            horizontal_offset = 0.1 * box.width if i == 1 or i == 3 else 0
            vertical_offset = 0.5 * box.height if i == 1 else 0.5 * box.height
            ax[i//2, i%2].set_position([box.x0 + horizontal_offset, box.y0 + vertical_offset, box.width * 1.15, box.height * 0.65])
            ax[i // 2, i % 2].bar(accs, accs, width=bars_width, alpha=0.5, linewidth=2, edgecolor="black", color='blue')
            ax[i // 2, i % 2].bar(accs, percs, width=bars_width, alpha=0.5, color='red')
            # ax[i//2, i%2].set_xticks([round(accs[j*2+1],2) for j in range(len(accs)//2)])
            # ax[i//2, i%2].set_xticklabels(["{}\n{}".format(str(round(accs[j * 2 + 1], 2)), str(total_counts_per_acc_[j * 2 + 1])) for j in
            #    range(len(accs) // 2 )],fontsize=6)
            ax[i//2, i%2].set_xticks([round(accs[j],2) for j in range(len(accs))])
            ax[i//2, i%2].set_xticklabels(["{}/{}".format(str(round(accs[j], 2)), str(total_counts_per_acc_[j])) for j in
               range(len(accs))],fontsize=6, rotation=300)

            ax[i // 2, i % 2].set_xlim([0,1])
            ax[i // 2, i % 2].set_title("tol {}, ece: {}".format(i, ece[i]), fontsize=12.5)
            ax[i//2, i%2].set_yticks([0, 0.5, 1])
            ax[i//2, i%2].set_yticklabels([0, 0.5, 1], fontsize=6)

        if i > 1:
            ax[i//2, i%2].set_xlabel("Confidence/No of samples", fontsize=12.5)
            ax[i//2, i%2].xaxis.set_label_coords(0.5, -0.6)

        if i == 0 or i == 2:
            ax[i//2, i%2].set_ylabel("Accuracy", fontsize=12.5)
    fig.legend(loc='center left', bbox_to_anchor=(0.2, 0.05), fontsize=12.5, ncol=2)
    plt.show()

def plot_reliability_diagrams(resulted_perc_by_acc, name, total_counts_per_acc):

    import matplotlib as mpl
    fig = mpl.pyplot.gcf()
    mpl.rcParams['figure.dpi'] = 350
    fig.set_size_inches(12, 8)

    accs = [acc_to_perc[0] for acc_to_perc in resulted_perc_by_acc]
    total_counts_per_acc = list(total_counts_per_acc)
    percs = [acc_to_perc[1] for acc_to_perc in resulted_perc_by_acc]
    bars_width = accs[0] - accs[1]
    plt.title(name, fontsize=26)
    plt.bar(accs, accs, width=bars_width, alpha=0.5, linewidth=2, edgecolor="black", color='blue',
            label='Perfect calibration')
    plt.bar(accs, percs, width=bars_width, alpha=0.5, color='red', label="Model's calibration",
            tick_label=["{}\n{}".format(str(round(accs[i], 2)), str(total_counts_per_acc[i])) for i in
                        range(len(accs))])
    plt.xlabel("Prob/No of preds")
    plt.ylabel("Prob")
    plt.legend()
    plt.show()


def get_prob_calibration_and_plot(probabilities_file="", life_grp=None, seqs=None, true_lbls=None, pred_lbls=None,
                                  bins=15, plot=True, sp2probs=None, plot_together=True):
    # initialize bins
    bin_limmits = np.linspace(0, 1, bins)
    correct_calibration_accuracies = [(bin_limmits[i] + bin_limmits[i + 1]) / 2 for i in range(bins - 1)]

    # initialize cleavage site and binary sp dictionaries of the form:
    # binary_sp_dict: {<life_group> : { total : { <accuracy> : count }, correct : { <accuracy> : count} } }
    # cs_pred: {<life_group> : { <tol> : { total : {<accuracy> : count}, correct: {<accuracy> : count} } } }
    binary_sp_calibration_by_grp = {}
    cs_by_lg_and_tol_accs = {}
    for lg in life_grp:
        lg = lg.split("|")[0]
        crct_cal_acc_2_correct_preds = {crct_cal_acc: 0 for crct_cal_acc in correct_calibration_accuracies}
        crct_cal_acc_2_totals = {crct_cal_acc: 0 for crct_cal_acc in correct_calibration_accuracies}
        wrap_dict = {'total': crct_cal_acc_2_totals, 'correct': crct_cal_acc_2_correct_preds}
        binary_sp_calibration_by_grp[lg] = wrap_dict
        tol_based_cs_accs = {}
        for tol in range(4):
            crct_cal_acc_2_correct_preds = {crct_cal_acc: 0 for crct_cal_acc in correct_calibration_accuracies}
            crct_cal_acc_2_totals = {crct_cal_acc: 0 for crct_cal_acc in correct_calibration_accuracies}
            wrap_dict = {'total': crct_cal_acc_2_totals, 'correct': crct_cal_acc_2_correct_preds}
            tol_based_cs_accs[tol] = wrap_dict
        cs_by_lg_and_tol_accs[lg] = tol_based_cs_accs

    sp2probs = pickle.load(open(probabilities_file, "rb")) if sp2probs is None else sp2probs
    for s, tl, pl, lg in zip(seqs, true_lbls, pred_lbls, life_grp):
        lg = lg.split("|")[0]
        predicted_sp_prob, all_sp_probs, _ = sp2probs[s]
        bin = get_bin(predicted_sp_prob, bin_limmits)
        coresp_acc = correct_calibration_accuracies[bin]
        if tl[0] == "S":
            binary_sp_calibration_by_grp[lg]['total'][coresp_acc] += 1
            if pl[0] == "S":
                binary_sp_calibration_by_grp[lg]['correct'][coresp_acc] += 1
        if tl[0] == pl[0] == "S":
            # in order to extract lower confidence probabilities, move left and right from the cleave site prediction
            # in a [-3, 3] window. See if 1-SP_conf_pred (which should be approx = CS pred) correlates with the
            # accuracy of the CS prediction
            # TODO it may be possible for cleavage site predictions to continue predicting S (sp signal) even after
            #  the predominant-sp probability is finished. And retrieve those as [0,0.5] probabilities
            for i in range(-3, 4):
                bin_cs = get_bin(1-all_sp_probs[pl.rfind("S") + 1], bin_limmits)
                coresp_acc_cs = correct_calibration_accuracies[bin_cs]
                # if 0.5  <= 1- all_sp_probs[pl.rfind("S") + 1] <= 0.8:
                #     print(1-np.array(all_sp_probs[pl.rfind("S")-3:pl.rfind("S")+3]))
                # print(coresp_acc_cs)
            correct_preds_by_tol = get_cs_preds_by_tol(tl, pl)
            for tol in range(4):
                cs_by_lg_and_tol_accs[lg][tol]['correct'][coresp_acc_cs] += correct_preds_by_tol[tol]
                cs_by_lg_and_tol_accs[lg][tol]['total'][coresp_acc_cs] += 1
        elif pl[0] == "S" and tl[0] != "S":
            binary_sp_calibration_by_grp[lg]['total'][coresp_acc] += 1
    binary_ece, cs_ece = [], [[], [], [], []]
    lg_and_tol2_lg = {}
    for lg_ind, lg in enumerate(['EUKARYA', 'NEGATIVE', 'POSITIVE', 'ARCHAEA']):
        correct_binary_preds, total_binary_preds = binary_sp_calibration_by_grp[lg]['correct'].values(), \
                                                   binary_sp_calibration_by_grp[lg]['total'].values()
        results = []
        current_binary_ece = []
        for ind, (crct, ttl) in enumerate(zip(correct_binary_preds, total_binary_preds)):
            actual_acc = crct / ttl if ttl != 0 else 0
            results.append((correct_calibration_accuracies[ind], actual_acc if ttl != 0 else 0))
            current_binary_ece.append(
                np.abs(correct_calibration_accuracies[ind] - actual_acc) * (ttl / sum(total_binary_preds)))
        binary_ece.append(round(sum(current_binary_ece), 3))
        if plot:# and not plot_together:
            print("Binary preds for {} with ECE {}: ".format(lg, sum(current_binary_ece)), results, total_binary_preds)
            plot_reliability_diagrams(results, "Binary sp pred results for {} with ECE {}".format(lg, round(
                sum(current_binary_ece), 3)), total_binary_preds)
        all_results, all_titles, all_total_cs_preds= [], [], []
        all_cs_ece = []
        for tol in range(4):
            correct_cs_preds, total_cs_preds = cs_by_lg_and_tol_accs[lg][tol]['correct'].values(), \
                                               cs_by_lg_and_tol_accs[lg][tol]['total'].values()
            results = []
            # this is the binary sp ece
            current_cs_ece = []
            # this will be computed according to the 1-noSP probability at the cleavage site
            # current_actual_cs_ece = []
            for ind, (crct, ttl) in enumerate(zip(correct_cs_preds, total_cs_preds)):
                results.append((correct_calibration_accuracies[ind], crct / ttl if ttl != 0 else 0))
                actual_acc = crct / ttl if ttl != 0 else 0
                current_cs_ece.append(
                    np.abs(correct_calibration_accuracies[ind] - actual_acc) * (ttl / sum(total_binary_preds)))
            cs_ece[lg_ind].append(round(sum(current_cs_ece), 3))
            lg_and_tol2_lg["{}_{}".format(lg, tol)] = sum(current_cs_ece)
            if plot and not plot_together:
                plot_reliability_diagrams(results, "CS pred results for tol {} for {} with ECE {}".format(tol, lg,
                                      round(sum(current_cs_ece),3)),total_cs_preds)
                print("Cs preds for {} for tol {}:".format(lg, tol), results)
            if plot_together:
                all_results.append(results)
                all_titles.append("{}{}: {}".format(lg[0], tol, round(sum(current_cs_ece),3)))
                all_total_cs_preds.append(total_cs_preds)
            all_cs_ece.append(current_cs_ece)
        if plot_together:
            plot_all_reliability_diagrams(all_results, all_titles, all_total_cs_preds, ece=[round(sum(all_cs_ece[j]),3) for j in range(4)])
    return lg_and_tol2_lg

def extract_seq_group_for_predicted_aa_lbls(filename="run_wo_lg_info.bin", test_fold=2, dict_=None, benchmark=True):
    seq2preds = pickle.load(open(filename, "rb")) if dict_ is None else dict_
    tested_seqs = set(seq2preds.keys())
    seq2id = {}
    life_grp, seqs, true_lbls, pred_lbls = [], [], [], []
    info_dictionary = {}
    for t in ['train','test']:
        for tf in [0, 1,2]:
            file = get_data_folder()+"sp6_partitioned_data_{}_{}.bin".format(t,tf)
            for seq, data in pickle.load(open(file, "rb")).items():
                info_dictionary[seq] = ["|".join([data[-2],data[-1]]), data[-3]]

    for seq_record in SeqIO.parse(get_data_folder() + "sp6_data/train_set.fasta", "fasta"):
        seq, lbls = seq_record.seq[:len(seq_record.seq) // 2], seq_record.seq[len(seq_record.seq) // 2:]
        if seq in tested_seqs and str(seq) not in seqs:
            life_grp.append(info_dictionary[str(seq)][0])
            seqs.append(str(seq))
            true_lbls.append(info_dictionary[str(seq)][1])
            pred_lbls.append(seq2preds[str(seq)])
            # life_grp.append("|".join(str(seq_record.id).split("|")[1:-1]))
            # seqs.append(seq)
            # true_lbls.append(lbls)
            # pred_lbls.append(seq2preds[seq])
    return life_grp, seqs, true_lbls, pred_lbls


def get_data_folder():
    if os.path.exists("sp6_data/"):
        return "./"
    elif os.path.exists("results"):
        return "../sp_data/"


def get_cs_and_sp_pred_results(filename="run_wo_lg_info.bin", v=False, probabilities_file=None, return_everything=False,
                               return_class_prec_rec=False,):
    sptype_filename = filename.replace(".bin", "")  + "_sptype.bin"
    if os.path.exists(sptype_filename):
        sptype_preds = pickle.load(open(sptype_filename, "rb"))
    else:
        sptype_preds = None
    life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename=filename)
    if probabilities_file is not None:
        get_prob_calibration_and_plot(probabilities_file, life_grp, seqs, true_lbls, pred_lbls)
    sp_pred_mccs = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=v, sptype_preds=sptype_preds)
    all_recalls, all_precisions, total_positives, \
    false_positives, predictions, all_f1_scores = get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=v, sptype_preds=sptype_preds)
    if return_everything:
        sp_pred_mccs, sp_pred_mccs2 = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=v,
                                                               return_mcc2=True, sp_type="SP", sptype_preds=sptype_preds)
        lipo_pred_mccs, lipo_pred_mccs2 = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=v,
                                                                   return_mcc2=True, sp_type="LIPO", sptype_preds=sptype_preds)
        tat_pred_mccs, tat_pred_mccs2 = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=v,
                                                                 return_mcc2=True, sp_type="TAT", sptype_preds=sptype_preds)

        all_recalls_lipo, all_precisions_lipo, _, _, _, all_f1_scores_lipo = get_cs_perf(life_grp, seqs, true_lbls,
                                                                                        pred_lbls, v=False,
                                                                                        only_cs_position=False,
                                                                                        sp_type="LIPO", sptype_preds=sptype_preds)
        all_recalls_tat, all_precisions_tat, _, _, _, all_f1_scores_tat = get_cs_perf(life_grp, seqs, true_lbls,
                                                                                     pred_lbls, v=False,
                                                                                     only_cs_position=False,
                                                                                     sp_type="TAT", sptype_preds=sptype_preds)
        if return_class_prec_rec:
            return sp_pred_mccs, sp_pred_mccs2, lipo_pred_mccs, lipo_pred_mccs2, tat_pred_mccs, tat_pred_mccs2, \
                   all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat, all_f1_scores_lipo, all_f1_scores_tat, \
                   all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores, \
                   get_class_sp_accs(life_grp, seqs, true_lbls, pred_lbls)
        return sp_pred_mccs, sp_pred_mccs2, lipo_pred_mccs, lipo_pred_mccs2, tat_pred_mccs, tat_pred_mccs2, \
               all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat, all_f1_scores_lipo, all_f1_scores_tat, \
               all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores
    if return_class_prec_rec:
        return sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores, \
               get_class_sp_accs(life_grp, seqs, true_lbls, pred_lbls)
    return sp_pred_mccs, all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores


def get_summary_sp_acc(sp_pred_accs):
    return np.mean(sp_pred_accs), sp_pred_accs[0]


def get_summary_cs_acc(all_cs_preds):
    return np.mean(np.array(all_cs_preds)), np.mean(all_cs_preds[0]), all_cs_preds[0][0]


def plot_losses(losses, name="param_search_0.2_2048_0.0001_"):
    train_loss, valid_loss = losses
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    axs.set_title("Train and validation loss over epochs")
    axs.plot(train_loss, label="Train loss")
    axs.plot(valid_loss, label="Validation loss")
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Loss")
    axs.legend()
    axs.set_ylim(0, 0.2)
    # plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/" + name + "loss.png")
    plt.show()


def plot_mcc(mccs, name="param_search_0.2_2048_0.0001_"):
    euk_mcc, neg_mcc, pos_mcc, arc_mcc = mccs
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(euk_mcc, label="Eukaryote mcc")
    axs[0, 0].set_ylabel("mcc")
    axs[0, 0].set_ylim(-1.1, 1.1)
    axs[0, 0].legend()

    axs[0, 1].plot(neg_mcc, label="Negative mcc")
    axs[0, 1].set_ylim(-1.1, 1.1)
    axs[0, 1].legend()

    axs[1, 0].plot(pos_mcc, label="Positive mcc")
    axs[1, 0].legend()
    axs[1, 0].set_ylim(-1.1, 1.1)
    axs[1, 0].set_ylabel("mcc")

    axs[1, 0].set_xlabel("epochs")

    axs[1, 1].plot(arc_mcc, label="Archaea mcc")
    axs[1, 1].legend()

    axs[1, 1].set_ylim(-1.1, 1.1)
    axs[1, 1].set_xlabel("epochs")
    # plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/{}_{}.png".format(name, "mcc"))
    plt.show()


def extract_and_plot_prec_recall(results, metric="recall", name="param_search_0.2_2048_0.0001_", sp_type_f1=[[]]):
    cs_res_euk, cs_res_neg, cs_res_pos, cs_res_arc = results
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"
    fig, axs = plt.subplots(2, 2)

    for i in range(4):
        if i == 0:
            for j in range(4):
                box = axs[j//2, j%2].get_position()
                axs[j//2, j%2].set_position([box.x0, box.y0 + box.height * 0.35, box.width * 1, box.height * 0.75])

        if i == 0 and metric == "f1"  and len(sp_type_f1[0]) != 0:
            axs[0, 0].plot(sp_type_f1[0], color='black', label="SP type F1", linewidth=0.8)
        axs[0, 0].plot(cs_res_euk[i], label="CS tol={}".format(i), linewidth=0.8)
        axs[0, 0].set_ylabel(metric, fontsize=12.5)
        # axs[0, 0].legend()
        axs[0, 0].set_ylim(-0.1, 1.1)
        axs[0,0].set_yticks([0,0.2,0.4,0.6,0.8,1])
        axs[0,0].set_yticklabels([0,0.2,0.4,0.6,0.8,1], fontsize=12.5)
        axs[0, 0].set_xticks([0, 25, 50, 75])
        axs[0, 0].set_xticklabels([0, 25, 50, 75], fontsize=12.5)
        axs[0, 0].set_title("eukarya", fontsize=12.5)
        if i == 0:
            axs[0, 0].annotate("SWA start", xy=(60, cs_res_euk[0][60]), xytext=(50, 0.001),
                            arrowprops=dict(arrowstyle="->", linewidth=0.5), fontsize=12.5, color='black')

        axs[0, 1].plot(cs_res_neg[i], linewidth=0.8)
        if i == 0 and metric == "f1" and len(sp_type_f1[0]) != 0:
            axs[0, 1].plot(sp_type_f1[1], color='black', linewidth=0.8)
        # axs[0, 1].legend()
        axs[0, 1].set_ylim(-0.1, 1.1)
        if i == 0:
            axs[0, 1].annotate("SWA start", xy=(60, cs_res_neg[0][60]), xytext=(50, 0.001),
                            arrowprops=dict(arrowstyle="->", linewidth=0.5), fontsize=12.5, color='black')
        axs[0, 1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axs[0, 1].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
        axs[0, 1].set_xticks([0, 25, 50, 75])
        axs[0, 1].set_xticklabels([0, 25, 50, 75], fontsize=12.5)
        axs[0, 1].set_title("gn bacteria", fontsize=12.5)

        axs[1, 0].plot(cs_res_pos[i], linewidth=0.8)
        if i == 0  and metric == "f1"  and len(sp_type_f1[0]) != 0:
            axs[1, 0].plot(sp_type_f1[2], color='black', linewidth=0.8)
        # axs[1, 0].legend()
        axs[1, 0].set_xlabel("epoch", fontsize=12.5)
        axs[1, 0].set_ylim(-0.1, 1.1)
        axs[1, 0].set_ylabel(metric, fontsize=12.5)
        if i == 0:
            axs[1, 0].annotate("SWA start", xy=(60, cs_res_pos[0][60]), xytext=(50, 0.001),
                            arrowprops=dict(arrowstyle="->", linewidth=0.5), fontsize=12.5, color='black')
        axs[1, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axs[1, 0].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
        axs[1, 0].set_xticks([0, 25, 50, 75])
        axs[1, 0].set_xticklabels([0, 25, 50, 75], fontsize=12.5)
        axs[1, 0].set_title("gp bacteria", fontsize=12.5)

        axs[1, 1].plot(cs_res_arc[i], linewidth=0.8)
        if i == 0  and metric == "f1" and len(sp_type_f1[0]) != 0:
            axs[1, 1].plot(sp_type_f1[3], color='black', linewidth=0.8)
        # axs[1, 1].legend()
        axs[1, 1].set_xlabel("epoch", fontsize=12.5)
        axs[1, 1].set_ylim(-0.1, 1.1)
        if i == 0:
            axs[1, 1].annotate("SWA start", xy=(60, cs_res_arc[0][60]), xytext=(50, 0.001),
                            arrowprops=dict(arrowstyle="->", linewidth=0.5), fontsize=12.5, color='black')
        axs[1, 1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axs[1, 1].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
        axs[1, 1].set_xticks([0, 25, 50, 75])
        axs[1, 1].set_xticklabels([0, 25, 50, 75], fontsize=12.5)
        axs[1, 1].set_title("archaea", fontsize=12.5)

    fig.legend(loc='center left', bbox_to_anchor=(0.01, 0.05), ncol=5, fontsize=12.5)

    # plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/{}_{}.png".format(name, metric))
    plt.show()


def visualize_validation(run="param_search_0.2_2048_0.0001_", folds=[0, 1], folder=""):

    all_results = []
    euk_mcc, neg_mcc, pos_mcc, arc_mcc, train_loss, valid_loss, cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, \
    cs_recalls_arc, cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc, sp_type_f1 = extract_results(run, folds=folds,
                                                                                                folder=folder)
    best_vl = 20
    patience = 0
    best_patience = 0
    for v_l in valid_loss:
        if v_l < best_vl:
            best_patience = patience
            best_vl = v_l
        else:
            patience -= 1
    print(best_vl, best_patience)
    all_f1 = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    for lg_ind, (lg_rec, lg_prec) in enumerate([(cs_recalls_euk, cs_precs_euk), (cs_recalls_neg, cs_precs_neg),
                                                (cs_recalls_pos, cs_precs_pos), (cs_recalls_arc, cs_precs_arc)]):
        for tol in range(4):
            for prec, rec in zip(lg_rec[tol], lg_prec[tol]):
                all_f1[lg_ind][tol].append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    cs_f1_euk, cs_f1_neg, cs_f1_pos, cs_f1_arc = all_f1
    plot_mcc([euk_mcc, neg_mcc, pos_mcc, arc_mcc], name=run)
    plot_losses([train_loss, valid_loss], name=run)
    extract_and_plot_prec_recall([cs_f1_euk, cs_f1_neg, cs_f1_pos, cs_f1_arc], metric="f1", name=run, sp_type_f1=sp_type_f1)
    extract_and_plot_prec_recall([cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, cs_recalls_arc], metric="recall",
                                 name=run)
    extract_and_plot_prec_recall([cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc], metric="precision", name=run)
    # extract_and_plot_losses(lines)


def extract_results(run="param_search_0.2_2048_0.0001_", folds=[0, 1], folder='results_param_s_2/'):
    euk_mcc, neg_mcc, pos_mcc, arc_mcc = [], [], [], []
    train_loss, valid_loss = [], []
    cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, cs_recalls_arc = [[], [], [], []], [[], [], [], []], \
                                                                     [[], [], [], []], [[], [], [], []]
    cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc = [[], [], [], []], [[], [], [], []], \
                                                             [[], [], [], []], [[], [], [], []]
    class_preds = [[], [], [], []]
    with open(folder + run + "{}_{}.log".format(folds[0], folds[1]), "rt") as f:
        lines = f.readlines()
    for l in lines:
        if "sp_pred mcc" in l and "VALIDATION" in l:
            mccs = l.split(":")[-1].replace(" ", "").split(",")
            euk_mcc.append(float(mccs[0]))
            neg_mcc.append(float(mccs[1]))
            pos_mcc.append(float(mccs[2]))
            arc_mcc.append(float(mccs[3]))
        elif "train/validation" in l:
            train_l, valid_l = l.split(":")[-1].replace(" ", "").split("/")
            valid_l = valid_l.split(",")[0].replace(" ", "")
            train_l, valid_l = float(train_l), float(valid_l)
            train_loss.append(train_l)
            valid_loss.append(valid_l)
        elif "cs recall" in l and "VALIDATION" in l:
            cs_res = l.split(":")[-1].replace(" ", "").split(",")
            cs_res = [float(c_r) for c_r in cs_res]
            cs_recalls_euk[0].append(cs_res[0])
            cs_recalls_euk[1].append(cs_res[1])
            cs_recalls_euk[2].append(cs_res[2])
            cs_recalls_euk[3].append(cs_res[3])

            cs_recalls_neg[0].append(cs_res[4])
            cs_recalls_neg[1].append(cs_res[5])
            cs_recalls_neg[2].append(cs_res[6])
            cs_recalls_neg[3].append(cs_res[7])

            cs_recalls_pos[0].append(cs_res[8])
            cs_recalls_pos[1].append(cs_res[9])
            cs_recalls_pos[2].append(cs_res[10])
            cs_recalls_pos[3].append(cs_res[11])

            cs_recalls_arc[0].append(cs_res[12])
            cs_recalls_arc[1].append(cs_res[13])
            cs_recalls_arc[2].append(cs_res[14])
            cs_recalls_arc[3].append(cs_res[15])

        elif "cs precision" in l and "VALIDATION" in l:
            prec_res = l.split(":")[-1].replace(" ", "").split(",")
            prec_res = [float(c_r) for c_r in prec_res]
            cs_precs_euk[0].append(prec_res[0])
            cs_precs_euk[1].append(prec_res[1])
            cs_precs_euk[2].append(prec_res[2])
            cs_precs_euk[3].append(prec_res[3])

            cs_precs_neg[0].append(prec_res[4])
            cs_precs_neg[1].append(prec_res[5])
            cs_precs_neg[2].append(prec_res[6])
            cs_precs_neg[3].append(prec_res[7])

            cs_precs_pos[0].append(prec_res[8])
            cs_precs_pos[1].append(prec_res[9])
            cs_precs_pos[2].append(prec_res[10])
            cs_precs_pos[3].append(prec_res[11])

            cs_precs_arc[0].append(prec_res[12])
            cs_precs_arc[1].append(prec_res[13])
            cs_precs_arc[2].append(prec_res[14])
            cs_precs_arc[3].append(prec_res[15])
        elif "F1Score:" in l and "VALIDATION" in l:
            vals = l.split("F1Score:")[-1].replace(",","").split(" ")
            vals = [v for v in vals if v != '']
            vals = [float(v.replace("\n", "")) for v in vals]
            class_preds[0].append(float(vals[0]))
            class_preds[1].append(float(vals[1]))
            class_preds[2].append(float(vals[2]))
            class_preds[3].append(float(vals[3]))


    # fix for logs that have f1 score written as "precision"...
    if len(cs_precs_pos[0]) == 2 * len(cs_recalls_pos[0]):
        len_cs_rec = len(cs_recalls_pos[0])
        for j in range(4):
            cs_precs_euk[j], cs_precs_neg[j], cs_precs_pos[j], cs_precs_arc[j] = [cs_precs_euk[j][i * 2] for i in
                                                                                  range(len_cs_rec)], \
                                                                                 [cs_precs_neg[j][i * 2] for i in
                                                                                  range(len_cs_rec)], \
                                                                                 [cs_precs_pos[j][i * 2] for i in
                                                                                  range(len_cs_rec)], \
                                                                                 [cs_precs_arc[j][i * 2] for i in
                                                                                  range(len_cs_rec)]

    return euk_mcc, neg_mcc, pos_mcc, arc_mcc, train_loss, valid_loss, cs_recalls_euk, cs_recalls_neg, cs_recalls_pos, \
           cs_recalls_arc, cs_precs_euk, cs_precs_neg, cs_precs_pos, cs_precs_arc, class_preds


def remove_from_dictionary(res_dict, test_fld):
    tb_removed = pickle.load(open("../sp_data/sp6_partitioned_data_test_{}.bin".format(test_fld[0]), "rb"))
    tb_removed = tb_removed.keys()
    trimmed_res_dict = {}

    for seq, res in res_dict.items():
        if seq not in tb_removed:
            trimmed_res_dict[seq] = res
    return trimmed_res_dict


def extract_mean_test_results(run="param_search_0.2_2048_0.0001", result_folder="results_param_s_2/",
                              only_cs_position=False, remove_test_seqs=False, return_sptype_f1=False, benchmark=True,
                              restrict_types=None):
    full_dict_results = {}
    full_sptype_dict = {}
    epochs = []
    for tr_folds in [[0, 1], [1, 2], [0, 2]]:
        with open(os.path.join(result_folder, run + "_{}_{}.log".format(tr_folds[0], tr_folds[1])), "rt") as f:
            lines = f.readlines()
            try:
                epochs.append(int(lines[-2].split(" ")[2]))
            except:
                epochs.append(int(lines[-2].split(":")[-3].split(" ")[-1]))
    avg_epoch = np.mean(epochs)
    id2seq, id2lg, id2type, id2truelbls = extract_id2seq_dict()
    seq2id = {v:k for k,v in id2seq.items()}
    unique_bench_seqs = set(id2seq.values())
    seq2type = {}
    for id, seq in id2seq.items():
        seq2type[seq] = id2type[id]
    for tr_folds in [[0, 1], [1, 2], [0, 2]]:
        res_dict = pickle.load(open(os.path.join(result_folder, run + "_{}_{}_best.bin".format(tr_folds[0], tr_folds[1])), "rb"))
        if benchmark:
            res_dict = {k:v for k,v in res_dict.items() if k in unique_bench_seqs}
        if restrict_types is not None:
            res_dict = {k:v for k,v in res_dict.items() if seq2type[k] in restrict_types}
        if os.path.exists(os.path.join(result_folder, run + "_{}_{}_best_sptype.bin".format(tr_folds[0], tr_folds[1]))):
            sptype_dict = pickle.load(open(os.path.join(result_folder, run + "_{}_{}_best_sptype.bin".format(tr_folds[0], tr_folds[1])), "rb"))
            full_sptype_dict.update(sptype_dict)
        else:
            full_sptype_dict = None
        test_fld = list({0, 1, 2} - set(tr_folds))
        if remove_test_seqs:
            full_dict_results.update(remove_from_dictionary(res_dict, test_fld))
        else:
            full_dict_results.update(res_dict)
    print(len(full_dict_results.keys()), len(set(full_dict_results.keys())), len(unique_bench_seqs))
    # exit(1)

    life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename="w_lg_w_glbl_lbl_100ep.bin",
                                                                                   dict_=full_dict_results)
    mccs, mccs2 = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                           sp_type="SP", sptype_preds=full_sptype_dict)
    # LIPO is SEC/SPII
    mccs_lipo, mccs2_lipo = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                                     sp_type="LIPO", sptype_preds=full_sptype_dict)
    # TAT is TAT/SPI
    mccs_tat, mccs2_tat = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                                   sp_type="TAT", sptype_preds=full_sptype_dict)
    # TATLIPO Tat/SPase II
    mccs_tatlipo, mccs2_tatlipo = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                                   sp_type="TATLIPO", sptype_preds=full_sptype_dict)
    # PILIN Sec/SPase IV
    mccs_pilin, mccs2_pilin = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                                   sp_type="PILIN", sptype_preds=full_sptype_dict)
    if "param_search_w_nl_nh_0.0_4096_1e-05_4_4" in run:
        v = False
    else:
        v = False
    all_recalls, all_precisions, _, _, _, f1_scores = \
        get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=v, only_cs_position=only_cs_position, sp_type="SP", sptype_preds=full_sptype_dict)
    all_recalls_lipo, all_precisions_lipo, _, _, _, f1_scores_lipo = \
        get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=v, only_cs_position=only_cs_position, sp_type="LIPO", sptype_preds=full_sptype_dict )
    all_recalls_tat, all_precisions_tat, _, _, _, f1_scores_tat = \
        get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=v, only_cs_position=only_cs_position, sp_type="TAT", sptype_preds=full_sptype_dict)
    all_recalls_tatlipo, all_precisions_tatlipo, _, _, _, f1_scores_tatlipo = \
        get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=v, only_cs_position=only_cs_position, sp_type="TATLIPO",
                   sptype_preds=full_sptype_dict)
    all_recalls_pilin, all_precisions_pilin, _, _, _, f1_scores_pilin = \
        get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=v, only_cs_position=only_cs_position, sp_type="PILIN",
                   sptype_preds=full_sptype_dict)
    if return_sptype_f1:
        return mccs, mccs2, mccs_lipo, mccs2_lipo, mccs_tat, mccs2_tat, mccs_tatlipo, mccs2_tatlipo, mccs_pilin, mccs2_pilin, all_recalls, all_precisions, all_recalls_lipo, \
               all_precisions_lipo, all_recalls_tat, all_precisions_tat, all_recalls_tatlipo, all_precisions_tatlipo, all_recalls_pilin, all_precisions_pilin,  avg_epoch, \
               f1_scores, f1_scores_lipo, f1_scores_tat, f1_scores_pilin, f1_scores_tatlipo, get_class_sp_accs(life_grp, seqs, true_lbls, pred_lbls)
    return mccs, mccs2, mccs_lipo, mccs2_lipo, mccs_tat, mccs2_tat, all_recalls, all_precisions, all_recalls_lipo, \
           all_precisions_lipo, all_recalls_tat, all_precisions_tat, avg_epoch, f1_scores, f1_scores_lipo, f1_scores_tat


def get_best_corresponding_eval_mcc(result_folder="results_param_s_2/", model="", metric="mcc"):
    tr_fold = [[0, 1], [1, 2], [0, 2]]
    all_best_mccs = []
    for t_f in tr_fold:
        with open(os.path.join(result_folder, model + "_{}_{}.log".format(t_f[0], t_f[1]))) as f:
            lines = f.readlines()
        ep2mcc = {}
        best_mcc = -1
        for l in lines:
            if best_mcc != -1:
                continue
            elif "VALIDATION" in l and metric == "mcc":
                if "mcc" in l:
                    ep, mccs = l.split(":")[2], l.split(":")[4]
                    ep = int(ep.split("epoch")[-1].replace(" ", ""))
                    mccs = mccs.replace(" ", "").split(",")
                    mccs = np.mean([float(mcc) for mcc in mccs])
                    ep2mcc[ep] = mccs
            elif "train/validation loss" in l and metric == "loss":

                ep, mccs = l.split(":")[2], l.split(":")[3]
                ep = int(ep.split("epoch")[-1].split(" ")[1])
                if "," in mccs:
                    mccs = float(mccs.replace(" ", "").split(",")[0].split("/")[1])
                    ep2mcc[ep] = mccs / 2
                else:
                    mccs = float(mccs.replace(" ", "").split("/")[1])
                    ep2mcc[ep] = mccs
            elif "TEST" in l and "epoch" in l and "mcc" in l:
                best_ep = int(l.split(":")[2].split("epoch")[-1].replace(" ", ""))
                avg_last_5 = []
                for i in range(5):
                    # best_mcc = 0
                    # best_mcc = ep2mcc[best_ep - i]
                    avg_last_5.append(best_mcc)
                best_mcc = np.mean(avg_last_5)
        all_best_mccs.append(best_mcc)
    return np.mean(all_best_mccs)


def get_mean_results_for_mulitple_runs(mdlind2mdlparams, mdl2results, plot_type="prec-rec", tol=1):
    avg_mcc, avg_mcc2, avg_mcc_lipo, avg_mcc2_lipo, avg_mccs_tat, avg_mccs2_tat, avg_prec, avg_recall, avg_prec_lipo, \
    avg_recall_lipo, avg_prec_tat, avg_recall_tat, no_of_mdls = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for ind, results in mdl2results.items():
        mccs, mccs2, mccs_lipo, mccs2_lipo, mccs_tat, mccs2_tat, all_recalls, all_precisions, \
        all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat, _ = results
        mdl = mdlind2mdlparams[ind].split("run_no")[0]
        if "patience_30" in mdl:
            mdl = "patience_30"
        else:
            mdl = "patience_60"
        if mdl in no_of_mdls:
            no_of_mdls[mdl] += 1
            avg_mcc[mdl].append(np.array(mccs))
            avg_recall[mdl].append(np.array(all_recalls))
            avg_prec[mdl].append(np.array(all_precisions))
            avg_mcc2[mdl].append(np.array(mccs2))
            avg_mcc_lipo[mdl].append(np.array(mccs_lipo))
            avg_mcc2_lipo[mdl].append(np.array(mccs2_lipo))
            avg_mccs_tat[mdl].append(np.array(avg_mccs_tat))
            avg_mccs2_tat[mdl].append(np.array(avg_mccs2_tat))
            avg_recall_lipo[mdl].append(np.array(all_recalls_lipo))
            avg_prec_lipo[mdl].append(np.array(all_precisions_lipo))
            avg_recall_tat[mdl].append(np.array(all_recalls_tat))
            avg_prec_tat[mdl].append(np.array(all_precisions_tat))
        else:
            no_of_mdls[mdl] = 1
            avg_mcc[mdl] = [np.array(mccs)]
            avg_recall[mdl] = [np.array(all_recalls)]
            avg_prec[mdl] = [np.array(all_precisions)]
            avg_mcc2[mdl] = [np.array(mccs2)]
            avg_mcc_lipo[mdl] = [np.array(mccs_lipo)]
            avg_mcc2_lipo[mdl] = [np.array(mccs2_lipo)]
            avg_mccs_tat[mdl] = [np.array(avg_mccs_tat)]
            avg_mccs2_tat[mdl] = [np.array(avg_mccs2_tat)]
            avg_recall_lipo[mdl] = [np.array(all_recalls_lipo)]
            avg_prec_lipo[mdl] = [np.array(all_precisions_lipo)]
            avg_prec_tat[mdl] = [np.array(all_precisions_tat)]
            avg_recall_tat[mdl] = [np.array(all_recalls_tat)]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8)) if plot_type == "mcc" else plt.subplots(2, 4, figsize=(12, 8))
    plt.subplots_adjust(right=0.7)
    colors = ['red', 'green', 'orange', 'blue', 'brown', 'black']

    models = list(avg_mcc.keys())
    mdl2colors = {models[i]: colors[i] for i in range(len(models))}

    print(set(models))
    # TODO: Configure this for model names (usually full names are quite long and ugly)
    mdl2mdlnames = {}
    for mdl in models:
        if "patience_30" in mdl:
            mdl2mdlnames[mdl] = "patience 30"
        else:
            mdl2mdlnames[mdl] = "patience 60"
    # for mdl in models:
    #     if "lr_sched_searchlrsched_step_wrmpLrSched_0_" in mdl:
    #         mdl2mdlnames[mdl] = "step, 0 wrmp"
    #
    #     if "lr_sched_searchlrsched_expo_wrmpLrSched_10_" in mdl:
    #         mdl2mdlnames[mdl] = "expo, 10 wrmp"
    #
    #     if "lr_sched_searchlrsched_expo_wrmpLrSched_0_" in mdl:
    #         mdl2mdlnames[mdl] = "expo, 0 wrmp"
    #
    #     if "lr_sched_searchlrsched_step_wrmpLrSched_10_" in mdl:
    #         mdl2mdlnames[mdl] = "step, 10 wrmp"
    #
    #     if "test_beam_search" in mdl:
    #         mdl2mdlnames[mdl] = "no sched"
    # FOR GLBL LBL
    # for mdl in models:
    #     if "glbl_lbl_search_use_glbl_lbls_version_1_weight_1_" in mdl:
    #         mdl2mdlnames[mdl] = "version 1 weight 1"
    #     if "glbl_lbl_search_use_glbl_lbls_version_2_weight_1_" in mdl:
    #         mdl2mdlnames[mdl] = "version 2 weight 1"
    #     if "glbl_lbl_search_use_glbl_lbls_version_2_weight_0.1_" in mdl:
    #         mdl2mdlnames[mdl] = "version 2 weight 0.1"
    #     if "glbl_lbl_search_use_glbl_lbls_version_1_weight_0.1_" in mdl:
    #         mdl2mdlnames[mdl] = "version 1 weight 0.1"
    #     if "test_beam_search" in mdl:
    #         mdl2mdlnames[mdl] = "no glbl labels"
    plot_lgs = ['NEGATIVE', 'POSITIVE', 'ARCHAEA'] if plot_type == "mcc" else ['EUKARYA', 'NEGATIVE', 'POSITIVE',
                                                                               'ARCHAEA']
    for lg_ind, lg in enumerate(plot_lgs):
        if plot_type == "mcc":
            axs[0, 0].set_ylabel("MCC")
            axs[1, 0].set_ylabel("MCC2")
        else:
            axs[0, 0].set_ylabel("Recall")
            axs[1, 0].set_ylabel("Precision")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mcc[mdl][i][lg_ind + 1] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_recall[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])

            if lg_ind == 0:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[mdl])  # , label="{}".format(mdl2mdlnames[mdl]))
            else:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            axs[0, lg_ind].set_title(lg + " SEC/SPI")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mcc2[mdl][i][lg_ind + 1] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_prec[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])
            axs[1, lg_ind].plot(x, kde(x),
                                color=mdl2colors[mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            if lg_ind == len(plot_lgs) - 1:
                axs[1, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl], label="{}".format(mdl2mdlnames[mdl]))

    plot_lgs = ['NEGATIVE', 'POSITIVE', 'ARCHAEA']
    plt.legend(loc=(1.04, 1))
    plt.show()

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(right=0.7)
    for lg_ind, lg in enumerate(['NEGATIVE', 'POSITIVE', 'ARCHAEA']):
        if plot_type == "mcc":
            axs[0, 0].set_ylabel("MCC")
            axs[1, 0].set_ylabel("MCC2")
        else:
            axs[0, 0].set_ylabel("Recall")
            axs[1, 0].set_ylabel("Precision")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mcc_lipo[mdl][i][lg_ind] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_recall_lipo[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])
            if lg_ind == 0:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[mdl])  # , label="{}".format(mdl2mdlnames[mdl]))
            else:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            axs[0, lg_ind].set_title(lg + " SEC/SPII")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mcc2_lipo[mdl][i][lg_ind] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_prec_lipo[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])
            axs[1, lg_ind].plot(x, kde(x),
                                color=mdl2colors[mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            if lg_ind == 2:
                axs[1, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl], label="{}".format(mdl2mdlnames[mdl]))

    plt.legend(loc=(1.04, 1))
    plt.show()

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(right=0.7)
    for lg_ind, lg in enumerate(['NEGATIVE', 'POSITIVE', 'ARCHAEA']):
        if plot_type == "mcc":
            axs[0, 0].set_ylabel("MCC")
            axs[1, 0].set_ylabel("MCC2")
        else:
            axs[0, 0].set_ylabel("Recall")
            axs[1, 0].set_ylabel("Precision")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mccs_tat[mdl][i][lg_ind] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_recall_tat[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])
            if lg_ind == 0:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[mdl])  # , label="{}".format(mdl2mdlnames[mdl]))
            else:
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            axs[0, lg_ind].set_title(lg + " TAT/SPI")
        for mdl in avg_mccs_tat.keys():
            x = np.linspace(0, 1, 1000)
            if plot_type == "mcc":
                kde = stats.gaussian_kde([avg_mccs2_tat[mdl][i][lg_ind] for i in range(no_of_mdls[mdl])])
            else:
                kde = stats.gaussian_kde([avg_prec_tat[mdl][i][lg_ind * 4 + tol] for i in range(no_of_mdls[mdl])])
            axs[1, lg_ind].plot(x, kde(x),
                                color=mdl2colors[mdl])  # , label="Eukaryote {} param search".format(mdl2mdlnames[mdl]))
            if lg_ind == 2:
                axs[1, lg_ind].plot(x, kde(x), color=mdl2colors[
                    mdl], label="{}".format(mdl2mdlnames[mdl]))

    plt.legend(loc=(1.04, 1))
    plt.show()
    if plot_type == "mcc":
        fig, axs = plt.subplots(1, 1, figsize=(4, 8))
        plt.subplots_adjust(right=0.7)
        for lg_ind, lg in enumerate(['EUKARYOTE']):
            axs[0, 0].set_ylabel("MCC")
            for mdl in avg_mccs_tat.keys():
                x = np.linspace(0, 1, 1000)
                kde = stats.gaussian_kde([avg_mcc[mdl][i][lg_ind + 1] for i in range(no_of_mdls[mdl])])
                axs[0, lg_ind].plot(x, kde(x), color=mdl2colors[mdl], label="{}".format(mdl2mdlnames[mdl]))
                axs[0, lg_ind].set_title(lg + " SEC/SPI")
        plt.legend(loc=(1.04, 1))
        plt.show()


def get_f1_scores(rec, prec):
    return [2 * rec[i] * prec[i] / (rec[i] + prec[i]) if rec[i] + prec[i] != 0 else 0 for i in range(len(rec))]


def extract_all_param_results(result_folder="results_param_s_2/", only_cs_position=False, compare_mdl_plots=False,
                              remove_test_seqs=False, benchmark=True, restrict_types=None, return_results=False, prints=True):
    sp6_recalls_sp1 = [0.747, 0.774, 0.808, 0.829, 0.639, 0.672, 0.689, 0.721, 0.800, 0.800, 0.800, 0.800, 0.500, 0.556,
                       0.556, 0.583]
    sp6_recalls_sp2 = [0.852, 0.852, 0.856, 0.864, 0.875, 0.883, 0.883, 0.883, 0.778, 0.778, 0.778, 0.778]
    sp6_recalls_tat = [0.706, 0.765, 0.784, 0.804, 0.556, 0.556, 0.667, 0.667, 0.333, 0.444, 0.444, 0.444]
    sp6_precs_sp1 = [0.661, 0.685, 0.715, 0.733, 0.534, 0.562, 0.575, 0.603, 0.632, 0.632, 0.632, 0.632, 0.643, 0.714,
                     0.714, 0.75]
    sp6_precs_sp2 = [0.913, 0.913, 0.917, 0.925, 0.929, 0.938, 0.938, 0.938, 0.583, 0.583, 0.583, 0.583]
    sp6_precs_tat = [0.679, 0.736, 0.755, 0.774, 0.714, 0.714, 0.857, 0.857, 0.375, 0.5, 0.5, 0.5]
    sp6_f1_sp1 = get_f1_scores(sp6_recalls_sp1, sp6_precs_sp1)
    sp6_f1_sp2 = get_f1_scores(sp6_recalls_sp2, sp6_precs_sp2)
    sp6_f1_tat = get_f1_scores(sp6_recalls_tat, sp6_precs_tat)

    if benchmark:
        no_of_seqs_sp1 = list(np.array([146, 61, 15, 36]).repeat(4))
        no_of_seqs_sp2 = list(np.array([257, 120, 9]).repeat(4))
        no_of_seqs_tat = list(np.array([51 ,18, 9]).repeat(4))
        no_of_tested_sp_seqs = sum([146, 61, 15, 36]) + sum([257, 120, 9]) + sum([51 ,18, 9])
    else:
        no_of_seqs_sp1 = list(np.array([2040, 44, 142, 356]).repeat(4))
        no_of_seqs_sp2 = list(np.array([1087, 516, 12]).repeat(4))
        no_of_seqs_tat = list(np.array([313 ,39, 13]).repeat(4))
        no_of_tested_sp_seqs = sum([2040, 44, 142, 356]) + sum([1087, 516, 12]) + sum([313 ,39, 13])
    sp6_summarized = np.sum((np.array(sp6_f1_sp1) * np.array(no_of_seqs_sp1))) / no_of_tested_sp_seqs + \
                        np.sum((np.array(sp6_f1_sp2) * np.array(no_of_seqs_sp2))) / no_of_tested_sp_seqs + \
                        np.sum((np.array(sp6_f1_tat) * np.array(no_of_seqs_tat))) / no_of_tested_sp_seqs
    sp6_summarized_non_weighted = (np.mean(sp6_f1_sp1) + np.mean(sp6_f1_sp2) + np.mean(sp6_f1_tat))/3
    sp6_recalls_sp1 = [str(round(sp6_r_sp1, 3)) for sp6_r_sp1 in sp6_recalls_sp1]
    sp6_recalls_sp2 = [str(round(sp6_r_sp2, 3)) for sp6_r_sp2 in sp6_recalls_sp2]
    sp6_recalls_tat = [str(round(sp6_r_tat, 3)) for sp6_r_tat in sp6_recalls_tat]
    sp6_precs_sp1 = [str(round(sp6_prec_sp1, 3)) for sp6_prec_sp1 in sp6_precs_sp1]
    sp6_precs_sp2 = [str(round(sp6_p_sp2, 3)) for sp6_p_sp2 in sp6_precs_sp2]
    sp6_precs_tat = [str(round(sp6_p_tat, 3)) for sp6_p_tat in sp6_precs_tat]
    sp6_f1_sp1 = [str(round(sp6_f1_sp1_, 3)) for sp6_f1_sp1_ in sp6_f1_sp1]
    sp6_f1_sp2 = [str(round(sp6_f1_sp2_, 3)) for sp6_f1_sp2_ in sp6_f1_sp2]
    sp6_f1_tat = [str(round(sp6_f1_tat_, 3)) for sp6_f1_tat_ in sp6_f1_tat]
    files = os.listdir(result_folder)
    unique_params = set()
    for f in files:
        if "log" in f:
            # check if all 3 folds have finished
            dont_add = False
            for tr_f in [[0, 1], [1, 2], [0, 2]]:
                if "_".join(f.split("_")[:-2]) + "_{}_{}_best.bin".format(tr_f[0], tr_f[1]) not in files:
                    dont_add = True
            if not dont_add:
                unique_params.add("_".join(f.split("_")[:-2]))
    mdl2results = {}
    mdl2summarized_results = {}
    mdlind2mdlparams = {}
    # order results by the eukaryote mcc
    eukaryote_mcc = []
    for ind, u_p in enumerate(unique_params):
        mccs, mccs2, mccs_lipo, mccs2_lipo, mccs_tat, mccs2_tat, mccs_tatlipo, mccs2_tatlipo, mccs_pilin, mccs2_pilin, \
        all_recalls, all_precisions, all_recalls_lipo, all_precisions_lipo, all_recalls_tat, all_precisions_tat, \
        all_recalls_tatlipo, all_precisions_tatlipo, all_recalls_pilin, all_precisions_pilin, \
        avg_epoch, f1_scores, f1_scores_lipo, f1_scores_tat, f1_scores_tatlipo, f1_scores_pilin, f1_scores_sptype \
            = extract_mean_test_results(run=u_p, result_folder=result_folder,
                                        only_cs_position=only_cs_position,
                                        remove_test_seqs=remove_test_seqs, return_sptype_f1=True, benchmark=benchmark,
                                        restrict_types=restrict_types)
        if prints:
            TSignalSummary = ((np.mean(np.array(f1_scores)) + np.mean(np.array(f1_scores_lipo)) + np.mean(np.array(f1_scores_tat)))/3)
            print("Non-weighted summary (sp6/Tsignal):", sp6_summarized_non_weighted, TSignalSummary)
        all_recalls_lipo, all_precisions_lipo, \
        all_recalls_tat, all_precisions_tat,\
        all_recalls_tatlipo, all_precisions_tatlipo,\
        all_recalls_pilin, all_precisions_pilin= list(np.reshape(np.array(all_recalls_lipo), -1)), list(
            np.reshape(np.array(all_precisions_lipo), -1)), \
                                               list(np.reshape(np.array(all_recalls_tat), -1)), list(
            np.reshape(np.array(all_precisions_tat), -1)), \
                                           list(np.reshape(np.array(all_recalls_tatlipo), -1)), list(
            np.reshape(np.array(all_precisions_tatlipo), -1)), \
                               list(np.reshape(np.array(all_recalls_pilin), -1)), list(
            np.reshape(np.array(all_precisions_pilin), -1))
        mdl2results[ind] = (
            mccs, mccs2, mccs_lipo, mccs2_lipo, mccs_tat, mccs2_tat, mccs_tatlipo, mccs2_tatlipo, mccs_pilin, mccs2_pilin,
            list(np.reshape(np.array(all_recalls), -1)), list(np.reshape(np.array(all_precisions), -1)), all_recalls_lipo,
            all_precisions_lipo, all_recalls_tat, all_precisions_tat, all_recalls_tatlipo, all_precisions_tatlipo,
            all_recalls_pilin, all_precisions_pilin, f1_scores, f1_scores_lipo, f1_scores_tat, f1_scores_tatlipo,
                f1_scores_pilin, f1_scores_sptype, avg_epoch)
        mdl2summarized_results[ind] = np.sum((np.array(f1_scores).reshape(-1) * np.array(no_of_seqs_sp1)))/no_of_tested_sp_seqs + \
                                      np.sum((np.array(f1_scores_lipo).reshape(-1) * np.array(no_of_seqs_sp2)))/no_of_tested_sp_seqs + \
                                      np.sum((np.array(f1_scores_tat).reshape(-1) * np.array(no_of_seqs_tat)))/no_of_tested_sp_seqs
        mdlind2mdlparams[ind] = u_p
        eukaryote_mcc.append(get_best_corresponding_eval_mcc(result_folder, u_p))
    if compare_mdl_plots:
        get_mean_results_for_mulitple_runs(mdlind2mdlparams, mdl2results)
    best_to_worst_mdls = np.argsort(eukaryote_mcc)[::-1]
    for mdl_ind in best_to_worst_mdls:
        params = ""
        mdl_params = mdlind2mdlparams[mdl_ind]
        if "use_glbl_lbls" in mdl_params:
            params += "wGlbl"
        else:
            params += "nGlbl"
        # patience_ind = mdl_params.find("patience_") + len("patience_")
        # patience = mdl_params[patience_ind:patience_ind+2]
        # params += "_{}".format(patience)
        if "nlayers" in mdl_params:
            nlayers = mdl_params[mdl_params.find("nlayers") + len("nlayers"):].split("_")[1]
            params += "_{}".format(nlayers)
        if "nhead" in mdl_params:
            nhead = mdl_params[mdl_params.find("nhead") + len("nhead"):].split("_")[1]
            params += "_{}".format(nhead)
        if "lrsched" in mdl_params:
            lr_sched = mdl_params[mdl_params.find("lrsched") + len("lrsched"):].split("_")[1]
            params += "_{}".format(lr_sched)
        if "dos" in mdl_params:
            dos = mdl_params[mdl_params.find("dos"):].split("_")[1]
            params += "_{}".format(dos)
        mdlind2mdlparams[mdl_ind] = params
    if prints:
        print("\n\nMCC SEC/SPI TABLE\n\n")
        for mdl_ind in best_to_worst_mdls:
            mdl_params = " & ".join(mdlind2mdlparams[mdl_ind].split("_"))
            print(mdl_params, " & ", " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][0]]), "&",
                  " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][1][1:]]), " & ",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nF1 table SEC/SPI\n\n")
        no_of_params = len(mdlind2mdlparams[best_to_worst_mdls[0]].split("_"))
        print(" SP6 ", " & " * no_of_params, " & ".join(sp6_f1_sp1), " & \\\\ \\hline")
        for mdl_ind in best_to_worst_mdls:
            print("total f1 for {}: {} compared to sp6: {}".format(mdl_ind, mdl2summarized_results[mdl_ind]/4, sp6_summarized/4))
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in np.concatenate(mdl2results[mdl_ind][-7])]), " & ",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nRecall table SEC/SPI\n\n")
        print(" SP6 ", " & " * no_of_params, " & ".join(sp6_recalls_sp1), " & \\\\ \\hline")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][10]]), " & ",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")
        print("\n\nPrec table SEC/SPI\n\n")
        print(" SP6 ", " & " * no_of_params, " & ".join(sp6_precs_sp1), " & \\\\ \\hline")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][11]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nF1 table SEC/SPII \n\n")
        print(" SP6 ", " & " * no_of_params, " & ".join(sp6_f1_sp2), " & \\\\ \\hline")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in np.concatenate(mdl2results[mdl_ind][-6])]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nRecall table SEC/SPII \n\n")
        print(" SP6 ", " & " * no_of_params, " & ".join(sp6_recalls_sp2), " & \\\\ \\hline")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][12]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nPrec table SEC/SPII \n\n")
        print(" SP6 ", " & " * no_of_params, " & ".join(sp6_precs_sp2), " & \\\\ \\hline")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][13]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nF1 table TAT/SPI \n\n")
        print(" SP6 ", " & " * no_of_params, " & ".join(sp6_f1_tat), " & \\\\ \\hline")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in np.concatenate(mdl2results[mdl_ind][-5])]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nRecall table TAT/SPI \n\n")
        print(" SP6 ", " & " * no_of_params, " & ".join(sp6_recalls_tat), " & \\\\ \\hline")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][14]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nPrec table TAT/SPI \n\n")
        print(" SP6 ", " & " * no_of_params, " & ".join(sp6_precs_tat), " & \\\\ \\hline")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][15]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")


        print("\n\nF1 table TAT/SPII \n\n")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in np.concatenate(mdl2results[mdl_ind][-4])]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nRecall table TAT/SPII \n\n")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][16]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nPrec table TAT/SPII \n\n")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][17]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nF1 table Sec/SPIV \n\n")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in np.concatenate(mdl2results[mdl_ind][-3])]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nRecall table Sec/SPIV \n\n")
        print(" SP6 ", " & " * no_of_params, " & ".join(sp6_recalls_tat), " & \\\\ \\hline")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][18]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nPrec table Sec/SPIV \n\n")
        print(" SP6 ", " & " * no_of_params, " & ".join(sp6_precs_tat), " & \\\\ \\hline")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(rec, 3)) for rec in mdl2results[mdl_ind][19]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nMCC SEC/SPII TABLE\n\n")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][2]]), "&",
                  " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][3]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nMCC TAT/SPI TABLE\n\n")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][4]]), "&",
                  " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][5]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

        print("\n\nSP-type preds F1\n\n")
        for mdl_ind in best_to_worst_mdls:
            print(" & ".join(mdlind2mdlparams[mdl_ind].split("_")), " & ",
                  " & ".join([str(round(mcc, 3)) for mcc in mdl2results[mdl_ind][-2]]), "&",
                  round(mdl2results[mdl_ind][-1], 3), "\\\\ \\hline")

    return mdl2results


def sanity_checks(run="param_search_0_2048_0.0001_", folder="results/"):
    # S = signal_peptide; T = Tat/SPI or Tat/SPII SP; L = Sec/SPII SP; P = SEC/SPIII SP; I = cytoplasm; M = transmembrane; O = extracellular;

    def get_last_contiguous_index(seq, signal_peptide):
        ind = 0
        while seq[ind] == signal_peptide and ind < len(seq) - 1:
            ind += 1
        return ind - 1

    def check_contiguous_sp(lbl_seqs):
        for l in lbl_seqs:
            signal_peptide = None
            l = l.replace("ES", "")
            if "S" in l:
                signal_peptide = "S"
            elif "T" in l:
                signal_peptide = "T"
            elif "L" in l:
                signal_peptide = "L"
            elif "P" in l:
                signal_peptide = "P"
            if signal_peptide is not None:

                if l.rfind(signal_peptide) != get_last_contiguous_index(l, signal_peptide):
                    print(l, l.rfind(signal_peptide), get_last_contiguous_index(l, signal_peptide), signal_peptide)

    for tr_fold in [[0, 1], [1, 2], [0, 2]]:
        labels = pickle.load(open(folder + run + "{}_{}.bin".format(tr_fold[0], tr_fold[1]), "rb")).values()
        check_contiguous_sp(labels)


def extract_all_mdl_results(mdl2results):
    euk_mcc, neg_mcc, pos_mcc, arch_mcc = [], [], [], []
    euk_rec, neg_rec, pos_rec, arch_rec = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
    euk_prec, neg_prec, pos_prec, arch_prec = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
    for _, (mccs, recalls, precisions, epochs) in mdl2results.items():
        euk_mcc.append(mccs[0])
        neg_mcc.append(mccs[1])
        pos_mcc.append(mccs[2])
        arch_mcc.append(mccs[3])
        euk_rec[0].append(recalls[0])
        euk_rec[1].append(recalls[1])
        euk_rec[2].append(recalls[2])
        euk_rec[3].append(recalls[3])
        neg_rec[0].append(recalls[4])
        neg_rec[1].append(recalls[5])
        neg_rec[2].append(recalls[6])
        neg_rec[3].append(recalls[7])
        pos_rec[0].append(recalls[8])
        pos_rec[1].append(recalls[9])
        pos_rec[2].append(recalls[10])
        pos_rec[3].append(recalls[11])
        arch_rec[0].append(recalls[12])
        arch_rec[1].append(recalls[13])
        arch_rec[2].append(recalls[14])
        arch_rec[3].append(recalls[15])

        euk_prec[0].append(precisions[0])
        euk_prec[1].append(precisions[1])
        euk_prec[2].append(precisions[2])
        euk_prec[3].append(precisions[3])
        neg_prec[0].append(precisions[4])
        neg_prec[1].append(precisions[5])
        neg_prec[2].append(precisions[6])
        neg_prec[3].append(precisions[7])
        pos_prec[0].append(precisions[8])
        pos_prec[1].append(precisions[9])
        pos_prec[2].append(precisions[10])
        pos_prec[3].append(precisions[11])
        arch_prec[0].append(precisions[12])
        arch_prec[1].append(precisions[13])
        arch_prec[2].append(precisions[14])
        arch_prec[3].append(precisions[15])
    return euk_mcc, neg_mcc, pos_mcc, arch_mcc, euk_rec, neg_rec, pos_rec, arch_rec, euk_prec, neg_prec, pos_prec, arch_prec


def visualize_training_variance(mdl2results, mdl2results_hps=None):
    def plot_4_figs(measures, sp6_measure, hp_s_measures=None, name="", plot_hps=False):
        euk, neg, pos, arch = measures
        if plot_hps:
            euk_hps, neg_hps, pos_hps, arch_hps = hp_s_measures

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(name)
        x = np.linspace(0, 1, 1000)
        kde = stats.gaussian_kde(euk)
        # axs[0, 0].plot(x, kde(x), color = "blue", label="Eukaryote {}".format(name))
        axs[0, 0].hist(euk, color="blue", bins=10, label="Eukaryote {}".format(name))
        if plot_hps:
            kde = stats.gaussian_kde(euk_hps)
            axs[0, 0].hist(euk_hps, alpha=0.5, color="red", bins=10, label="Eukaryote {} param search".format(name))
            # axs[0, 0].plot(x, kde(x), color = "red", label="Eukaryote {} param search".format(name))
        axs[0, 0].set_ylabel("No of models")
        axs[0, 0].set_xlim(0, 1)
        axs[0, 0].plot([sp6_measure[0], sp6_measure[0]], [0, 5], 'r--', label="SP6 result")
        axs[0, 0].legend()

        kde = stats.gaussian_kde(neg)
        axs[0, 1].hist(neg, bins=10, color="blue", label="Negative {}".format(name))
        # axs[0, 1].plot(x, kde(x), color = "blue", label="Negative {}".format(name))
        if plot_hps:
            kde = stats.gaussian_kde(neg_hps)
            axs[0, 1].hist(neg_hps, alpha=0.5, bins=10, color="red", label="Negative {} param search".format(name))
            # axs[0, 1].plot(x, kde(x), color = "red", label="Negative {} param search".format(name))
        axs[0, 1].set_xlim(0, 1)
        axs[0, 1].plot([sp6_measure[1], sp6_measure[1]], [0, 5], 'r--', label="SP6 result")

        axs[0, 1].legend()

        # axs[1, 0].hist(pos,bins=10,color = "blue", label="Positive {}".format(name))
        kde = stats.gaussian_kde(pos)
        axs[1, 0].hist(pos, bins=10, color="blue", label="Positive {}".format(name))
        # axs[1, 0].plot(x, kde(x), color = "blue", label="Positive {}".format(name))

        if plot_hps:
            # kde = stats.gaussian_kde(pos_hps)
            axs[1, 0].hist(pos_hps, color="red", alpha=0.5, bins=10, label="Positive {} param search".format(name))
            axs[1, 0].plot(x, kde(x), color="red", label="Positive {} param search".format(name))

        axs[1, 0].set_xlim(0, 1)
        axs[1, 0].set_ylabel("No of models")
        axs[1, 0].plot([sp6_measure[2], sp6_measure[2]], [0, 5], 'r--', label="SP6 result")
        axs[1, 0].legend()
        axs[1, 0].set_xlabel(name.split(" ")[0])

        kde = stats.gaussian_kde(arch)
        axs[1, 1].hist(arch, bins=10, color="blue", label="Archaea {}".format(name))
        # axs[1, 1].plot(x, kde(x), color = "blue",label="Archaea {}".format(name))
        if plot_hps:
            kde = stats.gaussian_kde(arch_hps)
            axs[1, 1].hist(arch_hps, alpha=0.5, color="red", bins=10, label="Archaea {} param search".format(name))
            # axs[1, 1].plot(x, kde(x), color = "red", label="Archaea {} param search".format(name))
        axs[1, 1].set_xlim(0, 1)
        axs[1, 1].plot([sp6_measure[3], sp6_measure[3]], [0, 5], 'r--', label="SP6 result")

        axs[1, 1].legend()
        axs[1, 1].set_xlabel(name.split(" ")[0])
        plt.show()
        # plt.savefig("/home/alex/Desktop/sp6_ds_transformer_nmt_results/{}_{}.png".format(name, "mcc"))

    euk_mcc_sp6, neg_mcc_sp6, pos_mcc_sp6, arch_mcc_sp6 = 0.868, 0.811, 0.878, 0.737
    euk_rec_sp6, neg_rec_sp6, pos_rec_sp6, arch_rec_sp6 = [0.747, 0.774, 0.808, 0.829], [0.639, 0.672, 0.689, 0.721], \
                                                          [0.800, 0.800, 0.800, 0.800], [0.500, 0.556, 0.556, 0.583]
    euk_prec_sp6, neg_prec_sp6, pos_prec_sp6, arch_prec_sp6 = [0.661, 0.685, 0.715, 0.733], [0.534, 0.562, 0.575,
                                                                                             0.603], \
                                                              [0.632, 0.632, 0.632, 0.632], [0.643, 0.714, 0.714, 0.75]
    euk_mcc, neg_mcc, pos_mcc, arch_mcc, euk_rec, neg_rec, \
    pos_rec, arch_rec, euk_prec, neg_prec, pos_prec, arch_prec = extract_all_mdl_results(mdl2results)
    if mdl2results_hps is not None:
        euk_hps_mcc, neg_hps_mcc, pos_hps_mcc, arch_hps_mcc, euk_hps_rec, neg_hps_rec, pos_hps_rec, arch_hps_rec, \
        euk_hps_prec, neg_hps_prec, pos_hps_prec, arch_hps_prec = extract_all_mdl_results(mdl2results_hps)
    else:
        euk_hps_mcc, neg_hps_mcc, pos_hps_mcc, arch_hps_mcc, euk_hps_rec, neg_hps_rec, pos_hps_rec, arch_hps_rec, \
        euk_hps_prec, neg_hps_prec, pos_hps_prec, arch_hps_prec = None, None, None, None, [None, None, None, None], \
                                                                  [None, None, None, None], [None, None, None, None], \
                                                                  [None, None, None, None], [None, None, None, None], \
                                                                  [None, None, None, None], [None, None, None, None], \
                                                                  [None, None, None, None]
    plot_hps = mdl2results_hps is not None
    plot_4_figs([euk_mcc, neg_mcc, pos_mcc, arch_mcc],
                [euk_mcc_sp6, neg_mcc_sp6, pos_mcc_sp6, arch_mcc_sp6],
                [euk_hps_mcc, neg_hps_mcc, pos_hps_mcc, arch_hps_mcc],
                name='mcc', plot_hps=plot_hps)
    for i in range(4):
        plot_4_figs([euk_rec[i], neg_rec[i], pos_rec[i], arch_rec[i]],
                    [euk_rec_sp6[i], neg_rec_sp6[i], pos_rec_sp6[i], arch_rec_sp6[i]],
                    [euk_hps_rec[i], neg_hps_rec[i], pos_hps_rec[i], arch_hps_rec[i]],
                    name='recall tol={}'.format(i), plot_hps=plot_hps)
        plot_4_figs([euk_prec[i], neg_prec[i], pos_prec[i], arch_prec[i]],
                    [euk_prec_sp6[i], neg_prec_sp6[i], pos_prec_sp6[i], arch_prec_sp6[i]],
                    [euk_hps_prec[i], neg_hps_prec[i], pos_hps_prec[i], arch_hps_prec[i]],
                    name='precision tol={}'.format(i), plot_hps=plot_hps)


def extract_calibration_probs_for_mdl(model="parameter_search_patience_60use_glbl_lbls_use_glbl_lbls_versio"
                                            "n_1_weight_0.1_lr_1e-05_nlayers_3_nhead_16_lrsched_none_trFlds_",
                                      folder='huge_param_search/patience_60/', plot=True):
    all_lg, all_seqs, all_tl, all_pred_lbls, sp2probs = [], [], [], [], {}
    for tr_f in [[0, 1], [0, 2], [1, 2]]:
        prob_file = "{}{}_{}_best_sp_probs.bin".format(model, tr_f[0], tr_f[1])
        preds_file = "{}{}_{}_best.bin".format(model, tr_f[0], tr_f[1])
        life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename=folder + preds_file)
        all_lg.extend(life_grp)
        all_seqs.extend(seqs)
        all_tl.extend(true_lbls)
        all_pred_lbls.extend(pred_lbls)
        sp2probs.update(pickle.load(open(folder + prob_file, "rb")))

    return get_prob_calibration_and_plot("", all_lg, all_seqs, all_tl, all_pred_lbls, sp2probs=sp2probs, plot=plot)


def duplicate_Some_logs():
    from subprocess import call

    files = os.listdir("beam_test")
    for f in files:
        if "log" in f:
            file = f.replace(".bin", "_best.bin")
            cmd = ["cp", "beam_test/" + f, "beam_test/actual_beams/best_beam_" + file]
            call(cmd)
    exit(1)

def prep_sp1_sp2():

    file = "../sp_data/benchmark_set_sp5.fasta"
    file_new = "../sp_data/train_set.fasta"
    ids_benchmark_sp5 = []
    for seq_record in SeqIO.parse(file, "fasta"):
        ids_benchmark_sp5.append(seq_record.id.split("|")[0])
    seqs, ids = [], []
    lines = []
    for seq_record in SeqIO.parse(file_new, "fasta"):
        if seq_record.id.split("|")[0] in ids_benchmark_sp5 and seq_record.id.split("|")[2] in ["SP", "NO_SP"]:
            seqs.append(seq_record.seq[:len(seq_record.seq) // 2])
            ids.append(seq_record.id)
            lines.append(">"+ids[-1]+"\n")
            lines.append(str(seqs[-1])+"\n")
    for i in range(len(lines) // 1000 + 1) :
        with open("sp1_sp2_fastas/pred_signal{}.fasta".format(i), "wt") as f:
            f.writelines(lines[i * 1000:(i+1) * 1000])

def ask_uniprot():
    cID='Q70UQ6'

    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl=baseUrl+cID+".gff"
    response = r.post(currentUrl)
    cData=''.join(response.text)
    print(cData)


def correct_duplicates_training_data():
    sublbls=True
    file_new = "../sp_data/sp6_data/train_set.fasta"
    decided_ids = ['B3GZ85', 'B0R5Y3', 'Q0T616', 'Q7CI09', 'P33937', 'P63883', 'P33937', 'Q9P121', 'C1CTN0', 'Q8FAX0',
                   'P9WK51', 'Q5GZP1', 'P0AD45', 'P0DC88', 'Q8E6W4', 'Q5HMD1', 'Q2FWG4', 'Q5HLG6', 'Q8Y7A9', 'P65631',
                   'B1AIC4', 'Q2FZJ9', ' P0ABJ2', 'P0AD46', 'P0ABJ2', 'Q99V36', 'Q7A698', 'Q5HH23', 'Q6GI23', 'Q7A181',
                   'Q2YX14', 'Q6GAF2', 'P65628', 'P65629', 'P65630', 'Q5HEA9', 'P0DC86', 'Q2YUI9', 'Q5XDY9', 'Q2FF36',
                   'Q1R3H8', 'P0DC87', 'A5IUN6', 'A6QIT4', 'A7X4S6', 'Q6G7M0', 'Q1CHD5']
    #
    decided_ids_2_info = {}
    decided_str_2_info = {}
    processed_seqs = []
    for seq_record in SeqIO.parse(file_new, "fasta"):
        if str(seq_record.seq[: len(seq_record.seq) // 2]) in processed_seqs:
            if seq_record.id.split("|")[0] in decided_ids:
                decided_str_2_info[str(seq_record.seq[: len(seq_record.seq) // 2])] = (seq_record.id.split("|")[1],
                                                                                  seq_record.id.split("|")[2],
                                                                                  seq_record.id.split("|")[3],
                                                                                  str(seq_record.seq[len(seq_record.seq)//2:]))
        else:
            decided_str_2_info[str(seq_record.seq[: len(seq_record.seq) // 2])] = (seq_record.id.split("|")[1],
                                                                                  seq_record.id.split("|")[2],
                                                                                  seq_record.id.split("|")[3],
                                                                                  str(seq_record.seq[len(seq_record.seq)//2:]))
    # POSITIVE', 'TAT', '0', 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO'
    # (emb, , 'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII', 'EUKARYA', 'NO_SP')
    # sp6_partitioned_data_test_2.bin
    remove_count, remove_count2 = 0, 0
    total_count = 0
    seen_seqs = []
    removed_lbls = []
    for tr_f in [0, 1, 2]:
        for t_s in ["train", "test"]:
            new_seqs_2_info = {}
            seqs = pickle.load(open("../sp_data/sp6_partitioned_data_{}_{}.bin".format(t_s, tr_f), "rb")) \
                if not sublbls else pickle.load(open("../sp_data/sp6_partitioned_data_sublbls_{}_{}.bin".format(t_s, tr_f), "rb"))
            for k, info in seqs.items():
                total_count += 1
                if info[1] != decided_str_2_info[k][-1] or info[2] != decided_str_2_info[k][0] or  info[3] != decided_str_2_info[k][1] :
                    remove_count += 1
                    removed_lbls.append((info[1], decided_str_2_info[k][-1]))
                    newlbls = info[1] if decided_str_2_info[k][1] != "TATLIPO" or sublbls else info[1].replace("T", "W")
                    new_seqs_2_info[k] = (newlbls, decided_str_2_info[k][-1], decided_str_2_info[k][0], decided_str_2_info[k][1])
                elif k in seen_seqs:
                    removed_lbls.append((info[1], decided_str_2_info[k][-1]))
                    remove_count2+=1

                elif k not in seen_seqs:
                    seen_seqs.append(k)
                    newlbls = info[1].replace("T", "W") if info[-1] == "TATLIPO" and not sublbls else info[1]
                    new_seqs_2_info[k] = (info[0], newlbls, info[2], info[3])
            key = list(new_seqs_2_info.keys())[0]
            if sublbls:
                pickle.dump(new_seqs_2_info, open("../sp_data/sp6_partitioned_data_sublbls_{}_{}.bin".format(t_s, tr_f), "wb"))
            else:
                pickle.dump(new_seqs_2_info, open("../sp_data/sp6_partitioned_data_{}_{}.bin".format(t_s, tr_f), "wb"))

def count_seqs_lgs(seqs):
    file_new = "../sp_data/sp6_data/train_set.fasta"
    id2seq = {}
    id2lg = {}
    id2type = {}
    id2truelbls = {}
    ids_benchmark_sp5 = []
    seen_seqs = []
    count_seqs = {"EUKARYA":{"NO_SP":0, "SP":0}, "NEGATIVE":{"NO_SP":0, "SP":0, "TAT":0, "TATLIPO":0, "PILIN":0, "LIPO":0},
                  "POSITIVE":{"NO_SP":0, "SP":0, "TAT":0, "TATLIPO":0, "PILIN":0, "LIPO":0}, "ARCHAEA":{"NO_SP":0, "SP":0, "TAT":0, "TATLIPO":0, "PILIN":0, "LIPO":0}}
    for seq_record in SeqIO.parse(file_new, "fasta"):
        if str(seq_record.seq[:len(seq_record.seq) // 2]) in seqs and str(seq_record.seq[:len(seq_record.seq) // 2]) not in seen_seqs:
            seen_seqs.append(seq_record.seq[:len(seq_record.seq) // 2])
            sp_type = str(seq_record.id.split("|")[2])
            lg = str(seq_record.id.split("|")[1])
            count_seqs[lg][sp_type] +=1
    print(count_seqs)

def extract_id2seq_dict(file="train_set.fasta"):
    """
        function that retrieves id dictionaries with only the benchmark ids
    """

    # for seq_record in SeqIO.parse(file_new, "fasta"):
    # id2seq[seq_record.id.split("|")[0]] = str(seq_record.seq[:len(seq_record.seq) // 2])
    # id2truelbls[seq_record.id.split("|")[0]] = str(seq_record.seq[len(seq_record.seq) // 2:])
    # id2lg[seq_record.id.split("|")[0]] = str(seq_record.id.split("|")[1])
    # id2type[seq_record.id.split("|")[0]] = str(seq_record.id.split("|")[2])
    # decided_ids = ['B3GZ85', 'B0R5Y3', 'Q0T616', 'Q7CI09', 'P33937', 'P63883', 'P33937', 'Q9P121', 'C1CTN0', 'Q8FAX0',
    #                'P9WK51', 'Q5GZP1', 'P0AD45', 'P0DC88', 'Q8E6W4', 'Q5HMD1', 'Q2FWG4', 'Q5HLG6', 'Q8Y7A9', 'P65631',
    #                'B1AIC4', 'Q2FZJ9', ' P0ABJ2', 'P0AD46', 'P0ABJ2', 'Q99V36', 'Q7A698', 'Q5HH23', 'Q6GI23', 'Q7A181',
    #                'Q2YX14', 'Q6GAF2', 'P65628', 'P65629', 'P65630', 'Q5HEA9', 'P0DC86', 'Q2YUI9', 'Q5XDY9', 'Q2FF36',
    #                'Q1R3H8', 'P0DC87', 'A5IUN
    #                6', 'A6QIT4', 'A7X4S6', 'Q6G7M0', 'Q1CHD5']
    file = "../sp_data/benchmark_set_sp5.fasta"
    file_new = "../sp_data/train_set.fasta"
    id2seq = {}
    id2lg = {}
    id2type = {}
    id2truelbls = {}
    ids_benchmark_sp5 = []
    seqs_set = set()
    for seq_record in SeqIO.parse(file, "fasta"):
        ids_benchmark_sp5.append(seq_record.id.split("|")[0])
    for seq_record in SeqIO.parse(file_new, "fasta"):
        if seq_record.id.split("|")[0] in ids_benchmark_sp5:
            if seq_record.id.split("|")[0] not in id2seq:
                id2seq[seq_record.id.split("|")[0]] = str(seq_record.seq[:len(seq_record.seq) // 2])
                id2truelbls[seq_record.id.split("|")[0]] = str(seq_record.seq[len(seq_record.seq) // 2:])
                id2lg[seq_record.id.split("|")[0]] = str(seq_record.id.split("|")[1])
                id2type[seq_record.id.split("|")[0]] = str(seq_record.id.split("|")[2])
            if id2seq[seq_record.id.split("|")[0]] not in seqs_set:
                seqs_set.add(id2seq[seq_record.id.split("|")[0]])
            else:
                print(id2seq[seq_record.id.split("|")[0]])
    lg_sptype2count = {}
    for k, v in id2lg.items():
        if v + "_" + id2type[k] not in lg_sptype2count:
            lg_sptype2count[v + "_" + id2type[k]] = 1
        else:
            lg_sptype2count[v + "_" + id2type[k]] += 1
    return id2seq, id2lg, id2type, id2truelbls

def extract_compatible_binaries_predtat(restrict_types=None, return_mcc=False):
    ind2glbl_lbl = {0: 'NO_SP', 1: 'SP', 2: 'TATLIPO', 3: 'LIPO', 4: 'TAT', 5: 'PILIN'}
    glbllbl2_ind = {v: k for k, v in ind2glbl_lbl.items()}
    id2seq, id2lg, id2type, id2truelbls = extract_id2seq_dict(file="train_set.fasta")
    id2seq_b5, id2truelbls_b5, id2lg_b5, id2type_b5 = extract_id2seq_dict(file="benchmark_set_sp5.fasta")
    for k in id2seq_b5.keys():
        if k not in id2seq:
            id2seq[k] = id2seq_b5[k]
            id2truelbls[k] = id2truelbls_b5[k]
            id2lg[k] = id2lg_b5[k]
            id2type[k] = id2type_b5[k]
    with open("sp1_sp2_fastas/predTat_results.txt", "rt") as f:
        lines = f.readlines()
    # ids_ = [l.split("|")[0] for l in lines]
    # print(set([id2type[id_] for id_ in ids_]))
    # exit(1)
    ids = set(id2seq.keys())
    # print(len(ids), lines[0])
    # line = lines[0].replace("#", "")
    # all_ids = set([line.split("_")[0].replace("#", "").replace(" ", "") for line in lines])
    # print(len(all_ids.intersection(ids)))
    # exit(1)
    seq2sptype = {}
    seq2aalbls = {}
    sp_letter = ""
    life_grp, seqs, true_lbls, pred_lbls = [], [], [], []
    restrict_types = restrict_types if restrict_types is not None else ["SP", "TAT", "TATLIPO", "LIPO", "NO_SP"]
    # restrict_types = ["SP", "TAT", "TATLIPO", "LIPO", "NO_SP"]
    for l in lines:
        id = l.split("|")[0]
        if id in ids and id2type[id] in restrict_types:
            if id2type[id] not in ["SP", "TAT", "NO_SP"]:
                print(id, id2type[id])
            life_grp.append(id2lg[id] + "|" + id2type[id])
            seqs.append(id2seq[id])
            true_lbls.append(id2truelbls[id])
            if "Sec signal peptide" in l:
                seq2sptype[id2seq[id]] = glbllbl2_ind['SP']
                sp_letter = "S"
                cs_point = int(l.split("Most likely cleavage site: ")[-1].split("[")[0].split("-")[-1].replace(" ", ""))
            elif "Tat signal peptide" in l:
                seq2sptype[id2seq[id]] = glbllbl2_ind['TAT']
                sp_letter = "T"
                cs_point = int(l.split("Most likely cleavage site: ")[-1].split("[")[0].split("-")[-1].replace(" ", ""))
            else:
                seq2sptype[id2seq[id]] = glbllbl2_ind['NO_SP']
                cs_point = None
            if cs_point is not None:
                lblseq = cs_point * sp_letter + (len(id2seq[id]) - cs_point) * "O"
                seq2aalbls[id2seq[id]] = lblseq
            else:
                seq2aalbls[id2seq[id]] = "O" * len(id2seq[id])
            pred_lbls.append(seq2aalbls[id2seq[id]])
    if return_mcc:
        mcc, mcc2 = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True, sp_type="SP", sptype_preds=seq2sptype)
        # print("computed here", mcc, mcc2)
        # print("reported in sp6", [0.34,0.736,0.839,0.781],[0.238, 0.209, 0.655])
        mcc_tat= get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=False, sp_type="TAT", sptype_preds=seq2sptype)
        # print(mcc_tat)
        return  mcc, mcc_tat
    # all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores = \
    #     get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="TAT",
    #                sptype_preds=seq2sptype)
    # print([round(i, 2) for i in np.array(all_recalls).reshape(-1)])
    # print([round(i, 2) for i in np.array(all_precisions).reshape(-1)])
    # print([round(i, 2) for i in np.array(all_f1_scores).reshape(-1)])
    all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores = \
        get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="SP",
                   sptype_preds=seq2sptype)
    # print([round(i, 2) for i in np.array(all_recalls).reshape(-1)])
    # print([round(i, 2) for i in np.array(all_precisions).reshape(-1)])
    # print([round(i, 2) for i in np.array(all_f1_scores).reshape(-1)])
    return all_recalls, all_precisions, all_f1_scores

def extract_compatible_binaries_lipop(restrict_types=None, return_mcc=False):
    ind2glbl_lbl = {0: 'NO_SP', 1: 'SP', 2: 'TATLIPO', 3: 'LIPO', 4: 'TAT', 5: 'PILIN'}
    glbllbl2_ind = {v:k for k,v in ind2glbl_lbl.items()}
    # id2seq, id2lg, id2type, id2truelbls = extract_id2seq_dict(file="train_set.fasta")
    # id2seq_b5, id2truelbls_b5, id2lg_b5, id2type_b5 = extract_id2seq_dict(file="benchmark_set_sp5.fasta")
    # for k in id2seq_b5.keys():
    #     if k not in id2seq:
    #         id2seq[k] = id2seq_b5[k]
    #         id2truelbls[k] = id2truelbls_b5[k]
    #         id2lg[k] = id2lg_b5[k]
    #         id2type[k] = id2type_b5[k]
    id2seq, id2lg, id2type, id2truelbls = extract_id2seq_dict(file="train_set.fasta")

    with open("sp1_sp2_fastas/results.txt", "rt") as f:
        lines = f.readlines()
    ids = set(id2seq.keys())
    # print(len(ids), lines[0])
    # line = lines[0].replace("#", "")
    # all_ids = set([line.split("_")[0].replace("#", "").replace(" ", "") for line in lines])
    # print(len(all_ids.intersection(ids)))
    # exit(1)
    seq2sptype = {}
    seq2aalbls = {}
    sp_letter = ""
    life_grp, seqs, true_lbls, pred_lbls = [], [], [], []
    restrict_types = restrict_types if restrict_types is not None else ["SP", "TAT", "TATLIPO", "LIPO", "NO_SP"]
    # restrict_types = ["SP", "TAT", "TATLIPO", "LIPO", "NO_SP"]
    for l in lines:
        line = l.replace("#", "")
        id = line.split("_")[0].replace(" ", "")
        if id in ids and id2type[id] in restrict_types:
            # if id2type[id] not in ["SP", "LIPO", "NO_SP"]:
            #     print(id, id2type[id])
            life_grp.append(id2lg[id] + "|" + id2type[id])
            seqs.append(id2seq[id])
            true_lbls.append(id2truelbls[id])

            if "SpI" in line and not "SpII" in line:
                seq2sptype[id2seq[id]] = glbllbl2_ind['SP']
                sp_letter = "S"
            elif "SpII" in line:
                seq2sptype[id2seq[id]] = glbllbl2_ind['LIPO']
                sp_letter = "L"
            else:
                seq2sptype[id2seq[id]] = glbllbl2_ind['NO_SP']
            if "cleavage" in line:
                cs_point = int(line.split("cleavage=")[1].split("-")[0])
                lblseq = cs_point * sp_letter + (len(id2seq[id]) -cs_point) * "O"
                seq2aalbls[id2seq[id]] = lblseq
            else:
                seq2aalbls[id2seq[id]] = "O" * len(id2seq[id])
            pred_lbls.append(seq2aalbls[id2seq[id]])
        else:
            # print(id, id2type[id])
            continue
    # all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores = \
    #     get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="LIPO", sptype_preds=seq2sptype)
    # print([round(i, 2) for i in np.array(all_recalls).reshape(-1)])
    # print([round(i, 2) for i in np.array(all_precisions).reshape(-1)])
    # print([round(i, 2) for i in np.array(all_f1_scores).reshape(-1)])
    if return_mcc:
        mcc, mcc2 = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                             sp_type="SP", sptype_preds=seq2sptype)
        # print("Computed here LIPOP", mcc, mcc2[1:], len(seqs))
        # print("Reported in SP6", [0.196, 0.71, 0.879, 0.733], [0.342, 0.484, 0.552])
        mcc_lipop, mcc2_lipop = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                             sp_type="LIPO", sptype_preds=seq2sptype)
        return mcc, mcc_lipop
    all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores = \
        get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="SP", sptype_preds=seq2sptype)
    # print([round(i, 2) for i in np.array(all_recalls).reshape(-1)])
    # print([round(i, 2) for i in np.array(all_precisions).reshape(-1)])
    # print([round(i, 2) for i in np.array(all_f1_scores).reshape(-1)])
    all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores = \
        get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="SP", sptype_preds=seq2sptype)


    return all_recalls, all_precisions, all_f1_scores
    tp, fn, fp = 0, 0, 0
    different_types = 0
    crct =0
    # SANITY CHECKS
    # for s, pl, tl, lg in zip(seqs, pred_lbls, true_lbls, life_grp):
    #     if "EUKARYA" in lg and "SP" in lg and "NO_SP" not in lg:
    #         if -1 <= pl.rfind("S") - tl.rfind("S") <= 1:
    #             tp += 1
    #         else:
    #             fn += 1
    #     if "LIPO" in lg and "NO_SP" not in lg and pl[0] != tl[0]:
    #         different_types+=1
    #     elif "LIPO" in lg and "NO_SP" not in lg:
    #         crct +=1
    # print(different_types, crct)
    # print(tp/(tp+fn), tp, fn)
    # pickle.dump(seq2sptype, open("lipoP_0_1_best_sptype.bin", "wb"))
    # pickle.dump(seq2aalbls, open("lipoP_0_1.bin", "wb"))
    # for l in lines:

def extract_compatible_binaries_deepsig(restrict_types=None, return_mcc=False):
    ind2glbl_lbl = {0: 'NO_SP', 1: 'SP', 2: 'TATLIPO', 3: 'LIPO', 4: 'TAT', 5: 'PILIN'}
    glbllbl2_ind = {v:k for k,v in ind2glbl_lbl.items()}
    id2seq, id2lg, id2type, id2truelbls = extract_id2seq_dict(file="train_set.fasta")

    # with open("sp1_sp2_fastas/results_deepsig_fullds.txt", "rt") as f:
    with open("sp1_sp2_fastas/results_deepsig_v2.txt", "rt") as f:
        lines = f.readlines()
    seq2sptype = {}
    seq2aalbls = {}
    sp_letter = ""
    life_grp, seqs, true_lbls, pred_lbls = [], [], [], []
    added_seqs = set()
    # restrict_types = restrict_types if restrict_types is not None else ["SP", "TAT", "TATLIPO", "LIPO", "NO_SP"]
    # restrict_types = ["SP", "TAT", "TATLIPO", "LIPO", "NO_SP"]
    for l in lines:
        id = l.split("|")[0]
        if id in id2seq and id2seq[id] not in added_seqs and id2type[id] in restrict_types:
            life_grp.append(id2lg[id] + "|" + id2type[id])
            seqs.append(id2seq[id])
            added_seqs.add(id2seq[id])
            true_lbls.append(id2truelbls[id])

            if "Signal peptide" in l:
                seq2sptype[id2seq[id]] = glbllbl2_ind['SP']
                sp_letter = "S"
                cs_point = int(l.split("Signal peptide")[1].split("\t")[2])
                lblseq = cs_point * sp_letter + (len(id2seq[id]) -cs_point) * "O"
                seq2aalbls[id2seq[id]] = lblseq
            elif id2seq[id] not in seq2sptype:
                seq2sptype[id2seq[id]] = glbllbl2_ind['NO_SP']
                seq2aalbls[id2seq[id]] = "O" * len(id2seq[id])

            pred_lbls.append(seq2aalbls[id2seq[id]])
    count,count2 = 0,0

    seq2id = {v:k for k,v in id2seq.items()}
    # for l,s,a,t in zip(life_grp,    seqs,    added_seqs,    true_lbls):
    #     if l.split("|")[0] == "ARCHAEA":
    #         if id2type[seq2id[s]] == "NO_SP":
    #             count+=1
    #         elif id2type[seq2id[s]] == "TATLIPO":
    #             count2+=1
    # exit(1)
    if return_mcc:
        # print(len(seqs), len(set(seqs)))
        mcc, mcc2 = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True, sp_type="SP", sptype_preds=seq2sptype)
        # print("Computed here", mcc, mcc2[1:])
        # print("Reported in SP6", [0.792,0.735,0.798,0.711], [0.159,0.146,"n.d."])
        if return_mcc:
            return  mcc
        # print(all_recalls, all_precisions)
        # exit(1)

    all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores = \
        get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="SP",
                   sptype_preds=seq2sptype)
    # print([round(i, 2) for i in np.array(all_recalls).reshape(-1)])
    # print([round (i, 2) for i in np.array(all_precisions).reshape(-1)])
    # print([round (i, 2) for i in np.array(all_f1_scores).reshape(-1)])
    tp, fn, fp = 0, 0, 0
    different_types = 0
    crct =0
    return all_recalls, all_precisions, all_f1_scores
    # Just some sanity checks here
    # for s, pl, tl, lg in zip(seqs, pred_lbls, true_lbls, life_grp):
    #     if "EUKARYA" in lg and "SP" in lg and "NO_SP" not in lg:
    #         if -1 <= pl.rfind("S") - tl.rfind("S") <= 1:
    #             tp += 1
    #         else:
    #             fn += 1
        # if "LIPO" in lg and "NO_SP" not in lg and pl[0] != tl[0]:
        #     different_types+=1
        # elif "LIPO" in lg and "NO_SP" not in lg:
        #     crct +=1
    # print(different_types, crct)
    # print(tp/(tp+fn), tp, fn)
    # pickle.dump(seq2sptype, open("lipoP_0_1_best_sptype.bin", "wb"))
    # pickle.dump(seq2aalbls, open("lipoP_0_1.bin", "wb"))
    # for l in lines:

def extract_compatible_phobius_binaries(restrict_types=["SP", "NO_SP"], return_mcc=False):
    ind2glbl_lbl = {0: 'NO_SP', 1: 'SP', 2: 'TATLIPO', 3: 'LIPO', 4: 'TAT', 5: 'PILIN'}
    glbllbl2_ind = {v:k for k,v in ind2glbl_lbl.items()}
    file = "sp1_sp2_fastas/results_phobius.txt"
    life_grp, seqs, true_lbls, pred_lbls, added_seqs = [], [], [], [], []
    seq2sptype = {}
    seq2aalbls = {}
    id2seq, id2lg, id2type, id2truelbls = extract_id2seq_dict(file="train_set.fasta")
    with open(file, "rt") as f:
        lines = f.readlines()
    count = 0
    retain_nextl_flag = False
    sp_start, sp_end = -1, -1
    first_seq = True
    id2info = {}
    for l in lines:
        if "ID" in l:
            id = l.split(" ")[-1]
            if id.split("|")[0] in id2seq and id.split("|")[2] in restrict_types:
                id2info[id.split("|")[0]] = []
        else:
            if id.split("|")[0] in id2info:
                id2info[id.split("|")[0]].append(l)
    for k, v in id2info.items():
        actual_id = k
        if id2seq[actual_id] not in added_seqs:
            life_grp.append(id2lg[actual_id] + "|" + id2type[actual_id])
            seqs.append(id2seq[actual_id])
            added_seqs.append(id2seq[actual_id])
            true_lbls.append(id2truelbls[actual_id])
            if "SIGNAL" in v[0]:
                seq2sptype[id2seq[actual_id]] = glbllbl2_ind['SP']
                cs_position = int(v[0].replace("\n", "").split(" ")[-1])
                predicted = "S" * cs_position + "I" * (len(seqs[-1]) - cs_position)
            else:
                predicted  = "I" * len(seqs[-1])
                seq2sptype[id2seq[actual_id]] = glbllbl2_ind['NO_SP']
            pred_lbls.append(predicted)

    if return_mcc:
        mcc, mcc2 = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True, sp_type="SP", sptype_preds=seq2sptype)
        # print(len(seqs))
        # print("Computed here", mcc, mcc2[1:])
        # print("Reported in SP6", [0.531,0.766,0.716,0.796], [0.766,0.716,0.551])
        return mcc
    all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores = \
        get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="SP",
                   sptype_preds=seq2sptype)
    # all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores = \
    #     get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="SP", sptype_preds=seq2sptype)
    return all_recalls, all_precisions, all_f1_scores
    # common_sp6_phob_seqs = extract_phobius_trained_data()
    # remove_inds = [i for i in range(len(seqs)) if seqs[i] in common_sp6_phob_seqs]
    # remove_inds = []
    # len_ = len(seqs)
    # life_grp = [life_grp[i] for i in range(len(seqs)) if i not in remove_inds]
    # true_lbls = [true_lbls[i] for i in range(len(seqs)) if i not in remove_inds]
    # pred_lbls = [pred_lbls[i] for i in range(len(seqs)) if i not in remove_inds]
    # seqs = [seqs[i] for i in range(len_) if i not in remove_inds]
    # all_recalls, all_precisions, total_positives, false_positives, predictions, all_f1_scores = \
    #     get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="SP", sptype_preds=seq2sptype)
    # print(all_recalls, all_precisions)

def extract_phobius_trained_data():
    folder = "/home/alex/Desktop/work/phobius_data"
    files = os.listdir(folder)
    ids = []
    seqs = []
    for f in files:
        for seq_record in SeqIO.parse(os.path.join(folder,f), "fasta"):
            ids.append(seq_record.description.split(" ")[1])
            seq = seq_record.seq[:len(seq_record.seq) // 2]
            seq_70aa = seq[:70]
            seqs.append(seq_70aa)
            # print(str(seq_record.seq(:len(seq_record) //2)))
    id2seq, id2truelbls, id2lg, id2type = extract_id2seq_dict()
    return set(seqs).intersection(id2seq.values())

def remove_non_unique():
    file = "../sp_data/sp6_data/train_set.fasta"
    unqiue_seqs_2_info = {}
    count = 0
    for seq_record in SeqIO.parse(file, "fasta"):
        if str(seq_record.seq[:len(seq_record.seq) // 2]) in unqiue_seqs_2_info:
            count += 1
            already_added_id = unqiue_seqs_2_info[str(seq_record.seq[:len(seq_record.seq) // 2])][0]
            already_added_lbl  = unqiue_seqs_2_info[str(seq_record.seq[:len(seq_record.seq) // 2])][1]
            # print("_".join(already_added_id.split("|")[1:]), "_".join(seq_record.id.split("|")[1:]))
            # if "_".join(already_added_id.split("|")[1:]) != "_".join(seq_record.id.split("|")[1:]):
            #     print(already_added_id, seq_record.id)
            if (already_added_id.split("|")[2] == "NO_SP" or  seq_record.id.split("|")[2] == "NO_SP") and \
                already_added_lbl != seq_record.seq[len(seq_record.seq) // 2:]:
                #"_".join(already_added_id.split("|")[1:]) != "_".join(seq_record.id.split("|")[1:]):
                print("\n")
                print(already_added_id , seq_record.id )
                print(already_added_lbl)
                print(seq_record.seq[len(seq_record.seq) //2 :])
                print("\n")
        #     print(unqiue_seqs_2_info[str(seq_record.seq[:len(seq_record.seq) // 2])], seq_record.id)
        unqiue_seqs_2_info[str(seq_record.seq[:len(seq_record.seq) // 2])] = (seq_record.id, seq_record.seq[len(seq_record.seq)//2:])
    print(count)
    # import glob
    # unique_seq2info = {}
    # files = glob.glob("../sp_data/sp6_partitioned_data_sublbls*")
    # for f in files:
    #     items = pickle.load(open(f, "rb"))
    #     for k, v in items.items():
    #         if k in unique_seq2info:
    #             print(unique_seq2info[k])
    #             print((v[1:], f), "\n\n",)
    #         else:
    #             unique_seq2info[k] = (v[1:], f)

def pred_lipos():
    for tr_f in [2]:
        for t_s in ['train']:

            f = "../sp_data/sp6_partitioned_data_{}_{}.bin".format(t_s, tr_f)
            a = pickle.load(open(f, "rb"))
            for k,v in a.items():
                if v[-1] == "TATLIPO":
                    print("ok, wrong")

def plot_performance():
    sp1_euk_0tol_ = [0.537, 0.654, 0.693, 0.7]
    sp1_euk_3tol_ = [0.646, 0.786, 0.779, 0.78]
    sp1_neg_0tol_ = [0.365, 0.454, 0.493, 0.58]
    sp1_neg_3tol_ = [0.486, 0.589, 0.606, 0.66]
    sp1_pos_0tol_ = [0.103, 0.2, 0.486, 0.71]
    sp1_pos_3tol_ = [0.256, 0.233, 0.541, 0.71]
    sp1_arch_0tol_ = [0.256, 0.493, 0.533, 0.56]
    sp1_arch_3tol_ = [0.531, 0.667, 0.667, 0.66]
    all_sp1s = [sp1_euk_0tol_, sp1_euk_3tol_, sp1_neg_0tol_, sp1_neg_3tol_, sp1_pos_0tol_, sp1_pos_3tol_, sp1_arch_0tol_, sp1_arch_3tol_]

    sp2_neg_0tol_ = [0.811, 0.878, 0.896, 0.88]
    sp2_neg_3tol_ = [0.852, 0.896, 0.92, 0.89]
    sp2_pos_0tol_ = [0.66, 0.845, 0.93, 0.9]
    sp2_pos_3tol_ = [0.762, 0.858, 0.936, 0.91]
    sp2_arch_0tol_ = [0.571, 0.571, 0.706, 0.67]
    sp2_arch_3tol_ = [0.571, 0.571, 0.706,0.67]
    all_sp2s = [sp2_neg_0tol_,sp2_neg_3tol_, sp2_pos_0tol_, sp2_pos_3tol_, sp2_arch_0tol_, sp2_arch_3tol_]

    tat_neg_0tol_ = [0.397, 0.52, 0.556, 0.69]
    tat_neg_3tol_ = [0.661, 0.693, 0.857, 0.79]
    tat_pos_0tol_ = [0.244, 0.205, 0.2, 0.63]
    tat_pos_3tol_ = [0.39, 0.308, 0.8, 0.75]
    tat_arch_0tol_ = [0.348, 0.3, 0.453, 0.35]
    tat_arch_3tol_ = [0.522, 0.5, 0.453, 0.47]
    all_tats = [tat_neg_0tol_, tat_neg_3tol_, tat_pos_0tol_, tat_pos_3tol_, tat_arch_0tol_, tat_arch_3tol_]

    line_w = 0.15
    x_positions = []
    names = ["No-Tuning", "Pre-Tuning", "Tuning+training", "SP6"]
    colors = ["red", "blue", "green", "black"]
    names_xticks = ["euk", "neg", "pos", "arch"]
    for i in range(1, 9):
        x_positions.append([i-1.5*line_w, i-0.5*line_w, i+0.5*line_w, i+line_w*1.5])
    for ind, (xpos, heights) in enumerate(zip(x_positions, all_sp1s)):
        for i, (xp, h) in enumerate(zip(xpos, heights)):
            if ind == 7:
                plt.bar(xp, h, width=line_w, label=names[i], color=colors[i])
            else:
                plt.bar(xp, h, width=line_w, color=colors[i])
    plt.ylim(0,1)
    plt.legend()
    plt.title("Sec/SPI")
    plt.ylabel("CS-F1 performance")
    plt.xticks(list(range(1,9)),[names_xticks[i//2] + " tol0" if i%2 ==0 else names_xticks[i//2]+ " tol3" for i in range(8)])
    plt.show()

    x_positions = []
    names = ["No-Tuning", "Pre-Tuning", "Tuning+training", "SP6"]
    colors = ["red", "blue", "green", "black"]
    names_xticks = ["neg", "pos", "arch"]
    for i in range(1, 7):
        x_positions.append([i-1.5*line_w, i-0.5*line_w, i+0.5*line_w, i+line_w*1.5])
    for ind, (xpos, heights) in enumerate(zip(x_positions, all_sp2s)):
        for i, (xp, h) in enumerate(zip(xpos, heights)):
            if ind == 5:
                plt.bar(xp, h, width=line_w, label=names[i], color=colors[i])
            else:
                plt.bar(xp, h, width=line_w, color=colors[i])
    plt.ylim(0,1)
    plt.legend()
    plt.xticks(list(range(1,7)),[names_xticks[i//2] + " tol0" if i%2 ==0 else names_xticks[i//2]+ " tol3" for i in range(6)])
    plt.title("Sec/SPII")
    plt.ylabel("CS-F1 performance")
    plt.show()

    x_positions = []
    names = ["No-Tuning", "Pre-Tuning", "Tuning+training", "SP6"]
    colors = ["red", "blue", "green", "black"]
    names_xticks = ["neg", "pos", "arch"]
    for i in range(1, 7):
        x_positions.append([i-1.5*line_w, i-0.5*line_w, i+0.5*line_w, i+line_w*1.5])
    for ind, (xpos, heights) in enumerate(zip(x_positions, all_tats)):
        for i, (xp, h) in enumerate(zip(xpos, heights)):
            if ind == 5:
                plt.bar(xp, h, width=line_w, label=names[i], color=colors[i])
            else:
                plt.bar(xp, h, width=line_w, color=colors[i])
    plt.ylim(0,1)
    plt.legend()
    plt.xticks(list(range(1,7)),[names_xticks[i//2] + " tol0" if i%2 ==0 else names_xticks[i//2]+ " tol3" for i in range(6)])
    plt.title("TAT")
    plt.ylabel("CS-F1 performance")
    plt.show()

    # for pf in plot_for:
    #     total = lgandsptype2count_total[pf]
    #     totals.append(total)
    #     heights.extend([lgandsptype2counts[0][pf] /total, lgandsptype2counts[1][pf] /total, lgandsptype2counts[2][pf] /total])
    # plt.bar([x_positions[i*3] for i in range(8)], [heights[i*3] for i in range(8)], width=line_w, label="partition 1")
    # plt.bar([x_positions[i*3+1] for i in range(8)], [heights[i*3+1] for i in range(8)], width=line_w, label="partition 2")
    # plt.bar([x_positions[i*3+2] for i in range(8)], [heights[i*3+2] for i in range(8)], width=line_w, label="partition 3")
    # plt.legend()
    # plt.xticks(list(range(1,9)),[plot_for[i] + "\n" + str(totals[i]) for i in range(8)])
    # plt.xlabel("Life group, SP/NO-SP; No. of datapoints")
    # plt.ylabel("Percentage from that life group")
    # plt.show()

def plot_sp6_vs_tnmt_violin():
    import matplotlib.patches as mpatches
    all_mdl_2results = []
    sp1_f1s, sp1_recs, sp1_precs, sp2_f1s, sp2_recs, sp2_precs, tat_f1s, \
    tat_recs, tat_precs, mcc1_sp1, mcc2_sp1, mcc1_sp2, mcc2_sp2, mcc1_tat, mcc2_tat = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    runs = 4
    # for run_no in [1,1]:
    for run_no in [21,22,23]:
        print("Computing results for run number {}".format(run_no))
        run_results_folder = "tuning_bert_fixed_high_lr_swa_only_repeat{}/".format(run_no)
        mdl2results = extract_all_param_results(only_cs_position=False,
                                                result_folder=run_results_folder,
                                                compare_mdl_plots=False,
                                                remove_test_seqs=False,
                                                benchmark=True,
                                                prints=False)
        mdl_ind = 0
        sp1_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-5])]))
        sp1_recs.append(np.array([rec for rec in mdl2results[mdl_ind][6]]))
        sp1_precs.append(np.array([rec for rec in mdl2results[mdl_ind][7]]))
        sp2_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-4])]))
        sp2_recs.append(np.array([rec for rec in mdl2results[mdl_ind][8]]))
        sp2_precs.append(np.array([rec for rec in mdl2results[mdl_ind][9]]))
        tat_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-3])]))
        tat_f1s.append(np.array([rec for rec in mdl2results[mdl_ind][10]]))
        tat_precs.append(np.array([rec for rec in mdl2results[mdl_ind][11]]))
        mcc1_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][0]]))
        mcc2_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][1][1:]]))
        mcc1_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][2]]))
        mcc2_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][3]]))
        mcc1_tat.append(np.array([mcc for mcc in mdl2results[mdl_ind][4]]))
        mcc2_tat.append([mcc for mcc in mdl2results[mdl_ind][5]])
    tnmt_f1 = [[0.692, 0.737, 0.769, 0.782 ], [0.462, 0.564, 0.59, 0.59 ], [0.526, 0.684, 0.684, 0.684],
               [0.606, 0.697, 0.667, 0.727]]
    tnmt_f1_sp2 = [[0.906, 0.912, 0.914, 0.917] , [0.927 , 0.933 , 0.933 , 0.933] , [0.75 , 0.75 , 0.75 , 0.75]]
    tnmt_f1_tat = [[0.613 , 0.79 , 0.806 , 0.855] , [0.634 , 0.732 ,  0.829 ,  0.829] , [ 0.3 , 0.5 , 0.7 , 0.7]]
    sp6_recalls_sp1 = [0.747, 0.774, 0.808, 0.829, 0.639, 0.672, 0.689, 0.721, 0.800, 0.800, 0.800, 0.800, 0.500, 0.556,
                       0.556, 0.583]
    sp6_recalls_sp2 = [0.852, 0.852, 0.856, 0.864, 0.875, 0.883, 0.883, 0.883, 0.778, 0.778, 0.778, 0.778]
    sp6_recalls_tat = [0.706, 0.765, 0.784, 0.804, 0.556, 0.556, 0.667, 0.667, 0.333, 0.444, 0.444, 0.444]
    sp6_precs_sp1 = [0.661, 0.685, 0.715, 0.733, 0.534, 0.562, 0.575, 0.603, 0.632, 0.632, 0.632, 0.632, 0.643, 0.714,
                     0.714, 0.75]
    sp6_precs_sp2 = [0.913, 0.913, 0.917, 0.925, 0.929, 0.938, 0.938, 0.938, 0.583, 0.583, 0.583, 0.583]
    sp6_precs_tat = [0.679, 0.736, 0.755, 0.774, 0.714, 0.714, 0.857, 0.857, 0.375, 0.5, 0.5, 0.5]
    sp6_f1_sp1 = get_f1_scores(sp6_recalls_sp1, sp6_precs_sp1)
    sp6_f1_sp2 = get_f1_scores(sp6_recalls_sp2, sp6_precs_sp2)
    sp6_f1_tat = get_f1_scores(sp6_recalls_tat, sp6_precs_tat)
    sp6_results = [sp6_f1_sp1, sp6_f1_sp2, sp6_f1_tat]
    arrange_tol_lg_sp1 = []
    for lg_ind in [0, 4, 8, 12]:
        arrange_tol_lg_ = []
        for tol in range(4):
            arrange_tol_lg_ = [sp1_f1s[run_no][lg_ind+tol] for run_no in range(runs-1)]
            arrange_tol_lg_sp1.append(arrange_tol_lg_)
    arrange_tol_lg_sp2 = []
    for lg_ind in [0, 4, 8]:
        arrange_tol_lg_ = []
        for tol in range(4):
            arrange_tol_lg_ = [sp2_f1s[run_no][lg_ind+tol] for run_no in range(runs-1)]
            arrange_tol_lg_sp2.append(arrange_tol_lg_)
    arrange_tol_lg_tat = []
    for lg_ind in [0, 4, 8]:
        arrange_tol_lg_ = []
        for tol in range(4):
            arrange_tol_lg_ = [tat_f1s[run_no][lg_ind+tol] for run_no in range(runs-1)]
            arrange_tol_lg_tat.append(arrange_tol_lg_)
    arrange_sptype_tol_lg = [arrange_tol_lg_sp1, arrange_tol_lg_sp2, arrange_tol_lg_tat]
    print(arrange_tol_lg_sp1, len(arrange_tol_lg_sp2))
    print(arrange_tol_lg_sp2, len(arrange_tol_lg_sp2))
    print(arrange_tol_lg_tat, len(arrange_tol_lg_sp2))
    names = ["TSignal", "SignalP6", "LipoP", "DeepSig", "Phobius"]
    colors = ["c", "orage", "green", "black", "purple"]
    titles = ["eukarya", "gn-bacteria", "gp-bacteria", "archaea"]
    x_positions = []
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"
    fig, ax = plt.subplots(3, 4)
    line_w = 0.3
    offsets = [-line_w * 0.5, line_w * 0.5]
    sptypes = ["Sec/SPI", "Sec/SPII", "Tat/SP1"]
    for ind in range(3):
        for ind2 in range(4):
            if ind2 == 0:
                ax[ind, ind2].set_ylabel("F1 score\n{}".format(sptypes[ind]), fontsize=12.5)
                ax[ind, ind2].yaxis.set_label_coords(-0.6, 0.42)
            if ind !=0 and ind2 == 0:
                if ind ==2:
                    ax[ind,ind2].set_xlabel(titles[ind2],fontsize=12.5)
                box = ax[ind, ind2].get_position()
                ax[ind, ind2].set_position([box.x0+box.width * 0.15, box.y0 + box.height * 0.65, box.width * 0.8, box.height * 0.75])
                ax[ind, ind2].set_xticks([0, 1])
                ax[ind, ind2].set_xticklabels([" ", " "], fontsize=12.5)
                ax[ind, ind2].set_yticks([0, 1])
                ax[ind, ind2].set_yticklabels([" ", " "], fontsize=12.5)

                # ax[ind, ind2].tick_params( axis='y', which='both', bottom=False, top=False, labelbottom=False)
                # ax[ind, ind2].tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False)
                # ax[ind, ind2].tick_params( axis='y', which='both', bottom=False, top=False, labelbottom=False)
                continue

            second_index = ind2 if ind == 0 else ind2-1
            ax[ind, ind2].violinplot(arrange_sptype_tol_lg[ind][second_index*4:(second_index+1)*4], positions=list(range(1,5)), widths=0.5, showmeans=True)
            ranges = range(1,5)
            min_max_val = np.array(arrange_sptype_tol_lg[ind][second_index*4:(second_index+1)*4]).reshape(-1)
            min_max_val = np.append(min_max_val, sp6_results[ind][second_index * 4])
            min_max_val = np.append(min_max_val, sp6_results[ind][second_index * 4 + 3])
            min_, max_ = np.min(min_max_val), np.max(min_max_val)
            for idn, (region, sp6result) in enumerate(zip(ranges, sp6_results[ind][second_index*4:(second_index+1)*4])):
                if idn == 0:
                    ax[ind, ind2].plot([region-0.25, region+0.25], [sp6result, sp6result], linewidth=0.5, color='orange', label='SP6 results')
                else:
                    ax[ind, ind2].plot([region-0.25, region+0.25], [sp6result, sp6result], linewidth=0.5, color='orange')
            box = ax[ind,ind2].get_position()
            ax[ind,ind2].set_xlim(0.5, 4.5)
            ax[ind,ind2].set_position([box.x0+box.width * 0.15, box.y0 + box.height * 0.65, box.width * 0.8, box.height * 0.75])
            ax[ind,ind2].set_ylim([min_-(max_-min_)*0.1, max_+(max_-min_)*0.1])

            # ax[ind].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # ax[ind].legend(loc='center left', bbox_to_anchor=(1, 0.5, ), fontsize=26)
            ax[ind,ind2].set_xticks([1,2,3,4])
            if ind == 2 and ind2 == 2:
                handles, labels = ax[ind, ind2].get_legend_handles_labels()
            ax[ind,ind2].set_xticklabels([0,1,2,3], fontsize=12.5)
            ax[ind,ind2].set_yticks([min_, (min_+max_)/2, max_])
            ax[ind,ind2].set_yticklabels([round(min_,2), round((min_+max_)/2, 2), round(max_,2)], fontsize=12.5)
            if ind == 2:
                ax[ind, ind2].set_xlabel(titles[ind2], fontsize=12.5)
        # if ind == 0:
        #     ax[ind].set_title("Weighted F1 scores for TSignal/SignalP6: 0.8132/0.7976", fontsize=12.5, y=1.1)
    handles.append(mpatches.Patch(color='#1f77b4'))
    labels.append("TSignal")
    fig.text(0.5, 0.04, 'Tolerance/Life group', ha='center', fontsize=12.5)
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.01, 0.05), ncol=2, fontsize=12.5)
    fig.suppressComposite = False
    from matplotlib.patches import RegularPolygon, Rectangle, Patch, Arrow
    mini_line = Arrow(569, 100, 0, 100 + 800, color='black', alpha=1)
    # dotted_line(569,100, patches=fig.patches)
    # dotted_line(910,100, patches=fig.patches)
    # dotted_line(1251,100, patches=fig.patches)
    # ax[2].set_xlabel("eukarya                    gn bacteria             gp bacteria                    archaea\n\n "
    #                  "tolerance/life group", fontsize=12.5)

    # with plt.rc_context({'image.composite_image': False}):
    #     fig.savefig('t.pdf', dpi=350)

    plt.savefig("some_plot.pdf")
    exit(1)

def print_tables(sp1rec, sp1prec, sp2rec,sp2prec,tatrec,tatprec,sp6_sp1rec, sp6_sp1prec, sp6_sp2rec,sp6_sp2prec,sp6_tatrec,sp6_tatprec):
    def print_vecs_as_tbls(vec1,vec2,vec3):
        string1, string2 = "", ""
        for v1, v2, v3 in zip(vec1,vec2, vec3):
            v1, v2,v3= round(v1,3), round(v2,3), round(v3,3)
            if v1 > v2:
                string1 += "& $\mathbf{" + str(v1) + " \pm " + str(v3) + "}$ "
                string2 += "& $" + str(v2) +"$ "
            elif v1< v2:
                string1 += "& $"+str(v1) + " \pm " + str(v3) + "$"
                string2 += "& $\mathbf{" + str(v2) + "}$ "
            else:
                string1 += "& $\mathbf{" + str(v1) + " \pm " + str(v3) + "}$ "
                string2 += "& $\mathbf{" + str(v2) + "}$ "
        print("TSignal:", string1)
        print("SP6:", string2)

    mean_sp1rec, mean_sp1prec, \
        mean_sp2rec,mean_sp2prec,mean_tatrec,mean_tatprec = np.mean(np.stack(sp1rec), axis=0), np.mean(np.stack(sp1prec), axis=0), \
                                                            np.mean(np.stack(sp2rec),axis=0), np.mean(np.stack(sp2prec), axis=0), \
                                                            np.mean(np.stack(tatrec),axis=0), np.mean(np.stack(tatprec),axis=0)
    std_sp1rec, std_sp1prec, \
            std_sp2rec,std_sp2prec,std_tatrec,std_tatprec = np.std(np.stack(sp1rec), axis=0), np.std(np.stack(sp1prec), axis=0), \
                                                                np.std(np.stack(sp2rec),axis=0), np.std(np.stack(sp2prec), axis=0), \
                                                                np.std(np.stack(tatrec),axis=0), np.std(np.stack(tatprec),axis=0)
    print("\n")
    print("Recall sp1")
    print_vecs_as_tbls(mean_sp1rec, sp6_sp1rec, std_sp1rec)
    print("Precision sp1")
    print_vecs_as_tbls(mean_sp1prec, sp6_sp1prec, std_sp1prec)

    print("\n")
    print("Recall sp2")
    print_vecs_as_tbls(mean_sp2rec, sp6_sp2rec, std_sp2rec)
    print("Precision sp2")
    print_vecs_as_tbls(mean_sp2prec, sp6_sp2prec, std_sp2prec)

    print("\n")
    print("Recall tat")
    print_vecs_as_tbls(mean_tatrec, sp6_tatrec, std_tatrec)
    print("Precision tat")
    print_vecs_as_tbls(mean_tatprec, sp6_tatprec, std_tatprec)

def plot_sp6_vs_tnmt(result_folders=("only_decoder_tune_bert_extraOhOnOut_swa_run_1",
                                     "only_decoder_tune_bert_extraOhOnOut_swa_run_2",
                                     "only_decoder_tune_bert_extraOhOnOut_swa_run_3",
                                     "only_decoder_tune_bert_extraOhOnOut_swa_run_4",
                                     "only_decoder_tune_bert_extraOhOnOut_swa_run_5")):
    """
    Method for bar plots with mean and variance on CS performance results for Sec/SPase I and II and Tat/SPase I compared
    to SignalP 6.0

    :param list_or_tuple result_folders: contains the folders containing the sequence prediction binaries. Multiple
    runs are averaged and result variance is also computed, but the method may be used for a single run also (e.g.
    have result_folders = ["one_single_folder_with_results"]
    """
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"

    sp1_f1s, sp1_recs, sp1_precs, sp2_f1s, sp2_recs, sp2_precs, tat_f1s, \
    tat_recs, tat_precs, mcc1_sp1, mcc2_sp1, mcc1_sp2, mcc2_sp2, mcc1_tat, mcc2_tat = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for run_foldername in result_folders:
        print("Computing results for run {}".format(run_foldername))
        mdl2results = extract_all_param_results(only_cs_position=False,
                                                result_folder=run_foldername,
                                                compare_mdl_plots=False,
                                                remove_test_seqs=False,
                                                benchmark=True,
                                                prints=False)
        mdl_ind = 0
        sp1_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-7])]))
        sp1_recs.append(np.array([rec for rec in mdl2results[mdl_ind][10]]))
        sp1_precs.append(np.array([rec for rec in mdl2results[mdl_ind][11]]))
        sp2_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-6])]))
        sp2_recs.append(np.array([rec for rec in mdl2results[mdl_ind][12]]))
        sp2_precs.append(np.array([rec for rec in mdl2results[mdl_ind][13]]))
        tat_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-5])]))
        tat_recs.append(np.array([rec for rec in mdl2results[mdl_ind][14]]))
        tat_precs.append(np.array([rec for rec in mdl2results[mdl_ind][15]]))
        mcc1_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][0]]))
        mcc2_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][1][1:]]))
        mcc1_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][2]]))
        mcc2_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][3]]))
        mcc1_tat.append(np.array([mcc for mcc in mdl2results[mdl_ind][4]]))
        mcc2_tat.append([mcc for mcc in mdl2results[mdl_ind][5]])
    arrange_tol_lg_sp1 = []
    for og_ind in [0, 4, 8, 12]:
        for tol in range(4):
            arrange_tol_lg_ = [sp1_f1s[run_no][og_ind + tol] for run_no in range(len(result_folders))]
            arrange_tol_lg_sp1.append(arrange_tol_lg_)
    arrange_tol_lg_sp2 = []
    for og_ind in [0, 4, 8]:
        for tol in range(4):
            arrange_tol_lg_ = [sp2_f1s[run_no][og_ind + tol] for run_no in range(len(result_folders))]
            arrange_tol_lg_sp2.append(arrange_tol_lg_)
    arrange_tol_lg_tat = []
    for og_ind in [0, 4, 8]:
        for tol in range(4):
            arrange_tol_lg_ = [tat_f1s[run_no][og_ind + tol] for run_no in range(len(result_folders))]
            arrange_tol_lg_tat.append(arrange_tol_lg_)
    arrange_sptype_tol_lg = [np.array(arrange_tol_lg_sp1), np.array(arrange_tol_lg_sp2), np.array(arrange_tol_lg_tat)]

    # recall and precision results are retrieved from SignalP 6.0 reported results (from the supplementary material)
    sp6_recalls_sp1 = [0.747, 0.774, 0.808, 0.829, 0.639, 0.672, 0.689, 0.721, 0.800, 0.800, 0.800, 0.800, 0.500, 0.556,
                       0.556, 0.583]
    sp6_recalls_sp2 = [0.852, 0.852, 0.856, 0.864, 0.875, 0.883, 0.883, 0.883, 0.778, 0.778, 0.778, 0.778]
    sp6_recalls_tat = [0.706, 0.765, 0.784, 0.804, 0.556, 0.556, 0.667, 0.667, 0.333, 0.444, 0.444, 0.444]
    sp6_precs_sp1 = [0.661, 0.685, 0.715, 0.733, 0.534, 0.562, 0.575, 0.603, 0.632, 0.632, 0.632, 0.632, 0.643, 0.714,
                     0.714, 0.75]
    sp6_precs_sp2 = [0.913, 0.913, 0.917, 0.925, 0.929, 0.938, 0.938, 0.938, 0.583, 0.583, 0.583, 0.583]
    sp6_precs_tat = [0.679, 0.736, 0.755, 0.774, 0.714, 0.714, 0.857, 0.857, 0.375, 0.5, 0.5, 0.5]
    sp6_f1_sp1 = get_f1_scores(sp6_recalls_sp1, sp6_precs_sp1)
    sp6_f1_sp2 = get_f1_scores(sp6_recalls_sp2, sp6_precs_sp2)
    sp6_f1_tat = get_f1_scores(sp6_recalls_tat, sp6_precs_tat)


    all_sptypes_all_mean = [np.mean(arrange_sptype_tol_lg[0],axis=1), np.mean(arrange_sptype_tol_lg[1], axis=1), np.mean(arrange_sptype_tol_lg[2], axis=1)]
    all_sptypes_all_std = [np.std(arrange_sptype_tol_lg[0],axis=1), np.std(arrange_sptype_tol_lg[1], axis=1), np.std(arrange_sptype_tol_lg[2], axis=1)]

    all_f1s_sp1 = [np.array(all_sptypes_all_mean[0]).reshape(-1), np.array([sp6_f1_sp1[i * 4:(i + 1) * 4] for i in range(4)]).reshape(-1)]
    all_f1s_sp2 = [np.array(all_sptypes_all_mean[1]).reshape(-1), np.array([sp6_f1_sp2[i*4:(i+1)*4] for i in range(3)]).reshape(-1)]
    all_f1s_tat = [np.array(all_sptypes_all_mean[2]).reshape(-1), np.array([sp6_f1_tat[i*4:(i+1)*4] for i in range(3)]).reshape(-1)]
    all_sptypes_all_f1s = [all_f1s_sp1, all_f1s_sp2, all_f1s_tat]

    names = ["TSignal", "SignalP 6.0", "LipoP", "DeepSig", "Phobius"]
    colors = ["mediumblue", "saddlebrown", "green", "black", "purple","red"]
    titles = ["", "", "", ""]

    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"
    fig, ax = plt.subplots(3, 1,figsize=(8, 6),dpi=350)
    line_w = 0.3
    offsets = [-line_w*0.5, line_w*0.5]
    sptypes=["Sec/SPase I", "Sec/SPase II", "Tat/SPase I"]
    for ind in range(3):
        upper_lim = 17 if ind == 0 else 13
        lower_lim = 0 if ind == 0 else 1
        lower_lim_plots = 1 if ind == 0 else 5
        all_f1s = all_sptypes_all_f1s[ind]
        ax[ind].plot([4.5,4.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        ax[ind].plot([8.5,8.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        ax[ind].plot([12.5,12.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        for j in range(2):
            ax[ind].bar([i + offsets[j] for i in range(lower_lim_plots, 17)], all_f1s[j],  label=names[j],
                        width=line_w,alpha=0.6, color=colors[j])
        # error plots (below, if std is zero - e.g. when having a single run - plot a "dot" from low-low+0.001)
        for i in range(lower_lim_plots, 17):

            low,high = all_sptypes_all_mean[ind][i - lower_lim_plots] - 2 * all_sptypes_all_std[ind][i - lower_lim_plots], \
                       all_sptypes_all_mean[ind][i - lower_lim_plots] + 2 *  all_sptypes_all_std[ind][i - lower_lim_plots]
            ax[ind].plot([i+offsets[0],i+offsets[0]],[low,max(high,low+0.001)], color='black')
        box = ax[ind].get_position()
        ax[ind].set_xlim(0.5,16.5)
        ax[ind].set_position([box.x0, box.y0 + box.height * 0.35, box.width * 1, box.height * 0.95])
        ax[ind].set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax[ind].set_ylim([0, 1.1])
        ax[ind].grid(axis='y',color='black', linestyle='-', linewidth=0.5,alpha=0.4)
        ax[ind].set_xticks(list(range(lower_lim_plots, 17)))
        if ind == 2:
            handles, labels = ax[ind].get_legend_handles_labels()
        ax[ind].set_xticklabels(['{}{}'.format(titles[lower_lim + i//4], i%4) for i in range(upper_lim-1)], fontsize=12.5)
        ax[ind].set_ylabel("F1 score\n{}".format(sptypes[ind]), fontsize=12.5)
        ax[ind].yaxis.set_label_coords(-0.07, 0.42)
        ax[ind].set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=12.5)
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.0, 0.05), ncol=2, fontsize=12.5)
    fig.suppressComposite = False
    ax[2].set_xlabel("eukarya                 gn bacteria                  gp bacteria                 archaea\n\n "
                     "tolerance/life group",fontsize=12.5)
    plt.savefig("some_plot.pdf")

    # repeat 4 times for each tolerance level
    no_of_seqs_sp1 = np.array([146, 61, 15, 36]).repeat(4)
    no_of_seqs_sp2 = np.array([257, 120, 9]).repeat(4)
    no_of_seqs_tat = np.array([51, 18, 9]).repeat(4)
    no_of_tested_sp_seqs = sum([146, 61, 15, 36]) + sum([257, 120, 9]) + sum([51, 18, 9])

    print("Mean weighted F1 score across all SP types, organism groups and tolerance levels TSignal: ", (  np.sum(all_sptypes_all_mean[0]*no_of_seqs_sp1) +
                                         np.sum(all_sptypes_all_mean[1]*no_of_seqs_sp2) +
                                         np.sum(all_sptypes_all_mean[2]*no_of_seqs_tat) ) / np.sum(no_of_tested_sp_seqs*4))
    print("Mean weighted F1 score across all SP types, organism groups and tolerance levels SP6: ", (np.sum(sp6_f1_sp1 * no_of_seqs_sp1) +
                                                           np.sum(sp6_f1_sp2 * no_of_seqs_sp2) +
                                                           np.sum(sp6_f1_tat * no_of_seqs_tat)) / np.sum(no_of_tested_sp_seqs * 4))



def bar_plot_all_mccs(mean_results_TSignal,std_results_TSignal,mean_results_SP6,mcc_deepsig,mcc_predTat,mcc_lipop,mcc_phobius):
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"
    fig, ax = plt.subplots(3, 2,figsize=(8, 6),dpi=350)
    line_w = 0.1
    offsets = [-line_w*2.5,-line_w*1.5,-line_w*0.5, line_w*0.5,line_w*1.5,line_w*2.5]
    colors = ["mediumblue", "saddlebrown", "green", "black", "purple","red"]
    ylabels = ["Sec/SPase I", "Sec/SPase II", "Tat/SPase I"]
    model_names = ["TSignal", "SignalP 6.0", "DeepSig", "Pred-TAT","LipoP", "Phobius"]
    for ind in range(3):
        current_sptype_mcc1, current_sptype_mcc2 = mean_results_TSignal[ind]
        current_sptype_mcc1_std, current_sptype_mcc2_std = std_results_TSignal[ind]
        current_sptype_mcc1_sp6, current_sptype_mcc2_sp6 = mean_results_SP6[ind]
        current_sptype_mcc1_deepsig, current_sptype_mcc1_predTat, \
        current_sptype_mcc1_LipoP, current_sptype_mcc1_Phobius = \
            mcc_deepsig[ind],mcc_predTat[ind],mcc_lipop[ind],mcc_phobius[ind]
        mcc1_mean_results = [current_sptype_mcc1,current_sptype_mcc1_sp6,current_sptype_mcc1_deepsig,current_sptype_mcc1_predTat,current_sptype_mcc1_LipoP,current_sptype_mcc1_Phobius]
        mcc2_mean_results = [current_sptype_mcc2,current_sptype_mcc2_sp6]
        # print(current_sptype_mcc1,current_sptype_mcc1_sp6,current_sptype_mcc1_deepsig,current_sptype_mcc1_predTat,current_sptype_mcc1_LipoP,current_sptype_mcc1_Phobius)
        box = ax[ind][0].get_position()
        ax[ind][0].set_position([box.x0, box.y0 + box.height * 0.35, box.width * 1.32, box.height * 0.95])
        box = ax[ind][1].get_position()
        ax[ind][1].set_position([box.x0+box.width * 0.42, box.y0 + box.height * 0.35, box.width * 0.8, box.height * 0.95])
        ax[ind][0].set_ylabel("MCC1\n"+ylabels[ind],fontsize=12.5)
        ax[ind][1].set_ylabel("MCC2\n"+ylabels[ind],fontsize=12.5)
        if ind != 2:
            ax[ind][0].set_xticks([1,2,3,4])
            ax[ind][0].set_xticklabels(["","","",""])
            ax[ind][1].set_xticks([1,2,3])
            ax[ind][1].set_xticklabels(["","",""])
        else:
            ax[ind][0].set_xticks([1,2,3,4])
            ax[ind][0].set_xticklabels(["eukarya", "gn-bacteria", "gp-bacteria", "archaea"], fontsize=12.5)
            ax[ind][1].set_xticks([1, 2, 3])
            ax[ind][1].set_xticklabels(["gn-bacteria", "gp-bacteria", "archaea"], fontsize=12.5)

        ax[ind][0].plot([1.5,1.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        ax[ind][0].plot([2.5,2.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        ax[ind][0].plot([3.5,3.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')

        ax[ind][1].plot([1.5,1.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        ax[ind][1].plot([2.5,2.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        for j in range(6):

            ax[ind][0].bar([i + offsets[j] for i in range(1 if ind == 0 else 2,5)], mcc1_mean_results[j], width=line_w,alpha=0.6,color=colors[j],label=model_names[j]) #, current_sptype_mcc1_sp6[j - 1],
            if j ==0:
                current_sptype_mcc1_std = [max(0.001,current_sptype_mcc1_std_) for current_sptype_mcc1_std_ in current_sptype_mcc1_std]
                low, high = np.array(mcc1_mean_results[j]) - 2 * np.array(current_sptype_mcc1_std), \
                            np.array(mcc1_mean_results[j]) + 2 *  np.array(current_sptype_mcc1_std)
                for index, ind_ in enumerate(range(1 if ind == 0 else 2, 5)):
                    ax[ind][0].plot([ind_+offsets[j], ind_+offsets[j]], [low[index],high[index]],color='black')
            if ind == 0:
                handles, labels = ax[ind][0].get_legend_handles_labels()
        for j in range(2):
            ax[ind][1].bar([i + offsets[j+2] for i in range(1,4)], mcc2_mean_results[j], width=line_w,alpha=0.6,color=colors[j]) #, current_sptype_mcc1_sp6[j - 1],

            if j ==0:
                current_sptype_mcc2_std = [max(0.0001,current_sptype_mcc2_std_) for current_sptype_mcc2_std_ in current_sptype_mcc2_std]
                low, high = np.array(mcc2_mean_results[j]) - 2 * np.array(current_sptype_mcc2_std), \
                            np.array(mcc2_mean_results[j]) + 2 *  np.array(current_sptype_mcc2_std)
                for index, ind_ in enumerate(range(1, 4)):
                    ax[ind][1].plot([ind_+offsets[2], ind_+offsets[2]], [low[index],high[index]],color='black')

        ax[ind][0].set_ylim([0,1.1])
        ax[ind][1].set_ylim([0,1.1])
        ax[ind][0].set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax[ind][0].set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=12.5)
        ax[ind][1].set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax[ind][1].set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=12.5)
        ax[ind][0].grid(axis='y',color='black', linestyle='-', linewidth=0.5,alpha=0.4)
        ax[ind][1].grid(axis='y',color='black', linestyle='-', linewidth=0.5,alpha=0.4)
        ax[ind][0].set_xlim([0.5,4.5])
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.01, 0.05), ncol=5, fontsize=12.5)
    plt.savefig("some_plot_mcc.pdf")


def plot_sp6_vs_tnmt_mcc(result_folders=("only_decoder_tune_bert_extraOhOnOut_swa_run_1",
                                     "only_decoder_tune_bert_extraOhOnOut_swa_run_2",
                                     "only_decoder_tune_bert_extraOhOnOut_swa_run_3",
                                     "only_decoder_tune_bert_extraOhOnOut_swa_run_4",
                                     "only_decoder_tune_bert_extraOhOnOut_swa_run_5"),
                         benchmark=True):
    """
    Method for MCC bar plots with mean and variance Sec/SPase I and II and Tat/SPase I compared to SignalP 6.0 and other
    models. The method computes mean and average across multiple runs, but can be used for a single run also.

    :param list_or_tuple result_folders: contains the folders containing the sequence prediction binaries. Multiple
    runs are averaged and result variance is also computed, but the method may be used for a single run also (e.g.
    have result_folders = ["one_single_folder_with_results"]
    """
    sp1_f1s, sp1_recs, sp1_precs, sp2_f1s, sp2_recs, sp2_precs, tat_f1s, \
    tat_recs, tat_precs, mcc1_sp1, mcc2_sp1, mcc1_sp2, mcc2_sp2, mcc1_tat, mcc2_tat = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    # result files for each of the other methods
    mcc_deepsig = extract_compatible_binaries_deepsig(restrict_types=["SP", "NO_SP"], return_mcc=True)
    mcc_predTat = extract_compatible_binaries_predtat(restrict_types=["SP", "NO_SP", "TAT"], return_mcc=True)
    mcc_lipop = extract_compatible_binaries_lipop(restrict_types=["SP", "NO_SP", "LIPO"], return_mcc=True)
    mcc_phobius = extract_compatible_phobius_binaries(restrict_types=["SP", "NO_SP"], return_mcc=True)
    mcc_deepsig[-1] = 0

    # "padd" with zeroes the SP types which are not predictable by their respective models
    mcc_deepsig = [mcc_deepsig, [0]*3, [0]*3]
    mcc_predTat = [mcc_predTat[0], [0]*3, mcc_predTat[1]]
    mcc_lipop = [mcc_lipop[0], mcc_lipop[1], [0]*3]
    mcc_phobius = [mcc_phobius, [0]*3, [0]*3]

    for run_folder in result_folders:
        print("Computing results for number {}".format(run_folder))
        mdl2results = extract_all_param_results(only_cs_position=False,
                                                result_folder=run_folder,
                                                compare_mdl_plots=False,
                                                remove_test_seqs=False,
                                                benchmark=benchmark,
                                                prints=False)
        mdl_ind = 0
        sp1_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-7])]))
        sp1_recs.append(np.array([rec for rec in mdl2results[mdl_ind][10]]))
        sp1_precs.append(np.array([rec for rec in mdl2results[mdl_ind][11]]))
        sp2_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-6])]))
        sp2_recs.append(np.array([rec for rec in mdl2results[mdl_ind][12]]))
        sp2_precs.append(np.array([rec for rec in mdl2results[mdl_ind][13]]))
        tat_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-5])]))
        tat_recs.append(np.array([rec for rec in mdl2results[mdl_ind][14]]))
        tat_precs.append(np.array([rec for rec in mdl2results[mdl_ind][15]]))
        mcc1_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][0]]))
        mcc2_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][1][1:]]))
        mcc1_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][2]]))
        mcc2_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][3]]))
        mcc1_tat.append(np.array([mcc for mcc in mdl2results[mdl_ind][4]]))
        mcc2_tat.append([mcc for mcc in mdl2results[mdl_ind][5]])

    mean_mcc1_sp1, mean_mcc2_sp1 = np.mean(np.stack(mcc1_sp1),axis=0), np.mean(np.stack(mcc2_sp1), axis=0)
    mean_mcc1_sp2, mean_mcc2_sp2 = np.mean(np.stack(mcc1_sp2),axis=0), np.mean(np.stack(mcc2_sp2), axis=0)
    mean_mcc1_tat, mean_mcc2_tat = np.mean(np.stack(mcc1_tat),axis=0), np.mean(np.stack(mcc2_tat), axis=0)

    std_mcc1_sp1, std_mcc2_sp1 = np.std(np.stack(mcc1_sp1), axis=0), np.std(np.stack(mcc2_sp1), axis=0)
    std_mcc1_sp2, std_mcc2_sp2 = np.std(np.stack(mcc1_sp2), axis=0), np.std(np.stack(mcc2_sp2), axis=0)
    std_mcc1_tat, std_mcc2_tat = np.std(np.stack(mcc1_tat), axis=0), np.std(np.stack(mcc2_tat), axis=0)

    sp6_mcc1_sp1, sp6_mcc2_sp1 = np.array([0.868, 0.811,0.878,0.737]), np.array([0.649, 0.734, 0.728])
    sp6_mcc1_sp2, sp6_mcc2_sp2 = np.array([0.838, 0.894, 0.871]), np.array([0.841, 0.893, 0.719])
    sp6_mcc1_tat, sp6_mcc2_tat = np.array([0.946, 0.788, 0.802]), np.array([0.934, 0.806,0.807])

    mean_results_TSignal = [[mean_mcc1_sp1, mean_mcc2_sp1], [mean_mcc1_sp2, mean_mcc2_sp2],[mean_mcc1_tat, mean_mcc2_tat]]
    std_results_TSignal = [[std_mcc1_sp1, std_mcc2_sp1], [std_mcc1_sp2, std_mcc2_sp2], [std_mcc1_tat, std_mcc2_tat]]
    mean_results_SP6 = [[sp6_mcc1_sp1, sp6_mcc2_sp1], [sp6_mcc1_sp2,sp6_mcc2_sp2],[sp6_mcc1_tat, sp6_mcc2_tat]]



    if benchmark:
        no_of_seqs_sp1 = np.array([146, 61, 15, 36])
        no_of_seqs_sp2 = np.array([257, 120, 9])
        no_of_seqs_tat = np.array([51, 18, 9])
        no_of_tested_sp_seqs = sum([146, 61, 15, 36]) + sum([257, 120, 9]) + sum([51, 18, 9])
    else:
        no_of_seqs_sp1 = np.array([2040, 44, 142, 356])
        no_of_seqs_sp2 = np.array([1087, 516, 12])
        no_of_seqs_tat = np.array([313, 39, 13])
        no_of_tested_sp_seqs = sum([2040, 44, 142, 356]) + sum([1087, 516, 12]) + sum([313, 39, 13])

    # no_of_seqs_sp1 = np.arraZ

    print("mcc1- (sp1,sp2,tat). Tsi", [round(mc, 3)  for mc in mean_mcc1_sp1], [round(mc, 3) for mc in mean_mcc1_sp2], [round(mc,3) for mc in mean_mcc1_tat])
    print("mcc1- (sp1,sp2,tat). SP6", list(sp6_mcc1_sp1), list(sp6_mcc1_sp2), list(sp6_mcc1_tat))
    print("mcc2- (sp1,sp2,tat). TSi", [round(mc,3) for mc in mean_mcc2_sp1], [round(mc,3) for mc in mean_mcc2_sp2], [round(mc,3) for mc in mean_mcc2_tat])
    print("mcc2- (sp1,sp2,tat). SP6", list(sp6_mcc2_sp1), list(sp6_mcc2_sp2), list(sp6_mcc2_tat))
    total_mcc1 = (sum(mean_mcc1_sp1 * no_of_seqs_sp1) + sum(mean_mcc1_sp2 * no_of_seqs_sp2) + sum(mean_mcc1_tat * no_of_seqs_tat))/no_of_tested_sp_seqs
    total_mcc1_sp6 = (sum(sp6_mcc1_sp1 * no_of_seqs_sp1) + sum(sp6_mcc1_sp2 * no_of_seqs_sp2) + sum(sp6_mcc1_tat * no_of_seqs_tat))/no_of_tested_sp_seqs
    total_mcc2 = (sum(mean_mcc2_sp1 * no_of_seqs_sp1[1:]) + sum(mean_mcc2_sp2 * no_of_seqs_sp2) + sum(mean_mcc2_tat * no_of_seqs_tat))/(no_of_tested_sp_seqs-146 if benchmark else no_of_tested_sp_seqs-2040)
    total_mcc2_sp6 =  (sum(sp6_mcc2_sp1 * no_of_seqs_sp1[1:]) + sum(sp6_mcc2_sp2 * no_of_seqs_sp2) + sum(sp6_mcc2_tat * no_of_seqs_tat))/(no_of_tested_sp_seqs-146 if benchmark else no_of_tested_sp_seqs-2040)
    total_mcc1_std = (sum(std_mcc1_sp1 * no_of_seqs_sp1) + sum(std_mcc1_sp2 * no_of_seqs_sp2) + sum(
        std_mcc1_tat * no_of_seqs_tat)) / (no_of_tested_sp_seqs-146 if benchmark else no_of_tested_sp_seqs-2040)
    total_mcc2_std = (sum(std_mcc2_sp1 * no_of_seqs_sp1[1:]) + sum(std_mcc2_sp2 * no_of_seqs_sp2) + sum(
        std_mcc2_tat * no_of_seqs_tat)) / (no_of_tested_sp_seqs-146 if benchmark else no_of_tested_sp_seqs-2040)

    print("non-w average sp6:",(np.mean(sp6_mcc1_sp1)+np.mean(sp6_mcc1_sp2) + np.mean(sp6_mcc1_tat) + np.mean(sp6_mcc2_sp1)+np.mean(sp6_mcc2_sp2) + np.mean(sp6_mcc2_tat))/6)
    print("non-w average tsignal:",(np.mean(mean_mcc1_sp1)+np.mean(mean_mcc2_sp1) + np.mean(mean_mcc1_sp2) + np.mean(mean_mcc2_sp2)+np.mean(mean_mcc1_tat) + np.mean(mean_mcc2_tat))/6)
    print("mcc1 w average sp6/tsignal:",total_mcc1_sp6,total_mcc1,"+/-", total_mcc1_std)
    print("mcc2 w average sp6/tsignal:",total_mcc2_sp6,total_mcc2,"+/-", total_mcc2_std)
    bar_plot_all_mccs(mean_results_TSignal,std_results_TSignal,mean_results_SP6,mcc_deepsig,mcc_predTat,mcc_lipop,mcc_phobius)
    exit(1)

def plot_comparative_performance_sp1_mdls(result_folders=("previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_1",
                                                         "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_2",
                                                         "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_3",
                                                         "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_4",
                                                         "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_5")):

    all_recalls, all_precisions, f1_deepsig = extract_compatible_binaries_deepsig(restrict_types=["SP", "NO_SP"])
    all_recalls, all_precisions, f1_predtat = extract_compatible_binaries_predtat(restrict_types=["SP", "NO_SP"])
    all_recalls, all_precisions, f1_lipop = extract_compatible_binaries_lipop(restrict_types=["SP", "NO_SP"])
    all_recalls, all_precisions, f1_phobius = extract_compatible_phobius_binaries(restrict_types=["SP", "NO_SP"])


    # all_f1s = [tnmt_f1, f1_predtat, f1_lipop, f1_deepsig, f1_phobius]
    all_f1s = [f1_deepsig, f1_phobius, f1_lipop, f1_predtat]

    # names = ["TSignal", "PredTAT", "LipoP", "DeepSig", "Phobius"]
    names = ["TSignal", "DeepSig", "Phobius", "LipoP", "PredTAT"]
    colors = ["navy", "red", "green", "black", "purple"]
    titles = ["eukarya", "gn bacteria", "gp bacteria", "archaea"]

    sp1_f1s, sp1_recs, sp1_precs, sp2_f1s, sp2_recs, sp2_precs, tat_f1s, \
    tat_recs, tat_precs, mcc1_sp1, mcc2_sp1, mcc1_sp2, mcc2_sp2, mcc1_tat, mcc2_tat = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for run in result_folders:
        print("Computing results for run {}".format(run))
        run_results_folder = run

        mdl2results = extract_all_param_results(only_cs_position=False,
                                                result_folder=run_results_folder,
                                                compare_mdl_plots=False,
                                                remove_test_seqs=False,
                                                benchmark=True,
                                                prints=False,
                                                restrict_types=["SP", "NO_SP"])
        mdl_ind = 0
        sp1_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-7])]))
        sp1_recs.append(np.array([rec for rec in mdl2results[mdl_ind][10]]))
        sp1_precs.append(np.array([rec for rec in mdl2results[mdl_ind][11]]))
        sp2_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-6])]))
        sp2_recs.append(np.array([rec for rec in mdl2results[mdl_ind][12]]))
        sp2_precs.append(np.array([rec for rec in mdl2results[mdl_ind][13]]))
        tat_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-5])]))
        tat_f1s.append(np.array([rec for rec in mdl2results[mdl_ind][14]]))
        tat_precs.append(np.array([rec for rec in mdl2results[mdl_ind][15]]))
        mcc1_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][0]]))
        mcc2_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][1][1:]]))
        mcc1_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][2]]))
        mcc2_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][3]]))
        mcc1_tat.append(np.array([mcc for mcc in mdl2results[mdl_ind][4]]))
    arrange_tol_lg_sp1 = []
    for og_ind in [0, 4, 8, 12]:
        for tol in range(4):
            arrange_tol_lg_ = [sp1_f1s[run_no][og_ind + tol] for run_no in range(len(result_folders))]
            arrange_tol_lg_sp1.append(arrange_tol_lg_)
    arrange_tol_lg_sp2 = []
    for og_ind in [0, 4, 8]:
        for tol in range(4):
            arrange_tol_lg_ = [sp2_f1s[run_no][og_ind + tol] for run_no in range(len(result_folders))]
            arrange_tol_lg_sp2.append(arrange_tol_lg_)
    arrange_tol_lg_tat = []
    for og_ind in [0, 4, 8]:
        for tol in range(4):
            arrange_tol_lg_ = [tat_f1s[run_no][og_ind + tol] for run_no in range(len(result_folders))]
            arrange_tol_lg_tat.append(arrange_tol_lg_)
    arrange_sptype_tol_lg = [np.array(arrange_tol_lg_sp1), np.array(arrange_tol_lg_sp2), np.array(arrange_tol_lg_tat)]

    all_sptypes_all_mean = [np.mean(arrange_sptype_tol_lg[0],axis=1), np.mean(arrange_sptype_tol_lg[1], axis=1), np.mean(arrange_sptype_tol_lg[2], axis=1)]
    all_sptypes_all_std = [np.std(arrange_sptype_tol_lg[0],axis=1), np.std(arrange_sptype_tol_lg[1], axis=1), np.std(arrange_sptype_tol_lg[2], axis=1)]

    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"
    from matplotlib.pyplot import figure


    line_w = 0.15
    offsets = [-2 * line_w, - 1 * line_w,0, 1 * line_w, line_w * 2]
    fig, ax = plt.subplots(4, 1,figsize=(8, 6),dpi=350)

    for ind in range(4):
        for j in range(5):
            if j == 0:
                ax[ind].bar([i+offsets[j] for i in range(1,5)],all_sptypes_all_mean[0][ind*4:(ind+1)*4], color=colors[j], label=names[j], width=line_w, alpha=0.6)
                for std_ind in range(1,5):
                    low, high = all_sptypes_all_mean[0][std_ind +ind * 4 -1] - 2 * all_sptypes_all_std[0][std_ind + ind * 4 -1], \
                                all_sptypes_all_mean[0][std_ind + ind * 4 -1] + 2 * all_sptypes_all_std[0][std_ind + ind * 4 -1]
                    ax[ind].plot([std_ind + offsets[0], std_ind + offsets[0]], [low, max(high, low + 0.001)], color='black')


            else:
                ax[ind].bar([i + offsets[j] for i in range(1,5)], all_f1s[j-1][ind], color=colors[j], label=names[j], width=line_w, alpha=0.6)
        box = ax[ind].get_position()
        ax[ind].set_position([box.x0 + box.width * 0.05, box.y0 + box.height * 0.35, box.width * 1, box.height*0.85])
        if ind == 3:
            handles, labels = ax[ind].get_legend_handles_labels()

        ax[ind].set_xticks(list(range(1,5)))
        ax[ind].set_xticklabels(['{}'.format(i) for i in range(4)], fontsize=12.5)
        ax[ind].set_yticks([0, 0.2,0.4,0.6,0.8, 1])
        ax[ind].grid(axis='y',color='black', linestyle='-', linewidth=0.5,alpha=0.4)
        ax[ind].set_yticklabels(labels=[0, 0.2,0.4,0.6,0.8, 1],fontsize=12.5)

        ax[ind].set_ylabel("F1 score\n"+titles[ind], fontsize=12.5)
        ax[ind].yaxis.set_label_coords(-0.08, 0.5)
        ax[ind].set_xlim(0.6, 4.4)
        if ind == 3:
            ax[ind].set_xlabel("tolerance", fontsize=12.5)

    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.0, 0.045), ncol=5, fontsize=12.5)
    plt.savefig("some_plot_cs_bench_other_mdls.pdf")
    exit(1)

def create_random_split_fold_data():
    all_data = []
    for p in [0, 1, 2]:
        for t in ["train", "test"]:
            data = pickle.load(open("../sp_data/sp6_partitioned_data_{}_{}.bin".format(t, p), "rb"))
            extract_data = [[k, v[1], v[2], v[3]] for k,v in data.items()]
            print("sp_data/sp6_partitioned_data_{}_{}.bin".format(t, p), len(extract_data))
            all_data.extend(extract_data)
    indices = random.sample(list(range(len(all_data))), len(all_data))
    indices_per_fold = [[], [], []]
    no_of_dp_per_fold = len(indices) // 3
    for i in range(3):
        if i == 2:
            train_indices = indices[2 * no_of_dp_per_fold:]
        else:
            train_indices = indices[i * no_of_dp_per_fold:(i+1) * no_of_dp_per_fold]
        test_indices = set(random.sample(train_indices, len(train_indices) // 10))
        train_indices = list(set(train_indices) - test_indices)
        test_indices = list(test_indices)
        train_dict = {all_data[ind][0]:[[1], all_data[ind][1], all_data[ind][2], all_data[ind][3]] for ind in train_indices}
        test_dict = {all_data[ind][0]:[[1], all_data[ind][1], all_data[ind][2], all_data[ind][3]] for ind in test_indices}
        pickle.dump(train_dict, open("../sp_data/random_folds_sp6_partitioned_data_train_{}.bin".format(i), "wb"))
        pickle.dump(test_dict, open("../sp_data/random_folds_sp6_partitioned_data_test_{}.bin".format(i), "wb"))

def compute_diversity(data):
    return

def compute_maximum_feature_wise_diversity():
    life_grp2sp_types2embeddings = {'EUKARYA': {"SP": [], "NO_SP": []},
                                    'NEGATIVE': {"SP": [], "NO_SP": [], "LIPO": [], "TAT": []},
                                    'POSITIVE': {"SP": [], "NO_SP": [], "LIPO": [], "TAT": []},
                                    'ARCHAEA': {"SP": [], "NO_SP": [], "LIPO": [], "TAT": []}}
    lifegrp2_sptype2_max_feat_wise_std = {'EUKARYA': {"SP": -1, "NO_SP": -1},
                                    'NEGATIVE': {"SP": -1, "NO_SP": -1, "LIPO": -1, "TAT": -1},
                                    'POSITIVE': {"SP": -1, "NO_SP": -1, "LIPO": -1, "TAT": -1},
                                    'ARCHAEA': {"SP": -1, "NO_SP": -1, "LIPO": -1, "TAT": -1}}
    desired_sp_types = ["SP", "TAT", "LIPO"]
    data_folder = "/scratch/work/dumitra1/sp_data/" if os.path.exists('/scratch/work') else "../sp_data/"

    for fold in range(3):
        fold_data = pickle.load(open(data_folder + "sp6_partitioned_data_sublbls_test_{}.bin".format(fold), "rb"))
        fold_data.update(pickle.load(open(data_folder + "sp6_partitioned_data_sublbls_train_{}.bin".format(fold), "rb")))
        for k,v in fold_data.items():
            if v[-1] in desired_sp_types:
                life_grp2sp_types2embeddings[v[-2]][v[-1]].append(v[0].reshape(-1))
    for lg in ['EUKARYA', 'NEGATIVE', 'POSITIVE', 'ARCHAEA']:
        for sp_type in list(life_grp2sp_types2embeddings[lg].keys()):
            max_std = max(np.std(np.stack(life_grp2sp_types2embeddings[lg][sp_type]), axis=0))
            lifegrp2_sptype2_max_feat_wise_std[lg][sp_type] = max_std * 70
            print(max_std)
    return life_grp2sp_types2embeddings

def compute_diversity_within_partition(std=None):
    if std is None:
        std = {'EUKARYA': {'SP': 47.25394785404205, 'NO_SP': 39.85664486885071},
         'NEGATIVE': {'SP': 39.905853271484375, 'NO_SP': 41.696882247924805, 'LIPO': 38.223400712013245,
                      'TAT': 50.940147042274475},
         'POSITIVE': {'SP': 46.76344096660614, 'NO_SP': 42.55532145500183, 'LIPO': 41.33163273334503,
                      'TAT': 45.166996717453},
         'ARCHAEA': {'SP': 43.09800326824188, 'NO_SP': 39.40081000328064, 'LIPO': 45.83507776260376,
                     'TAT': 46.772186160087585}}
        # uncomment to recompute max(feature-wise stds) * 70 (media seq length)
        std = compute_maximum_feature_wise_diversity()
    life_grp2sp_types2embeddings = {'EUKARYA':{"SP":[] ,"NO_SP":[]},
                                    'NEGATIVE':{"SP":[], "NO_SP":[], "LIPO":[], "TAT":[]},
                                    'POSITIVE':{"SP":[], "NO_SP":[], "LIPO":[], "TAT":[]},
                                    'ARCHAEA':{"SP":[], "NO_SP":[], "LIPO":[], "TAT":[]}}
    fold2life_grp2sp_types2embeddings = {0:life_grp2sp_types2embeddings.copy(), 1:life_grp2sp_types2embeddings.copy(),
                                         2:life_grp2sp_types2embeddings.copy()}
    desired_sp_types = ["SP", "NO_SP", "TAT", "LIPO"]
    data_folder = "/scratch/work/dumitra1/sp_data/" if os.path.exists('/scratch/work') else "../sp_data/"
    for fold in range(3):
        fold_data = pickle.load(open(data_folder + "sp6_partitioned_data_sublbls_test_{}.bin".format(fold), "rb"))
        fold_data.update(pickle.load(open(data_folder + "sp6_partitioned_data_sublbls_train_{}.bin".format(fold), "rb")))
        for k,v in fold_data.items():
            if v[-1] in desired_sp_types:
                # if "S" in v[-3]:
                #     ind_l, ind_r = v[-3].rfind("S")-3, v[-3].rfind("S")
                # else:
                #     ind_l, ind_r = 0, 3
                fold2life_grp2sp_types2embeddings[fold][v[-2]][v[-1]].append(v[0].reshape(-1))
                # fold2life_grp2sp_types2embeddings[fold][v[-2]][v[-1]].append(v[0][ind_l:ind_r].reshape(-1))
    for i in range(3):
        for j in range(3):
            for lg in ["EUKARYA", "NEGATIVE", "POSITIVE", "ARCHAEA"]:
                for sp_type in list(life_grp2sp_types2embeddings[lg].keys()):
                    if sp_type != "NO_SP":
                        dists = euclidian_distance(np.stack(fold2life_grp2sp_types2embeddings[i][lg][sp_type]), np.stack(fold2life_grp2sp_types2embeddings[j][lg][sp_type]))
                        if i != j:
                            dists = dists.reshape(-1)
                        else:
                            all_non_identical_dists = []
                            for k in range(dists.shape[0]):
                                all_non_identical_dists.extend(dists[k, k+1:])
                            dists = np.array(all_non_identical_dists)
                        std_ = std[lg][sp_type]
                        div = np.mean( list(np.exp(- (d_**2)/(2*std_**2) ) for d_ in dists)) ** (-1)
                        print("{}-{} on folds {}/{} has diversity {}:".format(lg, sp_type, i, j, div))

def visualize_data_amount2_results(benchmark_ds=False):
    subfolds = ["02", "04", "06", "08", "1"]
    id2seq, id2lg, id2type, id2truelbls = extract_id2seq_dict()
    lg2f1_tol0_sp1 = {'EUKARYA':[], 'NEGATIVE':[], 'POSITIVE':[], 'ARCHAEA':[]}
    lg2f1_tol3_sp1 = {'EUKARYA':[], 'NEGATIVE':[], 'POSITIVE':[], 'ARCHAEA':[]}
    lg2f1_tol0_sp2 = {'NEGATIVE':[], 'POSITIVE':[], 'ARCHAEA':[]}
    lg2f1_tol3_sp2 = {'NEGATIVE':[], 'POSITIVE':[], 'ARCHAEA':[]}
    lg2f1_tol0_tat = {'NEGATIVE':[], 'POSITIVE':[], 'ARCHAEA':[]}
    lg2f1_tol3_tat = {'NEGATIVE':[], 'POSITIVE':[], 'ARCHAEA':[]}

    for sf in subfolds:
        result_dict = {}
        sptype_dict = {}
        for folds in [[0, 1], [0,2], [1,2]]:
            result_dict.update(pickle.load(open("train_subset_results/best_model_subtrain_{}_random_folds_{}_{}_best.bin".format(sf,folds[0],folds[1]), "rb")))
            sptype_dict.update(pickle.load(open("train_subset_results/best_model_subtrain_{}_random_folds_{}_{}_best_sptype.bin".format(sf,folds[0],folds[1]), "rb")))
        # print(list(sptype_dict.keys())[0])
        # print(list(result_dict.keys())[0])
        if benchmark_ds:
            unique_bench_seqs = set(id2seq.values())
            result_dict = {k:v for k,v in result_dict.items() if k in unique_bench_seqs}
        life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(filename="w_lg_w_glbl_lbl_100ep.bin",
                                                                                       dict_=result_dict)
        mccs, mccs2 = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                               sp_type="SP")
        # LIPO is SEC/SPII
        mccs_lipo, mccs2_lipo = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                                         sp_type="LIPO")
        # TAT is TAT/SPI
        mccs_tat, mccs2_tat = get_pred_perf_sptype(life_grp, seqs, true_lbls, pred_lbls, v=False, return_mcc2=True,
                                                       sp_type="TAT")
        # print(mccs)
        all_recalls, all_precisions, _, _, _, f1_scores_sp1 = \
            get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="SP",
                       sptype_preds=sptype_dict)
        all_recalls, all_precisions, _, _, _, f1_scores_sp2 = \
            get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="LIPO",
                       sptype_preds=sptype_dict)
        all_recalls, all_precisions, _, _, _, f1_scores_tat = \
            get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="TAT",
                       sptype_preds=sptype_dict)
        lg2f1_tol0_sp1['EUKARYA'].append(f1_scores_sp1[0][0])
        lg2f1_tol3_sp1['EUKARYA'].append(f1_scores_sp1[0][3])
        lg2f1_tol0_sp1['NEGATIVE'].append(f1_scores_sp1[1][0])
        lg2f1_tol3_sp1['NEGATIVE'].append(f1_scores_sp1[1][3])
        lg2f1_tol0_sp1['POSITIVE'].append(f1_scores_sp1[2][0])
        lg2f1_tol3_sp1['POSITIVE'].append(f1_scores_sp1[2][3])
        lg2f1_tol0_sp1['ARCHAEA'].append(f1_scores_sp1[3][0])
        lg2f1_tol3_sp1['ARCHAEA'].append(f1_scores_sp1[3][3])

        lg2f1_tol0_sp2['NEGATIVE'].append(f1_scores_sp2[0][0])
        lg2f1_tol3_sp2['NEGATIVE'].append(f1_scores_sp2[0][3])
        lg2f1_tol0_sp2['POSITIVE'].append(f1_scores_sp2[1][0])
        lg2f1_tol3_sp2['POSITIVE'].append(f1_scores_sp2[1][3])
        lg2f1_tol0_sp2['ARCHAEA'].append(f1_scores_sp2[2][0])
        lg2f1_tol3_sp2['ARCHAEA'].append(f1_scores_sp2[2][3])

        lg2f1_tol0_tat['NEGATIVE'].append(f1_scores_tat[0][0])
        lg2f1_tol3_tat['NEGATIVE'].append(f1_scores_tat[0][3])
        lg2f1_tol0_tat['POSITIVE'].append(f1_scores_tat[1][0])
        lg2f1_tol3_tat['POSITIVE'].append(f1_scores_tat[1][3])
        lg2f1_tol0_tat['ARCHAEA'].append(f1_scores_tat[2][0])
        lg2f1_tol3_tat['ARCHAEA'].append(f1_scores_tat[2][3])

    plt.title("SEC/SPI results", fontsize=26)
    plt.xlabel("Percentage of dataset used for training")
    plt.ylabel("F1 score")
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol0_sp1['EUKARYA'], linestyle='solid', color='blue', label='Eukaryotes tol 0')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol3_sp1['EUKARYA'], linestyle='dashed', color='blue', label='Eukaryotes tol 3')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol0_sp1['NEGATIVE'], linestyle='solid', color='red', label='Negative tol 0')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol3_sp1['NEGATIVE'], linestyle='dashed', color='red', label='Negative tol 3')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol0_sp1['POSITIVE'], linestyle='solid', color='orange', label='Positive tol 0')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol3_sp1['POSITIVE'], linestyle='dashed', color='orange', label='Positive tol 3')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol0_sp1['ARCHAEA'], linestyle='solid', color='green', label='Archaea tol 0')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol3_sp1['ARCHAEA'], linestyle='dashed', color='green', label='Archaea tol 3')
    plt.legend(fontsize=26)
    plt.show()

    plt.title("SEC/SPII results", fontsize=26)
    plt.xlabel("Percentage of dataset used for training")
    plt.ylabel("F1 score")
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol0_sp2['NEGATIVE'], linestyle='solid', color='red', label='Negative tol 0')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol3_sp2['NEGATIVE'], linestyle='dashed', color='red', label='Negative tol 3')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol0_sp2['POSITIVE'], linestyle='solid', color='orange', label='Positive tol 0')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol3_sp2['POSITIVE'], linestyle='dashed', color='orange', label='Positive tol 3')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol0_sp2['ARCHAEA'], linestyle='solid', color='green', label='Archaea tol 0')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol3_sp2['ARCHAEA'], linestyle='dashed', color='green', label='Archaea tol 3')
    plt.legend(fontsize=26)
    plt.show()

    plt.title("TAT results",fontsize=26)
    plt.xlabel("Percentage of dataset used for training")
    plt.ylabel("F1 score")
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol0_tat['NEGATIVE'], linestyle='solid', color='red', label='Negative tol 0')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol3_tat['NEGATIVE'], linestyle='dashed', color='red', label='Negative tol 3')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol0_tat['POSITIVE'], linestyle='solid', color='orange', label='Positive tol 0')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol3_tat['POSITIVE'], linestyle='dashed', color='orange', label='Positive tol 3')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol0_tat['ARCHAEA'], linestyle='solid', color='green', label='Archaea tol 0')
    plt.plot([0.2,0.4,0.6, 0.8, 1], lg2f1_tol3_tat['ARCHAEA'], linestyle='dashed', color='green', label='Archaea tol 3')
    plt.legend(fontsize=26)
    plt.show()

def rename_files():
    folder = "train_subset_results/first_run"
    for file in os.listdir(folder):
        # if "data_perc" not in file:
        perc = file.split("subset_train")[1].split("_")[1]
        if "." not in perc and perc != "1":
            perc_ind = file.find("subset_train_") + len("subset_train_")
            new_name = file[:perc_ind] + perc[0] + "." + perc[1] + file[perc_ind+2]
            os.rename(folder + "/" + file, folder + "/" + new_name)

def plot_perf_over_data_perc():
    """
        Method to compute and plot performance over various amounts of data for Sec/SPase I CS-F1 predictions.
    """
    subsets = [0.25, 0.5, 0.75, 1]
    subset_2_f1 = {s:[] for s in subsets}
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"
    fig, ax = plt.subplots(2, 2,figsize=(8, 6))
    for subset in subsets:
        for run in range(6):
            print("Computing run {} for subset {}".format(run, subset))
            aa_pred_dict = {}
            glbl_lbl_dict = {}
            for tr_f in [[0,1],[0,2],[1,2]]:
                aa_pred_file = "random_folds_run_dos_0.1_lr_1e-05_nlayers_3_nhead_16_run_no_{}_subset_train_{}_trFlds_{}_{}_best.bin".format(run,str(subset), *tr_f)
                glbl_lbl_file = "random_folds_run_dos_0.1_lr_1e-05_nlayers_3_nhead_16_run_no_{}_subset_train_{}_trFlds_{}_{}_best_sptype.bin".format(run,str(subset), *tr_f)
                aa_pred_dict.update(pickle.load(open("tuning_bert_random_folds_only_decoder/"+aa_pred_file, "rb")))
                glbl_lbl_dict.update(pickle.load(open("tuning_bert_random_folds_only_decoder/"+glbl_lbl_file, "rb")))

            life_grp, seqs, true_lbls, pred_lbls = extract_seq_group_for_predicted_aa_lbls(
                filename="w_lg_w_glbl_lbl_100ep.bin",
                dict_=aa_pred_dict)

            all_recalls, all_precisions, _, _, _, f1_scores = \
                get_cs_perf(life_grp, seqs, true_lbls, pred_lbls, v=False, only_cs_position=False, sp_type="SP",
                           sptype_preds=glbl_lbl_dict)
            subset_2_f1[subset].append(f1_scores)
    sp1_plot = True
    all_euk_mean_tol0, all_neg_mean_tol0, all_pos_mean_tol0, all_archaea_mean_tol0 = [], [], [] ,[]
    all_euk_std_tol0, all_neg_std_tol0, all_pos_std_tol0, all_archaea_std_tol0 = [], [], [], []
    all_euk_mean_tol3, all_neg_mean_tol3, all_pos_mean_tol3, all_archaea_mean_tol3 = [], [], [] ,[]
    all_euk_std_tol3, all_neg_std_tol3, all_pos_std_tol3, all_archaea_std_tol3 = [], [], [], []
    for subset, f1 in subset_2_f1.items():
        all_euk, all_neg, all_pos, all_archea = [], [],[],[]
        for run_f1 in f1:
            if sp1_plot:
                eukaryote, negative, positive, archaea = run_f1
                all_euk.append(eukaryote)
            else:
                negative, positive, archaea = run_f1
            all_neg.append(negative)
            all_pos.append(positive)
            all_archea.append(archaea)
        if sp1_plot:
            all_euk_mean_tol0.append(np.mean([all_euk_[0] for all_euk_ in all_euk]))
            all_euk_std_tol0.append(np.std([all_euk_[0] for all_euk_ in all_euk]))
            all_euk_mean_tol3.append(np.mean([all_euk_[3] for all_euk_ in all_euk]))
            all_euk_std_tol3.append(np.std([all_euk_[3] for all_euk_ in all_euk]))
        all_neg_mean_tol0.append(np.mean([all_neg_[0] for all_neg_ in all_neg]))
        all_neg_std_tol0.append(np.std([all_neg_[0] for all_neg_ in all_neg]))
        all_neg_mean_tol3.append(np.mean([all_neg_[3] for all_neg_ in all_neg]))
        all_neg_std_tol3.append(np.std([all_neg_[3] for all_neg_ in all_neg]))
        all_pos_mean_tol0.append(np.mean([all_pos_[0] for all_pos_ in all_pos]))
        all_pos_std_tol0.append(np.std([all_pos_[0] for all_pos_ in all_pos]))
        all_pos_mean_tol3.append(np.mean([all_pos_[3] for all_pos_ in all_pos]))
        all_pos_std_tol3.append(np.std([all_pos_[3] for all_pos_ in all_pos]))

        all_archaea_mean_tol0.append(np.mean([all_archea_[0] for all_archea_ in all_archea]))
        all_archaea_std_tol0.append(np.std([all_archea_[0] for all_archea_ in all_archea]))
        all_archaea_mean_tol3.append(np.mean([all_archea_[3] for all_archea_ in all_archea]))
        all_archaea_std_tol3.append(np.std([all_archea_[3] for all_archea_ in all_archea]))
    if sp1_plot:
        all_euk_mean_tol0 = np.array(all_euk_mean_tol0)
        all_euk_mean_tol3 = np.array(all_euk_mean_tol3)
        all_euk_std_tol0 = np.array(all_euk_std_tol0)
        all_euk_std_tol3 = np.array(all_euk_std_tol3)
        ax[0,0].plot(subsets, all_euk_mean_tol0, '-', label='eukarya tol 0', color='blue')
        ax[0,0].fill_between(subsets, all_euk_mean_tol0 - 2 * all_euk_std_tol0, all_euk_mean_tol0 + 2 * all_euk_std_tol0, alpha=0.2, color='blue')
        ax[0,0].plot(subsets, all_euk_mean_tol3, '--',label='eukarya tol 3', color='blue')
        ax[0,0].fill_between(subsets, all_euk_mean_tol3 - 2 * all_euk_std_tol3, all_euk_mean_tol3 + 2 * all_euk_std_tol3, alpha=0.2, color='blue')
        box = ax[0,0].get_position()
        ax[0,0].set_position([box.x0 + box.width * 0.05, box.y0 + box.height * 0.45, box.width, box.height * 0.75])
        ax[0,0].set_ylabel("F1 score", fontsize=12.5)
        ax[0, 0].yaxis.set_label_coords(-0.2, 0.45)
        ax[0, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax[0, 0].set_xticks([0.25, 0.5, 0.75, 1])
        ax[0, 0].set_xticklabels([0.25, 0.5, 0.75, 1], fontsize=12.5)
        ax[0, 0].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)

    all_neg_mean_tol0 = np.array(all_neg_mean_tol0)
    all_neg_mean_tol3 = np.array(all_neg_mean_tol3)
    all_neg_std_tol0 = np.array(all_neg_std_tol0)
    all_neg_std_tol3 = np.array(all_neg_std_tol3)

    ax[0,1].plot(subsets, all_neg_mean_tol0, '-', label='gn bacteria tol 0', color='orange')
    ax[0,1].fill_between(subsets, all_neg_mean_tol0 - 2 * all_neg_std_tol0, all_neg_mean_tol0 + 2 * all_neg_std_tol0,
                    alpha=0.2, color='orange')
    ax[0,1].plot(subsets, all_neg_mean_tol3, '--', label='gn bacteria tol 3', color='orange')
    ax[0,1].fill_between(subsets, all_neg_mean_tol3 - 2 * all_neg_std_tol3, all_neg_mean_tol3 + 2 * all_neg_std_tol3,
                    alpha=0.2, color='orange')
    box = ax[0,1].get_position()
    ax[0,1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax[0,1].set_position([box.x0 + box.width * 0.05, box.y0 + box.height * 0.45, box.width, box.height * 0.75])
    ax[0,1].set_xticks([0.25, 0.5, 0.75, 1])
    ax[0,1].set_xticklabels([0.25, 0.5, 0.75, 1], fontsize=12.5)
    ax[0,1].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)

    all_pos_mean_tol0 = np.array(all_pos_mean_tol0)
    all_pos_mean_tol3 = np.array(all_pos_mean_tol3)
    all_pos_std_tol0 = np.array(all_pos_std_tol0)
    all_pos_std_tol3 = np.array(all_pos_std_tol3)
    ax[1,0].plot(subsets, all_pos_mean_tol0, '-', label='gp bacteria tol 0', color='purple')
    ax[1,0].fill_between(subsets, all_pos_mean_tol0 - 2 * all_pos_std_tol0, all_pos_mean_tol0 + 2 * all_pos_std_tol0,
                    alpha=0.2, color='purple')
    ax[1,0].plot(subsets, all_pos_mean_tol3, '--', label='gp bacteria tol 3', color='purple')
    ax[1,0].fill_between(subsets, all_pos_mean_tol3 - 2 * all_pos_std_tol3, all_pos_mean_tol3 + 2 * all_pos_std_tol3,
                    alpha=0.2, color='purple')
    ax[1, 0].set_ylabel("F1 score", fontsize=12.5)
    ax[1, 0].yaxis.set_label_coords(-0.2, 0.45)
    ax[1, 0].set_xticks([0.25, 0.5, 0.75, 1])

    box = ax[1,0].get_position()
    ax[1,0].set_position([box.x0 + box.width * 0.05, box.y0 + box.height * 0.6, box.width, box.height * 0.75])
    ax[1,0].set_xlabel("fraction of data", fontsize=12.5)
    ax[1,0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax[1,0].set_xticklabels([0.25, 0.5, 0.75, 1], fontsize=12.5)
    ax[1,0].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
    all_archaea_mean_tol0 = np.array(all_archaea_mean_tol0)
    all_archaea_mean_tol3 = np.array(all_archaea_mean_tol3)
    all_archaea_std_tol0 = np.array(all_archaea_std_tol0)
    all_archaea_std_tol3 = np.array(all_archaea_std_tol3)

    ax[1,1].plot(subsets, all_archaea_mean_tol0, '-', label='archaea tol 0', color='red')
    ax[1,1].fill_between(subsets, all_archaea_mean_tol0 - 2 * all_archaea_std_tol0, all_archaea_mean_tol0 + 2 * all_archaea_std_tol0,
                    alpha=0.2, color='red')

    ax[1,1].plot(subsets, all_archaea_mean_tol3, '--', label='archaea tol 3', color='red')
    ax[1,1].fill_between(subsets, all_archaea_mean_tol3 - 2 * all_archaea_std_tol3, all_archaea_mean_tol3 + 2 * all_archaea_std_tol3,
                    alpha=0.2, color='red')
    box = ax[1,1].get_position()
    ax[1,1].set_position([box.x0 + box.width * 0.05, box.y0 + box.height * 0.6, box.width, box.height * 0.75])
    # data percentage
    ax[1,1].set_xlabel("fraction of data", fontsize=12.5)
    ax[1,1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1    ])
    ax[1,1].set_xticks([0.25, 0.5, 0.75, 1])

    ax[1,1].set_xticklabels([0.25, 0.5, 0.75, 1], fontsize=12.5)
    ax[1,1].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
    ax[0, 0].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    ax[0, 1].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    ax[1, 0].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    ax[1, 1].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)

    all_handles, all_labels = [], []
    for ind in range(2):
        for ind_ in  range(2):
            handles, labels = ax[ind, ind_].get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

    fig.legend(all_handles, all_labels, loc='center left', bbox_to_anchor=(0.03, 0.1), ncol=4, fontsize=12.5)
    plt.savefig("some_plot_perf_over_data_perc.pdf")

def checkthis_():
    a = pickle.load(open("train_subset_results/data_perc_runs_dos_0.1_lr_1e-05_nlayers_3_nhead_16_run_no_0_subset_train_0.25_trFlds_0_1_best.bin", "rb"))
    b = pickle.load(open("train_subset_results/data_perc_runs_dos_0.1_lr_1e-05_nlayers_3_nhead_16_run_no_1_subset_train_0.25_trFlds_0_1_best.bin", "rb"))
    for k in a.keys():
        if a[k] != b[k]:
            print(k)
            print(a[k])
            print(b[k])
            print("\n")


def plot_ece_over_tolerance(lg_and_tol2_lg):
    colors = {'EUKARYA':'blue', 'NEGATIVE':'orange', 'POSITIVE':'red', 'ARCHAEA':'purple'}
    fig, ax = plt.subplots()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=26)
    for lg in ['EUKARYA', 'NEGATIVE', 'POSITIVE', 'ARCHAEA']:
        ece_values = []
        for tol in range(4):
            ece_values.append(lg_and_tol2_lg["{}_{}".format(lg, tol)])
        ax.plot(list(range(4)), ece_values, label=lg, color=colors[lg])
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=26)
    plt.xticks(list(range(4)))
    plt.title("ECE values over tolerance levels", fontsize=26)
    plt.xlabel("Tolerance")
    plt.ylabel("log(ECE)")
    plt.yscale("log")
    plt.show()

def extract_performance_over_tolerance():
    def plot_spase12_pairs(negative_mean,positive_mean,archaea_mean,negative_std,positive_std,archaea_std,
                           negative_sp2_mean,positive_sp2_mean,archaea_sp2_mean,negative_sp2_std,positive_sp2_std,archaea_sp2_std,
                           type='Tat'):
        tolerances = list(range(4))
        import matplotlib as mpl
        mpl.rcParams['figure.dpi'] = 350
        mpl.rcParams['font.family'] = "Arial"
        import matplotlib.gridspec as gridspec
        fig, ax = plt.subplots(3, 2, figsize=(8, 6))
        ax[0, 0].set_title(type+"/SPase " + "I" if type=='Tat' else type + "/SPase " + "II")
        ax[0, 0].plot(tolerances, negative_mean, '-', label='gn negative', color='orange')
        ax[0, 0].fill_between(tolerances, negative_mean - 2 * negative_std, negative_mean + 2 * negative_std, alpha=0.2,
                              color='orange')
        ax[0, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax[0, 0].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
        ax[0, 0].set_xticks([0, 1, 2, 3])
        ax[0, 0].set_xticklabels(["0", "1", "2", "3"], fontsize=12.5)
        box = ax[0, 0].get_position()
        ax[0, 0].set_position([box.x0, box.y0 + box.height * 0.5, box.width, box.height * 0.78])
        # ax[0, 0].yaxis.set_label_coords(-0.2, 0.5)

        ax[1, 0].plot(tolerances, positive_mean, '-', label='gp bacteria', color='purple')
        ax[1, 0].fill_between(tolerances, positive_mean - 2 * positive_std, positive_mean + 2 * positive_std, alpha=0.2,
                              color='purple')
        ax[1, 0].set_xticks([0, 1, 2, 3])
        ax[1, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax[1, 0].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
        ax[1, 0].set_xticklabels(["0", "1", "2", "3"], fontsize=12.5)
        box = ax[1, 0].get_position()
        ax[1, 0].set_position([box.x0, box.y0 + box.height * 0.5, box.width, box.height * 0.78])


        ax[2, 0].plot(tolerances, archaea_mean, '-', label='archaea', color='red')
        ax[2, 0].fill_between(tolerances, archaea_mean - 2 * archaea_std, archaea_mean + 2 * archaea_std, alpha=0.2,
                              color='red')
        ax[2, 0].set_xticks([0, 1, 2, 3])
        ax[2, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax[2, 0].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
        ax[2, 0].set_xticklabels(["0", "1", "2", "3"], fontsize=12.5)
        box = ax[2, 0].get_position()
        ax[2, 0].set_position([box.x0, box.y0 + box.height * 0.5, box.width, box.height * 0.78])

        ax[0, 1].set_title(type + "/SPase " + "II" if type == 'Tat' else type + "/SPase " +  "IV")
        ax[0, 1].plot(tolerances, negative_sp2_mean, '-', label='gn negative', color='orange')
        ax[0, 1].fill_between(tolerances, negative_sp2_mean - 2 * negative_sp2_std, negative_sp2_mean + 2 * negative_sp2_std, alpha=0.2,
                              color='orange')
        ax[0, 1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax[0, 1].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
        ax[0, 1].set_xticks([0, 1, 2, 3])
        ax[0, 1].set_xticklabels(["0", "1", "2", "3"], fontsize=12.5)
        box = ax[0, 1].get_position()
        ax[0, 1].set_position([box.x0, box.y0 + box.height * 0.5, box.width, box.height * 0.78])
        ax[0, 1].yaxis.set_label_coords(-0.2, 0.5)

        ax[1, 1].plot(tolerances, positive_sp2_mean, '-', label='gp bacteria', color='purple')
        ax[1, 1].fill_between(tolerances, positive_sp2_mean - 2 * positive_sp2_std, positive_sp2_mean + 2 * positive_sp2_std, alpha=0.2,
                              color='purple')
        ax[1, 1].set_xticks([0, 1, 2, 3])
        ax[1, 1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax[1, 1].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
        ax[1, 1].set_xticklabels(["0", "1", "2", "3"], fontsize=12.5)
        box = ax[1, 1].get_position()
        ax[1, 1].set_position([box.x0, box.y0 + box.height * 0.5, box.width, box.height * 0.78])

        ax[2, 1].plot(tolerances, archaea_sp2_mean, '-', label='archaea', color='red')
        ax[2, 1].fill_between(tolerances, archaea_sp2_mean - 2 * archaea_sp2_std, archaea_sp2_mean + 2 * archaea_sp2_std, alpha=0.2,
                              color='red')
        ax[2, 1].set_xticks([0, 1, 2, 3])
        ax[2, 1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax[2, 1].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
        ax[2, 1].set_xticklabels(["0", "1", "2", "3"], fontsize=12.5)
        box = ax[2, 1].get_position()
        ax[2, 1].set_position([box.x0, box.y0 + box.height * 0.5, box.width, box.height * 0.78])
        ax[0, 0].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
        ax[0, 1].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
        ax[1, 0].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
        ax[1, 1].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
        ax[2, 1].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
        ax[2, 0].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
        ax[0, 0].set_ylabel("F1 score", fontsize=12.5)
        ax[1, 0].set_ylabel("F1 score", fontsize=12.5)
        ax[2, 0].set_ylabel("F1 score", fontsize=12.5)

        all_handles, all_labels = [], []
        for ind in range(3):
            handles, labels = ax[ind, 0].get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

        fig.legend(all_handles, all_labels, loc='center left', bbox_to_anchor=(0.125, 0.09), ncol=4, fontsize=12.5)
        plt.savefig("some_plot_perf_over_tolerance_{}.pdf".format("tat" if type=="Tat" else "sp2"))
    sp1_f1s, sp1_recs, sp1_precs, sp2_f1s, sp2_recs, sp2_precs, tat_f1s, \
        tat_recs, tat_precs, mcc1_sp1, mcc2_sp1, mcc1_sp2, mcc2_sp2, mcc1_tat, mcc2_tat, \
        tatlipo_f1s, tatlipo_recs, tatlipo_precs, pilin_f1s, pilin_recs, pilin_precs, \
        mcc1_tatlipo, mcc2_tatlipo, mcc1_pilin, mcc2_pilin\
        = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],[],[],[],[],[],[],[],[],[]
    runs = [32,34,37, 39,40, 47]
    runs = [59,60,61,62,71]

    for run_no in runs:
        run_results_folder = "tuning_bert_fixed_high_lr_swa_only_repeat{}/".format(run_no)
        mdl2results = extract_all_param_results(only_cs_position=False,
                                                result_folder=run_results_folder,
                                                compare_mdl_plots=False,
                                                remove_test_seqs=False,
                                                benchmark=False,
                                                prints=True)
        mdl_ind = 0
        sp1_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-7])]))
        sp1_recs.append(np.array([rec for rec in mdl2results[mdl_ind][10]]))
        sp1_precs.append(np.array([rec for rec in mdl2results[mdl_ind][11]]))
        sp2_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-6])]))
        sp2_recs.append(np.array([rec for rec in mdl2results[mdl_ind][12]]))
        sp2_precs.append(np.array([rec for rec in mdl2results[mdl_ind][13]]))
        tat_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-5])]))
        tat_recs.append(np.array([rec for rec in mdl2results[mdl_ind][14]]))
        tat_precs.append(np.array([rec for rec in mdl2results[mdl_ind][15]]))
        tatlipo_f1s.append(np.array([f1 for f1 in np.concatenate(mdl2results[mdl_ind][-4])]))
        tatlipo_recs.append(np.array([rec for rec in mdl2results[mdl_ind][16]]))
        tatlipo_precs.append(np.array([prec for prec in mdl2results[mdl_ind][17]]))
        pilin_f1s.append(np.array([f1 for f1 in np.concatenate(mdl2results[mdl_ind][-3])]))
        pilin_recs.append(np.array([rec for rec in mdl2results[mdl_ind][18]]))
        pilin_precs.append(np.array([prec for prec in mdl2results[mdl_ind][19]]))
        mcc1_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][0]]))
        mcc2_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][1][1:]]))
        mcc1_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][2]]))
        mcc2_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][3]]))
        mcc1_tat.append(np.array([mcc for mcc in mdl2results[mdl_ind][4]]))
        mcc2_tat.append([mcc for mcc in mdl2results[mdl_ind][5]])
        mcc1_tatlipo.append([mcc for mcc in mdl2results[mdl_ind][6]])
        mcc2_tatlipo.append([mcc for mcc in mdl2results[mdl_ind][7]])
        mcc1_pilin.append([mcc for mcc in mdl2results[mdl_ind][8]])
        mcc2_pilin.append([mcc for mcc in mdl2results[mdl_ind][9]])


    names = ["eukarya", "gn bacteria", "gp bacteria", "archaea"]
    sp1_f1s = np.stack(sp1_f1s)
    sp1_mean_f1s = np.mean(np.stack(sp1_f1s), axis=0)
    sp2_mean_f1s = np.mean(np.stack(sp2_f1s), axis=0)
    tat_mean_f1s = np.mean(np.stack(tat_f1s), axis=0)
    tatlipo_mean_f1s = np.mean(np.stack(tatlipo_f1s), axis=0)
    pilin_mean_f1s = np.mean(np.stack(pilin_f1s), axis=0)

    sp1_std_f1s = np.std(np.stack(sp1_f1s), axis=0)
    sp2_std_f1s = np.std(np.stack(sp2_f1s), axis=0)
    tat_std_f1s = np.std(np.stack(tat_f1s), axis=0)
    tatlipo_std_f1s = np.std(np.stack(tatlipo_f1s), axis=0)
    pilin_std_f1s = np.std(np.stack(pilin_f1s), axis=0)
    print(tatlipo_mean_f1s,  pilin_mean_f1s)
    print(np.stack(pilin_f1s).shape,  np.stack(tat_f1s).shape)
    euk_means_sp1, neg_means_sp1, pos_means_sp1, arch_means_sp1 = sp1_mean_f1s[:4], sp1_mean_f1s[4:8], sp1_mean_f1s[8:12], sp1_mean_f1s[12:]
    neg_means_sp2, pos_means_sp2, arch_means_sp2 = sp2_mean_f1s[:4], sp2_mean_f1s[4:8], sp2_mean_f1s[8:]
    neg_means_tat, pos_means_tat, arch_means_tat = tat_mean_f1s[:4], tat_mean_f1s[4:8], tat_mean_f1s[8:]
    neg_means_tatlipo, pos_means_tatlipo, arch_means_tatlipo = tatlipo_mean_f1s[:4], tatlipo_mean_f1s[4:8], tatlipo_mean_f1s[8:]
    neg_means_pilin, pos_means_pilin, arch_means_pilin = pilin_mean_f1s[:4], pilin_mean_f1s[4:8], pilin_mean_f1s[8:]

    euk_std_sp1, neg_std_sp1, pos_std_sp1, arch_std_sp1 = sp1_std_f1s[:4], sp1_std_f1s[4:8], sp1_std_f1s[8:12], sp1_std_f1s[12:]
    neg_std_sp2, pos_std_sp2, arch_std_sp2 = sp2_std_f1s[:4], sp2_std_f1s[4:8], sp2_std_f1s[8:]
    neg_std_tat, pos_std_tat, arch_std_tat = tat_std_f1s[:4], tat_std_f1s[4:8], tat_std_f1s[8:]
    neg_std_tatlipo, pos_std_tatlipo, arch_std_tatlipo = tatlipo_std_f1s[:4], tatlipo_std_f1s[4:8], tatlipo_std_f1s[8:]
    neg_std_pilin, pos_std_pilin, arch_std_pilin = pilin_std_f1s[:4], pilin_std_f1s[4:8], pilin_std_f1s[8:]

    plot_spase12_pairs(neg_means_tat, pos_means_tat, arch_means_tat,neg_std_tat, pos_std_tat, arch_std_tat,
                       neg_means_tatlipo, pos_means_tatlipo, arch_means_tatlipo, neg_std_tatlipo, pos_std_tatlipo, arch_std_tatlipo)
    plot_spase12_pairs(neg_means_sp2, pos_means_sp2, arch_means_sp2, neg_std_sp2, pos_std_sp2, arch_std_sp2,
                       neg_means_pilin, pos_means_pilin, arch_means_pilin, neg_std_pilin, pos_std_pilin,
                       arch_std_pilin, type="Sec")

    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"
    import matplotlib.gridspec as gridspec
    fig, ax = plt.subplots(2, 2,figsize=(8, 6))
    tolerances = list(range(4))
    ax[0, 0].plot(tolerances, euk_means_sp1, '-', label='eukarya', color='blue')
    ax[0, 0].fill_between(tolerances, euk_means_sp1 - 2 * euk_std_sp1, euk_means_sp1 + 2 * euk_std_sp1, alpha=0.2, color='blue')
    ax[0, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax[0, 0].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
    ax[0, 0].set_xticks([0, 1, 2, 3])
    ax[0, 0].set_xticklabels(["0", "1", "2", "3"], fontsize=12.5)
    box = ax[0, 0].get_position()
    ax[0, 0].set_position([box.x0 , box.y0 + box.height * 0.5, box.width, box.height * 0.78])
    ax[0, 0].set_ylabel("F1 score", fontsize=12.5)
    ax[0, 0].yaxis.set_label_coords(-0.2, 0.5)

    ax[0, 1].plot(tolerances, neg_means_sp1, '-', label='gn bacteria', color='orange')
    ax[0, 1].fill_between(tolerances, neg_means_sp1 - 2 * neg_std_sp1, neg_means_sp1 + 2 * neg_std_sp1, alpha=0.2, color='orange')
    ax[0, 1].set_xticks([0, 1, 2, 3])
    ax[0, 1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax[0, 1].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
    ax[0, 1].set_xticklabels(["0", "1", "2", "3"], fontsize=12.5)
    box = ax[0, 1].get_position()
    ax[0, 1].set_position([box.x0 , box.y0 + box.height * 0.5, box.width, box.height * 0.78])

    ax[1, 0].plot(tolerances, pos_means_sp1, '-', label='gp bacteria', color='purple')
    ax[1, 0].fill_between(tolerances, pos_means_sp1 - 2 * pos_std_sp1, pos_means_sp1 + 2 * pos_std_sp1, alpha=0.2, color='purple')
    ax[1, 0].set_xticks([0, 1, 2, 3])
    ax[1, 0].set_xticklabels(["0", "1", "2", "3"], fontsize=12.5)
    ax[1, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax[1, 0].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
    box = ax[1, 0].get_position()
    ax[1, 0].set_position([box.x0 , box.y0 + box.height * 0.5, box.width, box.height * 0.78])
    ax[1, 0].set_ylabel("F1 score", fontsize=12.5)
    ax[1, 0].yaxis.set_label_coords(-0.2, 0.5)
    ax[1, 0].set_xlabel("tolerance", fontsize=12.5)

    ax[1, 1].plot(tolerances, arch_means_sp1, '-', label='archaea', color='red')
    ax[1, 1].fill_between(tolerances, arch_means_sp1 - 2 * arch_std_sp1, arch_means_sp1 + 2 * arch_std_sp1, alpha=0.2, color='red')
    ax[1, 1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax[1, 1].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
    ax[1, 1].set_xticks([0, 1, 2, 3])
    ax[1, 1].set_xticklabels(["0", "1", "2", "3"], fontsize=12.5)
    ax[1, 1].set_xlabel("tolerance", fontsize=12.5)
    box = ax[1, 1].get_position()
    ax[1, 1].set_position([box.x0 , box.y0 + box.height * 0.5, box.width, box.height * 0.78])

    ax[0, 0].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    ax[0, 1].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    ax[1, 0].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    ax[1, 1].grid(axis='y', color='black', linestyle='-', linewidth=0.5, alpha=0.4)

    all_handles, all_labels = [], []
    for ind in range(2):
        for ind_ in range(2):
            handles, labels = ax[ind, ind_].get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

    fig.legend(all_handles, all_labels, loc='center left', bbox_to_anchor=(0.125, 0.09),ncol=4, fontsize=12.5)
    plt.savefig("some_plot_perf_over_tolerance.pdf")
    exit(1)
    plt.show()

    # ax[0].plot(subsets, all_euk_mean_tol3, '--',label='e3', color='blue')
    # ax[0].fill_between(subsets, all_euk_mean_tol3 - 2 * all_euk_std_tol3, all_euk_mean_tol3 + 2 * all_euk_std_tol3, alpha=0.2, color='blue')

def plot_compare_pos_nopos():
    nopos_bench = [0.563, 0.679, 0.728, 0.752, 0.346, 0.506, 0.531, 0.556, 0.364, 0.409, 0.5, 0.545, 0.4, 0.615, 0.646, 0.646]
    nopos_nobench = [0.725, 0.853, 0.894, 0.918, 0.614, 0.722, 0.744, 0.764, 0.464, 0.594, 0.669, 0.717, 0.469, 0.667, 0.691, 0.691]
    linearpos_bench = [0.347, 0.687, 0.767, 0.82, 0.24, 0.507, 0.547, 0.587, 0.27, 0.378, 0.649, 0.649, 0.219, 0.469, 0.625, 0.688]
    linearpos_nobench = [0.494, 0.833, 0.892, 0.919, 0.467, 0.745, 0.788, 0.809, 0.292, 0.498, 0.648, 0.769, 0.275, 0.55, 0.675, 0.725]
    pos_bench = [0.692, 0.737, 0.769, 0.782, 0.462, 0.564, 0.59, 0.59, 0.526, 0.684, 0.684, 0.684, 0.606, 0.697, 0.697, 0.727]
    pos_nobench = [0.8, 0.869, 0.907, 0.928, 0.759, 0.807, 0.82, 0.828, 0.664, 0.728, 0.756, 0.763, 0.659, 0.732, 0.732, 0.756]
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"

    fig, ax = plt.subplots(2, 1)
    line_w = 0.25
    x_positions = []
    names = ["with pe", "w/o pe"]
    datasets = ["benchmark\ndataset\n(CS F1 score)", 'whole\ndataset\n(CS F1 score)']
    colors = ["red", "blue", "green", "black"]
    names_xticks = ["", "", "", ""]
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch

    xticklbls = []
    for lg in names_xticks:
        for tol in range(4):
            xticklbls.append(lg+str(tol))
    no_pos_enc = [nopos_bench, nopos_nobench]
    pos_enc = [pos_bench, pos_nobench]
    lin_pos_enc = [linearpos_bench, linearpos_nobench]
    for i in range(2):
        ax[i].plot([3.5,3.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        ax[i].plot([7.5,7.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        ax[i].plot([11.5,11.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        box = ax[i].get_position()
        ax[i].set_position([box.x0 + box.width *0.14, box.y0 + box.height*0.6, box.width * 0.9, box.height * 0.7])
        x_pos_nopos = np.array(list(range(len(nopos_bench)))) + line_w
        x_pos_lin_pos = np.array(list(range(len(nopos_bench))))
        x_pos_pos = np.array(list(range(len(nopos_bench)))) - line_w
        if i == 0:
            ax[i].bar(x_pos_pos, pos_enc[i], width=line_w, color='red',
                                  label="sine positional encoding")
            ax[i].bar(x_pos_lin_pos, lin_pos_enc[i], width=line_w, color='orange',
                      label="linear positional encoding")
            ax[i].bar(x_pos_nopos, no_pos_enc[i], width=line_w, color='black',
                                  label="no positional encoding")

        else:
            ax[i].bar(x_pos_pos, pos_enc[i], width=line_w, color='red')
            ax[i].bar(x_pos_lin_pos, lin_pos_enc[i], width=line_w, color='orange')
            ax[i].bar(x_pos_nopos, no_pos_enc[i], width=line_w, color='black')
        ax[i].set_ylim(0, 1)
        ax[i].set_xlim(-0.5, 15.5)
        ax[i].set_xticks(list(range(16)))
        ax[i].set_xticklabels(xticklbls, fontsize=12.5)
        ax[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax[i].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
        ax[i].set_ylabel(datasets[i], fontsize=12.5, rotation=0)
        ax[i].yaxis.set_label_coords(-0.2, 0.35)

        # ax[i//2, i%2].set_xticklabels(["{}\n{}".format(str(round(accs[i], 2)), str(total_counts_per_acc[i])) for i in
        #             range(len(accs)//2 - 2)], fontsize=12.5)
        # ax[i // 2, i % 2].set_yticks([0.5, 1])
        # ax[i // 2, i % 2].set_yticklabels([0.5, 1], fontsize=6)
    ax[1].set_xlabel("   eukarya               gn bacteria           gp bacteria           archaea\n\n tolerance/life group", fontsize=12.5)
    # ax[0].annotate('', xy=(3.5, 1), xycoords='axes fraction', xytext=(3.5, 2),
    #             arrowprops=dict(arrowstyle="-", color='b'))
    from matplotlib.patches import RegularPolygon, Rectangle
    # tri = Rectangle((752,150), width=2, height=800, color='purple', alpha=0.3)
    # fig.patches.append(tri)
    # tri = Rectangle((1075,150), width=1, height=800, color='purple', alpha=0.3)
    # fig.patches.append(tri)
    # tri = Rectangle((1398,150), width=1, height=800, color='purple', alpha=0.3)
    # fig.patches.append(tri)
    # dotted_line(752,100, patches=fig.patches)
    # dotted_line(1075,100, patches=fig.patches)
    # dotted_line(1398,100, patches=fig.patches)



    fig.legend(loc='center left', bbox_to_anchor=(0, 0.05), fontsize=12.5, ncol=3)
    plt.show()


def dotted_line(x, y_start, height=800, patches=None):
    from matplotlib.patches import RegularPolygon, Rectangle, Patch, Arrow
    mini_line_height = int((height*3/5)/30)
    # for i in range(30):
        # mini_line = Patch((x, y_start + (i * height)//20), height=mini_line_height, color='black')
    mini_line = Arrow(x, y_start, 0, y_start+800, color='black', alpha=1)
    patches.append(mini_line)

def compare_experiment_results():
    separate_bert_tuning = [0.781, 0.843, 0.885, 0.903, 0.701, 0.764, 0.772, 0.782, 0.517, 0.55, 0.57, 0.584, 0.593, 0.642, 0.642, 0.667  ]
    tuning_bert_together = [0.799, 0.853, 0.894, 0.915, 0.772, 0.796, 0.807, 0.812, 0.717, 0.746, 0.768, 0.775, 0.543, 0.617, 0.667, 0.667  ]
    no_bert_tuning = [0.675, 0.749, 0.829, 0.863, 0.615, 0.676, 0.704, 0.714, 0.439, 0.516, 0.542, 0.568, 0.487, 0.564, 0.59, 0.615  ]
    tuning_bert_only_enc = [0.8  , 0.869, 0.907, 0.928, 0.759, 0.807, 0.82 , 0.828, 0.664, 0.728, 0.756, 0.763, 0.659, 0.732, 0.732, 0.756 ]
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"
    fig, ax = plt.subplots(1, 1)
    line_w = 0.4
    x_positions = []
    names = ["with pe", "w/o pe"]
    datasets = ["benchmark\ndataset\n(CS F1 score)", 'whole\ndataset\n(CS F1 score)']
    colors = ["red", "blue", "green", "black"]
    names_xticks = ["", "", "", ""]
    xticklbls = []
    for lg in names_xticks:
        for tol in range(4):
            xticklbls.append(lg+str(tol))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.35, box.width * 1, box.height * 0.75])
    x_pos_notuning = np.array(list(range(len(separate_bert_tuning)))) + line_w * 0.75
    x_pos_notuning *= 2
    x_pos_tune_sep = np.array(list(range(len(separate_bert_tuning)))) + line_w * 0.25
    x_pos_tune_sep *= 2
    x_pos_tune_together = np.array(list(range(len(separate_bert_tuning)))) - line_w * 0.25
    x_pos_tune_together *= 2
    x_pos_tune_together_onlyEnc = np.array(list(range(len(separate_bert_tuning)))) - line_w * 0.75
    x_pos_tune_together_onlyEnc *= 2
    ax.bar(x_pos_tune_together_onlyEnc, tuning_bert_only_enc, width=line_w, color='black',
                          label="T decoder + tuning LM")
    ax.bar(x_pos_tune_together, tuning_bert_together, width=line_w, color='red',
                          label="T encoder-decoder + tuning LM")
    ax.bar(x_pos_tune_sep, separate_bert_tuning, width=line_w, color='purple',
                          label="T encoder-decoder + separate tuning LM")
    ax.bar(x_pos_notuning, no_bert_tuning, width=line_w, color='blue',
                          label="T encoder-decoder (no LM tuning)")
    ax.set_ylim(0, 1)
    xticks = list(np.array(list(range(16)))*2)
    xticks.append(-3)
    ax.set_xticks(np.array(list(range(16)))*2)
    ax.set_xticklabels(xticklbls, fontsize=12.5)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12.5)
    ax.set_ylabel("F1 score", fontsize=12.5)
    ax.set_xlim(-1, 31)
    ax.set_xlabel("eukarya                gn bacteria           gp bacteria                 archaea\n\n "
                  "    tolerance/life group",fontsize=12.5)

    ax.plot([7, 7], [0, 1.5], linestyle='--', dashes=(1, 1), color='black')
    ax.plot([15, 15], [0, 1.5], linestyle='--', dashes=(1, 1), color='black')
    ax.plot([23, 23], [0, 1.5], linestyle='--', dashes=(1, 1), color='black')
    # ax.yaxis.set_label_coords(-0.18, 0.45)

        # ax[i//2, i%2].set_xticklabels(["{}\n{}".format(str(round(accs[i], 2)), str(total_counts_per_acc[i])) for i in
        #             range(len(accs)//2 - 2)], fontsize=12.5)
        # ax[i // 2, i % 2].set_yticks([0.5, 1])
        # ax[i // 2, i % 2].set_yticklabels([0.5, 1], fontsize=6)
    # fig.legend(loc='center left', bbox_to_anchor=(0.8, 0.5), fontsize=12.5)
    fig.legend(loc='center left', bbox_to_anchor=(0.05, 0.09), fontsize=12.5, ncol=2)
    from matplotlib.patches import RegularPolygon, Rectangle
    # tri = Rectangle((587,150), width=2, height=800, color='black')
    # dotted_line(587, 150, patches=fig.patches)
    # dotted_line(946, 150, patches=fig.patches)
    # dotted_line(1304, 150, patches=fig.patches)
    # # fig.patches.append(tri)
    # tri = Rectangle((946,150), width=1, height=800, color='black')
    # fig.patches.append(tri)
    # tri = Rectangle((1304,150), width=1, height=800, color='black')
    # fig.patches.append(tri)

    plt.show()
def sanity_check(a, b):
    dict_a = {a_[0]:[a_[1],a_[2],a_[3]] for a_ in a}
    for elem in b:
        if elem[0] in dict_a:
            if sum(dict_a[elem[0]][-1] - elem[-1]) != 0 or sum(dict_a[elem[0]][-2] - elem[-2]) != 0:
                print("elem")
    print("Welp... :/")
    exit(1)

def get_hydro_values():
    hydro_vals = {"A":1.8, "C":2.5, "D":-3.5, "E":-3.5, "F":2.8, "G":-0.4, "H":-3.2, "I":4.5, "K":-3.9,"L":3.8, "M":1.9, "N":-3.5, "P":-1.6, "Q":-3.5,"R":-4.5,"S":-0.8,"T":-0.7,"V":4.2,"W":-0.9,"Y":-1.3}

def visualize_inp_gradients(files=["added_oh_gen_fold0.bin", "added_oh_gen_fold1.bin", "added_oh_gen_fold2.bin"]):
    """

    Using files resulted from running python main.py --compute_saliency for some model on some data, create the mean
    importance scores for sequences aligned to the RR motif (for Tat/SPase I) and Cys residue (for Sec/SPase II) and
    plot the normalized values over the residues.

    """
    import matplotlib as mpl
    from matplotlib.ticker import MultipleLocator

    hydro_vals = {"A":1.8, "C":2.5, "D":-3.5, "E":-3.5, "F":2.8, "G":-0.4, "H":-3.2, "I":4.5, "K":-3.9,"L":3.8, "M":1.9,
                  "N":-3.5, "P":-1.6, "Q":-3.5,"R":-4.5,"S":-0.8,"T":-0.7,"V":4.2,"W":-0.9,"Y":-1.3}

    folder = get_data_folder()
    seq2lbls = {}
    seq2lg = {}
    # gather sequence info from data
    for i in [0,1,2]:
        for t in ["train", "test"]:
            a = pickle.load(open(folder+"sp6_partitioned_data_{}_{}.bin".format(t,i), "rb"))
            seq2lbls.update({seq:lbls[1] for seq,lbls in a.items()})
            seq2lg.update({seq:lbls[-1] for seq,lbls in a.items()})
    sequences_and_grad_values = []
    for f in files:
        if len(sequences_and_grad_values) == 0:
            sequences_and_grad_values = pickle.load(open(f, "rb"))
        else:
            sequences_and_grad_values.extend(pickle.load(open(f, "rb")))
    ss = set()
    tbt = []
    for s, a1,a2,a3 in sequences_and_grad_values:
        if s not in ss:
            tbt.append((s,a1,a2,a3))
            ss.add(s)
    sequences_and_grad_values = tbt
    norm_values = np.zeros(150)
    counts = np.zeros(150)
    norm_values_sp1 = np.zeros(60)
    counts_sp1 = 0
    norm_values_sp2 = np.zeros(60)
    counts_sp2 = 0
    motif_test = "FLK"
    seqs2glbl_lbl = {}
    # gather sequence info from data
    for t in ['train', 'test']:
        for fold in [0,1,2]:
            dict_ = pickle.load(open("../sp_data/sp6_partitioned_data_{}_{}.bin".format(t,fold), "rb"))
            seqs2glbl_lbl.update({k:v[-1] for k,v in dict_.items()})
    avg_cs_pos = []
    right5_hydro_vals = []
    right5_hydro_aas = []
    all_norm_values = []
    all_cs_positions = []
    all_tat_seqs = []
    for seq, lbls, spTypeGrds, spCSgrds in sequences_and_grad_values:
        if lbls[0] == "T" and seqs2glbl_lbl[seq] == "TAT":
            if motif_test in seq[:lbls.rfind("T")] and seq[-3+seq.find(motif_test):+seq.find(motif_test)-1] == "RR":
                rr_seq = seq.find(motif_test)
                right5_hydro_vals.append([hydro_vals[s_] for s_ in seq[rr_seq:rr_seq+15]])
                right5_hydro_aas.append(seq[rr_seq:rr_seq+5])
                normalized_C_cs_values = np.array(spTypeGrds) / np.sum(spTypeGrds)
                norm_values[75-rr_seq:75+len(seq)-rr_seq]+=normalized_C_cs_values
                counts[75-rr_seq:75+len(seq)-rr_seq]+= np.ones(len(seq))
                current_norm_val = np.zeros(150)
                current_norm_val[75-rr_seq:75+len(seq)-rr_seq] += normalized_C_cs_values
                all_norm_values.append(current_norm_val)
                all_cs_positions.append(lbls.rfind("T"))
                avg_cs_pos.append(lbls.rfind("T")-rr_seq)
                all_tat_seqs.append(seq)
        if lbls[0] == "S" and seq2lbls[seq][0] == "S" and len(seq) >= 60:
            norm_values_sp1 += np.array(spTypeGrds[:60]) / np.sum(spTypeGrds[:60])
            counts_sp1 += 1
        if lbls[0] == "L" and seq2lbls[seq][0] == "L" and len(seq) >= 60:
            norm_values_sp2 += np.array(spTypeGrds[:60]) / np.sum(spTypeGrds[:60])
            counts_sp2 +=1
    start_ind, end_ind = 0, 0
    for i in range(150):
        if counts[i] not in [0, 1] and start_ind==0:
            start_ind = i
        elif counts[i] in [0, 1] and start_ind != 0 and end_ind == 0:
            end_ind = i-1
    normalized_Tat_values = norm_values[start_ind:end_ind]/counts[start_ind:end_ind]
    xticks_str = " "*(75-start_ind - 3) + "RRXFLK" + " "* (len(normalized_Tat_values)- 75 + start_ind - 3)
    motif_pos = xticks_str.find("RRXFLK")
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"
    fig, axs = plt.subplots(2, 1,figsize=(8, 6))
    inbetween_residues_mean_1 = (normalized_Tat_values[motif_pos] + normalized_Tat_values[motif_pos-1])/2
    inbetween_residues_mean_2 = (normalized_Tat_values[motif_pos+5] + normalized_Tat_values[motif_pos+6])/2

    box = axs[0].get_position()
    axs[0].set_position([box.x0 + box.width * 0.1, box.y0 + box.height * 0.3, box.width * 0.95, box.height * 0.85])

    axs[0].plot(range(motif_pos), normalized_Tat_values[:motif_pos], color='#1f77b4', label='Other Tat SP residues')
    axs[0].plot([motif_pos-1, motif_pos-0.5], [normalized_Tat_values[motif_pos-1], inbetween_residues_mean_1], color='#1f77b4')
    axs[0].plot([motif_pos-0.5, motif_pos], [inbetween_residues_mean_1, normalized_Tat_values[motif_pos]], color='#ff7f0e')
    axs[0].plot(range(motif_pos, motif_pos + 6), normalized_Tat_values[motif_pos:motif_pos+6], color='#ff7f0e')

    axs[0].plot([motif_pos+5, motif_pos+5.5], [normalized_Tat_values[motif_pos+5], inbetween_residues_mean_2], color='#ff7f0e', label='Twin arginine motif')
    axs[0].plot([motif_pos+5.5, motif_pos+6], [inbetween_residues_mean_2, normalized_Tat_values[motif_pos+6]], color='#1f77b4')#'#1f77b4')

    axs[0].plot(range(motif_pos+6, len(normalized_Tat_values)), normalized_Tat_values[motif_pos + 6:], color='#1f77b4')
    axs[0].set_ylabel('SP type residue scores',  fontsize=12.5)
    axs[0].xaxis.set_major_locator(MultipleLocator(11))
    axs[0].xaxis.set_major_formatter('{x:.0f}')
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].set_yticks([0, 0.01,0.02,0.03,0.04,0.05])
    axs[0].set_yticklabels([0, 0.01,0.02,0.03,0.04,0.05], fontsize=12.5)
    lbls = [5, -11, -0, 11 ,22, 33]
    axs[0].set_xticklabels([str(n) for n in lbls ],fontsize=12.5)
    axs[0].set_xlim(-2, 89)
    for ind_ in range(0,50):
        top_val = (normalized_Tat_values[ind_]+normalized_Tat_values[ind_+1])/2
        axs[0].plot([ind_+0.5, ind_+0.5],[-0.02,top_val], linestyle='--', color='black', linewidth=0.3,alpha=0.3)
    motif_positions = []
    seqs = ["MQRRHFLKNAAAALAALGLPTLPQWALAAKAVGLRRLGQPQPFDYAWLKGQARELANAPYKSHKQLLPGP",
            "MPNRRDFLKTAAFATLGSGIAVSQVLAGECMPSAIHINKYGIGGKMKMTFFPYELKLRHVFTVATYSRTT",
            "MLMYRRDFLKSVTAAWVAFGLPNPLGGAFATNRVIPLRRLGQSQRFDYEWLKERARALAATPYHSRKRVL",
            ["..."]*50,
            "MNTNNEETFYQAMRRKGVSRRSFLKYCSLAATSLGLGAAMTPRIAWALENKPRIPVVWIHGLECTCCTES",
            ]
    for s in all_tat_seqs:
        motif_positions.append(s.find("FLK"))
    for seq_ind, s in enumerate(seqs):
        start = motif_pos - s.find("FLK")+2 if "FLK" in s else -1
        dotdotdot = type(s) == list
        for ind_, s_ in enumerate(s[:60]):
            if dotdotdot:
                if 17 < start+ind_ < 24:
                    axs[0].annotate(".", xy=(13, normalized_Tat_values[13]),
                                    xytext=(start + ind_ + 0.80, 0.0035 * (1 - seq_ind)), fontsize=4,color='#ff7f0e')
                    axs[0].annotate(".", xy=(13, normalized_Tat_values[13]),
                                    xytext=(start + ind_ + 0.80, 0.0035 * (1 - seq_ind) - 0.001), fontsize=4,color='#ff7f0e')
                    axs[0].annotate(".", xy=(13, normalized_Tat_values[13]),
                                    xytext=(start + ind_ + 0.80, 0.0035 * (1 - seq_ind) - 0.002), fontsize=4,color='#ff7f0e')
                elif 6<start+ind_ < 39:
                    axs[0].annotate(".", xy=(13, normalized_Tat_values[13]), xytext=(start+ind_+0.80 + 0.1, 0.0035*(1-seq_ind) -0.000), fontsize=4)
                    axs[0].annotate(".", xy=(13, normalized_Tat_values[13]), xytext=(start+ind_+0.80 + 0.1, 0.0035*(1-seq_ind) - 0.001), fontsize=4)
                    axs[0].annotate(".", xy=(13, normalized_Tat_values[13]), xytext=(start+ind_+0.80 + 0.1, 0.0035*(1-seq_ind) - 0.002), fontsize=4)
            else:
                if seq_ind == len(seqs) - 1:
                    y_offset = -0.002
                else:
                    y_offset=0
                if 17 < start+ind_ < 24:
                    axs[0].annotate(s_, xy=(13, normalized_Tat_values[13]),
                                    xytext=(start + ind_ + 0.80, -0.001 + 0.0035 * (1 - seq_ind)+y_offset), fontsize=4,color='#ff7f0e')
                elif 6< start+ind_ < 38:
                    axs[0].annotate(s_, xy=(13, normalized_Tat_values[13]), xytext=(start+ind_+0.80, -0.001+0.0035*(1-seq_ind)+y_offset), fontsize=4)
                elif start+ind_ == 38:
                    axs[0].annotate("...", xy=(13, normalized_Tat_values[13]), xytext=(start+ind_+0.80, 0.0035*(1-seq_ind)+y_offset), fontsize=4)


    axs[0].set_ylim(-0.02,0.03)
    axs[0].set_xlim(7,40)
    axs[0].set_xlabel("Residue position (relative to RRXFLK motif of Tat/SPase I)", fontsize=12.5)

    normalized_sp1_cs_values_pm_5aas = np.zeros(150)
    normalized_sp1_cs_values_pm_5aas_counts = np.zeros(150)
    for seq, lbls, spTypeGrds, spCSgrds in sequences_and_grad_values:
        if lbls[0] == "S" and seq2lbls[seq][0] == "S":
            normalized_C_cs_values = np.array(spCSgrds)/np.sum(spCSgrds)
            normalized_C_cs_values = normalized_C_cs_values[:len(seq)]
            cs_pred = lbls[:-1].rfind("S") + 1
            if cs_pred < 60:
                normalized_sp1_cs_values_pm_5aas[75-cs_pred:75+len(seq)-cs_pred] += normalized_C_cs_values
                normalized_sp1_cs_values_pm_5aas_counts[75-cs_pred:75+len(seq)-cs_pred] += 1
    start_ind_sp1, end_ind_sp1 = 0, 0
    for i in range(150):
        if normalized_sp1_cs_values_pm_5aas_counts[i] not in [0] and start_ind_sp1==0:
            start_ind_sp1 = i
        elif normalized_sp1_cs_values_pm_5aas_counts[i] in [0] and start_ind_sp1 != 0 and end_ind_sp1 == 0:
            end_ind_sp1 = i-1
    normalized_C_cs_values_pm_5aas = np.zeros(150)
    normalized_C_cs_values_pm_5aas_counts = np.zeros(150)
    for seq, lbls, spTypeGrds, spCSgrds in sequences_and_grad_values:
        if lbls[0] == "L" and seq2lbls[seq][0] == "L" and lbls.rfind("L")+1 < len(seq) and seq[lbls.rfind("L")+1] == "C":
            normalized_C_cs_values = np.array(spCSgrds)/np.sum(spCSgrds)
            normalized_C_cs_values = normalized_C_cs_values[:len(seq)]
            cs_pred = lbls[:-1].rfind("L") + 1
            normalized_C_cs_values_pm_5aas[75-cs_pred:75+len(seq)-cs_pred] += normalized_C_cs_values
            normalized_C_cs_values_pm_5aas_counts[75-cs_pred:75+len(seq)-cs_pred] += 1
    start_ind, end_ind = 0, 0
    for i in range(150):
        if normalized_C_cs_values_pm_5aas[i] not in [0, 1] and start_ind==0:
            start_ind = i
        elif normalized_C_cs_values_pm_5aas[i] in [0, 1] and start_ind != 0 and end_ind == 0:
            end_ind = i-1
    normalized_C_cs_values_pm_5aas = normalized_C_cs_values_pm_5aas[start_ind:end_ind] / normalized_C_cs_values_pm_5aas_counts[start_ind:end_ind]
    inbetween_cysteine1 = (normalized_C_cs_values_pm_5aas[75-start_ind-1]+normalized_C_cs_values_pm_5aas[75-start_ind])/2
    inbetween_cysteine2 = (normalized_C_cs_values_pm_5aas[75-start_ind] + normalized_C_cs_values_pm_5aas[75-start_ind+1])/2

    box = axs[1].get_position()
    axs[1].set_position([box.x0 + box.width * 0.1, box.y0+box.height*0.1, box.width * 0.95, box.height * 0.85])

    axs[1].set_ylabel('CS residue scores', fontsize=12.5)
    axs[1].xaxis.set_major_locator(MultipleLocator(18))
    axs[1].xaxis.set_major_formatter('{x:.0f}')
    axs[1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1].set_yticks([0,0.01,0.02,0.03])
    axs[1].set_yticklabels([0,0.01,0.02,0.03], fontsize=12.5)
    lbls = [1,-36, -18, 0, 18 ,36, 51]
    axs[1].set_xticklabels([str(n) for n in lbls ], fontsize=12.5)
    axs[1].set_xlabel(r"Residue position (relative to $i_{c+1}$ of Sec/SPase I and Sec/SPase II)", fontsize=12.5)
    axs[1].set_xlim(-2, 100)
    axs[1].set_ylim(0,0.031)
    normalized_C_cs_values_pm_5aas = normalized_C_cs_values_pm_5aas[1:]
    normalized_sp1_cs_values_pm_5aas = normalized_sp1_cs_values_pm_5aas[1:]
    start_ind += 1
    start_ind_sp1 +=1
    axs[1].plot(range(75-start_ind), normalized_C_cs_values_pm_5aas[:75-start_ind], color='#1f77b4', label='Other Sec/SPII residues')
    axs[1].plot([75-start_ind-1, 75-start_ind-0.5], [normalized_C_cs_values_pm_5aas[75-start_ind-1], inbetween_cysteine1], color='#1f77b4')#'#1f77b4')
    axs[1].plot([75-start_ind-0.5, 75-start_ind], [inbetween_cysteine1, normalized_C_cs_values_pm_5aas[75-start_ind]], color='#ff7f0e')
    axs[1].plot([75-start_ind,  75-start_ind + 0.5], [normalized_C_cs_values_pm_5aas[75-start_ind], inbetween_cysteine2],color='#ff7f0e')#, color='')
    axs[1].plot([75-start_ind+0.5, 75-start_ind+1], [inbetween_cysteine2, normalized_C_cs_values_pm_5aas[75-start_ind+1]], color='#1f77b4')
    axs[1].plot(range(75-start_ind+1, len(normalized_C_cs_values_pm_5aas)), normalized_C_cs_values_pm_5aas[75-start_ind+1:], color='#1f77b4', label='Other Sec/SPII residues')
    normalized_sp1_cs_values_pm_5aas = normalized_sp1_cs_values_pm_5aas[start_ind_sp1:end_ind_sp1]/ normalized_sp1_cs_values_pm_5aas_counts[start_ind_sp1:end_ind_sp1]
    align_sp1sp2 = start_ind-start_ind_sp1
    start_ind_sp1 = start_ind_sp1 + align_sp1sp2
    normalized_sp1_cs_values_pm_5aas = normalized_sp1_cs_values_pm_5aas[align_sp1sp2:]

    inbetween_sp1cs_1 = (normalized_sp1_cs_values_pm_5aas[75-start_ind_sp1-1]+normalized_sp1_cs_values_pm_5aas[75-start_ind_sp1])/2
    inbetween_sp1cs_2 = (normalized_sp1_cs_values_pm_5aas[75-start_ind_sp1] + normalized_sp1_cs_values_pm_5aas[75-start_ind_sp1+1])/2
    axs[1].plot(range(75 - start_ind_sp1), normalized_sp1_cs_values_pm_5aas[:75 - start_ind_sp1], color='red',label='Other Sec/SPII residues')
    axs[1].plot([75 - start_ind_sp1 - 1, 75 - start_ind_sp1 - 0.5],[normalized_sp1_cs_values_pm_5aas[75 - start_ind_sp1 - 1], inbetween_sp1cs_1], color='red')
    axs[1].plot([75 - start_ind_sp1 - 0.5, 75 - start_ind_sp1],[inbetween_sp1cs_1, normalized_sp1_cs_values_pm_5aas[75 - start_ind_sp1]], color='#ff7f0e')
    axs[1].plot([75 - start_ind_sp1, 75 - start_ind_sp1 + 0.5],[normalized_sp1_cs_values_pm_5aas[75 - start_ind_sp1], inbetween_sp1cs_2], color='#ff7f0e')
    axs[1].plot([75 - start_ind_sp1 + 0.5, 75 - start_ind_sp1 + 1],[inbetween_sp1cs_2, normalized_sp1_cs_values_pm_5aas[75 - start_ind_sp1 + 1]], color='red')
    axs[1].plot(range(75 - start_ind_sp1 + 1, len(normalized_sp1_cs_values_pm_5aas)),normalized_sp1_cs_values_pm_5aas[75 - start_ind_sp1 + 1:], color='red', label='Other Sec/SPII residues')

    axs[1].annotate(r"$i_{c+1}$ Sec/SPase I", xy=(75-start_ind_sp1, normalized_sp1_cs_values_pm_5aas[75-start_ind_sp1]), xytext=(34, 0.001),
                arrowprops=dict(arrowstyle="->",linewidth=0.5), fontsize=11.5, color='#ff7f0e')
    axs[1].annotate("Sec/SPase II", xy=(4, normalized_C_cs_values_pm_5aas[4]), xytext=(1, 0.027),
                arrowprops=dict(arrowstyle="->",linewidth=0.5), fontsize=11.5, color='#1f77b4')
    axs[1].annotate(r"$i_{c+1}$ Sec/SPase II (Cys residue)", xy=(75-start_ind, normalized_C_cs_values_pm_5aas[75-start_ind]), xytext=(30, 0.027),
                arrowprops=dict(arrowstyle="->",linewidth=0.5), fontsize=11.5, color='#ff7f0e')
    axs[1].annotate("Sec/SPase I", xy=(4, normalized_sp1_cs_values_pm_5aas[4]), xytext=(1, 0.001),
                arrowprops=dict(arrowstyle="->",linewidth=0.5), fontsize=11.5, color='red')
    axs[1].set_xlim(0,50)
    plt.savefig("some_plot_imp_grads.pdf")


def rename():
    names = os.listdir("tuning_bert_fixed_high_lr_swa_only_repeat2")
    for n in names:
        os.rename("tuning_bert_fixed_high_lr_swa_only_repeat2/"+n, "tuning_bert_fixed_high_lr_swa_only_repeat2/"+n.replace("cycle_lr_s","repeat2_fixed_high_lr_"))


def compute_mcc_sp_only_mdls(mdl_name="cnn2_4resnets_tune_bert", folder="./"):
    # load_tuned_bert; bert_tuning; bert_tuning_deep, bert_tuning_crct_swa; cnn2_4resnets;cnn2_4resnets_tune_bert_
    folds = [[0,1],[0,2],[1,2]]
    res_dict = {}
    from Bio import SeqIO
    fasta_sequences = SeqIO.parse(open("../sp_data/sp6_data/train_set.fasta"), 'fasta')
    seq2id = {}
    for fasta in fasta_sequences:
        id, sequence = fasta.id, str(fasta.seq)
        len_ = len(sequence) // 2
        sequence = sequence[:len_]
        seq2id[sequence] = id.split("|")[-1]
    for fold in folds:
        test_fold = list({0,1,2}-set(fold))[0]
        file = folder+mdl_name+"_{}_{}_sp_type_test.bin".format(*fold)
        red_dict_ = pickle.load(open(file, "rb"))
        for res in red_dict_.items():
            if int(seq2id[res[0]]) == int(test_fold):
                res_dict[res[0]] = res[1]
    id2seq, id2lg, id2type, id2truelbls = extract_id2seq_dict()
    life_grps, seqs, true_lbls,pred_lbls = [], [],[],[]
    for id_ in id2lg.keys():
        life_grps.append(id2lg[id_]+"|"+id2type[id_])
        seqs.append(id2seq[id_])
        true_lbls.append(id2truelbls[id_])
        pred_lbls.append("J"*len(id2truelbls[id_]))
    print(len(seqs), len(res_dict.items()))
    mcc_sp1, mcc2_sp1 = get_pred_perf_sptype(life_grps, seqs,true_lbls,pred_lbls,v=False,return_mcc2=True,sp_type="SP",sptype_preds=res_dict)
    mcc_sp2, mcc2_sp2 = get_pred_perf_sptype(life_grps, seqs,true_lbls,pred_lbls,v=False,return_mcc2=True,sp_type="LIPO",sptype_preds=res_dict)
    mcc_tat, mcc2_tat = get_pred_perf_sptype(life_grps, seqs,true_lbls,pred_lbls,v=False,return_mcc2=True,sp_type="TAT",sptype_preds=res_dict)
    mcc_sp1 = np.array(mcc_sp1)
    mcc_sp2 = np.array(mcc_sp2)
    mcc_tat = np.array(mcc_tat)
    no_of_seqs_sp1 = np.array([2040, 44, 142, 356])
    no_of_seqs_sp2 = np.array([1087, 516, 12])
    no_of_seqs_tat = np.array([313, 39, 13])
    no_of_tested_sp_seqs = sum([2040, 44, 142, 356]) + sum([1087, 516, 12]) + sum([313, 39, 13])

    sp6_mcc1_sp1, sp6_mcc2_sp1 = np.array([0.868, 0.811,0.878,0.737]), np.array([0.649, 0.734, 0.728])
    sp6_mcc1_sp2, sp6_mcc2_sp2 = np.array([0.838, 0.894, 0.871]), np.array([0.841, 0.893, 0.719])
    sp6_mcc1_tat, sp6_mcc2_tat = np.array([0.946, 0.788, 0.802]), np.array([0.934, 0.806,0.807])

    print("tsignal", mcc_sp1, mcc_sp2, mcc_tat)
    print("sp6", sp6_mcc1_sp1, sp6_mcc1_sp2, sp6_mcc1_tat)
    print("tsignal",mcc2_sp1, mcc2_sp2, mcc2_tat)
    print("sp6",sp6_mcc2_sp1, sp6_mcc2_sp2, sp6_mcc2_tat)

    print("mcc1:",(np.sum(mcc_sp1*no_of_seqs_sp1) + np.sum(mcc_sp2*no_of_seqs_sp2) + np.sum(mcc_tat * no_of_seqs_tat) )/no_of_tested_sp_seqs)
    print("mcc2:", (np.sum(mcc2_sp1[1:]*no_of_seqs_sp1[1:]) + np.sum(mcc2_sp2*no_of_seqs_sp2) + np.sum(mcc2_tat * no_of_seqs_tat) )/(no_of_tested_sp_seqs-2040))
    exit(1)

def visualize_dot_products():
    import math
    max_len = 201
    d_model = 128
    pe = np.zeros((max_len, 1, d_model))
    position = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0, 0::2] = np.sin(position * div_term)
    pe[:, 0, 1::2] = np.cos(position * div_term)
    pe = pe.reshape(max_len, d_model)
    pos_i = 30
    dot_prod = pe @ pe[pos_i].T
    plt.plot(dot_prod)
    colors = ['black', 'red','blue','purple']
    for i in range(4):
        plt.plot([pos_i+i,pos_i+i], [min(dot_prod[pos_i-7:pos_i+7])*0.9, 1.1*max(dot_prod[pos_i-7:pos_i+7])], linewidth=0.8, alpha=1, label='pos i+/-{}'.format(i) if i != 0 else "pos i", color=colors[i-1])
        plt.plot([pos_i-i,pos_i-i], [min(dot_prod[pos_i-7:pos_i+7])*0.9, 1.1*max(dot_prod[pos_i-7:pos_i+7])], linewidth=0.8, alpha=1, color=colors[i-1])
    plt.legend()
    plt.xlim([pos_i-7,pos_i+7])
    plt.ylim([min(dot_prod[pos_i-7:pos_i+7])*0.9, 1.1*max(dot_prod[pos_i-7:pos_i+7])])
    plt.ylabel("Sine-based PE dot product")
    plt.xticks([])
    # plt.show()

    plt.savefig("dot_prod.pdf")
    exit(1)


def plot_og_vs_no_og(result_folders_no_og=("only_decoder_tune_bert_extraOhOnOut_swa_run_1",
                                     "only_decoder_tune_bert_extraOhOnOut_swa_run_2",),
                     result_folders_og=("run_w_og",)):
    sp1_f1s, sp1_recs, sp1_precs, sp2_f1s, sp2_recs, sp2_precs, tat_f1s, \
        tat_recs, tat_precs, mcc1_sp1, mcc2_sp1, mcc1_sp2, mcc2_sp2, mcc1_tat, mcc2_tat = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    sp1_f1s_og, sp1_recs_og, sp1_precs_og, sp2_f1s_og, sp2_recs_og, sp2_precs_og, tat_f1s_og, \
        tat_recs_og, tat_precs_og, mcc1_sp1_og, mcc2_sp1_og, mcc1_sp2_og, mcc2_sp2_og, mcc1_tat_og, mcc2_tat_og = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for run_foldername in result_folders_no_og:
        print("Computing results for run {}".format(run_foldername))
        mdl2results = extract_all_param_results(only_cs_position=False,
                                                result_folder=run_foldername,
                                                compare_mdl_plots=False,
                                                remove_test_seqs=False,
                                                benchmark=False,
                                                prints=False)
        mdl_ind = 0
        sp1_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-7])]))
        sp1_recs.append(np.array([rec for rec in mdl2results[mdl_ind][10]]))
        sp1_precs.append(np.array([rec for rec in mdl2results[mdl_ind][11]]))
        sp2_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-6])]))
        sp2_recs.append(np.array([rec for rec in mdl2results[mdl_ind][12]]))
        sp2_precs.append(np.array([rec for rec in mdl2results[mdl_ind][13]]))
        tat_f1s.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-5])]))
        tat_recs.append(np.array([rec for rec in mdl2results[mdl_ind][14]]))
        tat_precs.append(np.array([rec for rec in mdl2results[mdl_ind][15]]))
        mcc1_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][0]]))
        mcc2_sp1.append(np.array([mcc for mcc in mdl2results[mdl_ind][1][1:]]))
        mcc1_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][2]]))
        mcc2_sp2.append(np.array([mcc for mcc in mdl2results[mdl_ind][3]]))
        mcc1_tat.append(np.array([mcc for mcc in mdl2results[mdl_ind][4]]))
        mcc2_tat.append([mcc for mcc in mdl2results[mdl_ind][5]])
    arrange_tol_lg_sp1 = []
    # rearrange the results
    for og_ind in [0, 4, 8, 12]:
        for tol in range(4):
            arrange_tol_lg_ = [sp1_f1s[run_no][og_ind + tol] for run_no in range(len(result_folders_no_og))]
            arrange_tol_lg_sp1.append(arrange_tol_lg_)
    arrange_tol_lg_sp2 = []
    for og_ind in [0, 4, 8]:
        for tol in range(4):
            arrange_tol_lg_ = [sp2_f1s[run_no][og_ind + tol] for run_no in range(len(result_folders_no_og))]
            arrange_tol_lg_sp2.append(arrange_tol_lg_)
    arrange_tol_lg_tat = []
    for og_ind in [0, 4, 8]:
        for tol in range(4):
            arrange_tol_lg_ = [tat_f1s[run_no][og_ind + tol] for run_no in range(len(result_folders_no_og))]
            arrange_tol_lg_tat.append(arrange_tol_lg_)
    arrange_sptype_tol_lg = [np.array(arrange_tol_lg_sp1), np.array(arrange_tol_lg_sp2), np.array(arrange_tol_lg_tat)]
    all_sptypes_all_mean = [np.mean(arrange_sptype_tol_lg[0], axis=1), np.mean(arrange_sptype_tol_lg[1], axis=1),
                            np.mean(arrange_sptype_tol_lg[2], axis=1)]
    all_sptypes_all_std = [np.std(arrange_sptype_tol_lg[0], axis=1), np.std(arrange_sptype_tol_lg[1], axis=1),
                           np.std(arrange_sptype_tol_lg[2], axis=1)]

    for run_foldername in result_folders_og:
        print("Computing results for run {}".format(run_foldername))
        mdl2results = extract_all_param_results(only_cs_position=False,
                                                result_folder=run_foldername,
                                                compare_mdl_plots=False,
                                                remove_test_seqs=False,
                                                benchmark=False,
                                                prints=False)
        mdl_ind = 0
        sp1_f1s_og.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-7])]))
        sp1_recs_og.append(np.array([rec for rec in mdl2results[mdl_ind][10]]))
        sp1_precs_og.append(np.array([rec for rec in mdl2results[mdl_ind][11]]))
        sp2_f1s_og.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-6])]))
        sp2_recs_og.append(np.array([rec for rec in mdl2results[mdl_ind][12]]))
        sp2_precs_og.append(np.array([rec for rec in mdl2results[mdl_ind][13]]))
        tat_f1s_og.append(np.array([rec for rec in np.concatenate(mdl2results[mdl_ind][-5])]))
        tat_recs_og.append(np.array([rec for rec in mdl2results[mdl_ind][14]]))
        tat_precs_og.append(np.array([rec for rec in mdl2results[mdl_ind][15]]))
        mcc1_sp1_og.append(np.array([mcc for mcc in mdl2results[mdl_ind][0]]))
        mcc2_sp1_og.append(np.array([mcc for mcc in mdl2results[mdl_ind][1][1:]]))
        mcc1_sp2_og.append(np.array([mcc for mcc in mdl2results[mdl_ind][2]]))
        mcc2_sp2_og.append(np.array([mcc for mcc in mdl2results[mdl_ind][3]]))
        mcc1_tat_og.append(np.array([mcc for mcc in mdl2results[mdl_ind][4]]))
        mcc2_tat_og.append([mcc for mcc in mdl2results[mdl_ind][5]])
    arrange_tol_lg_sp1_wog = []
    # rearrange the results
    for og_ind in [0, 4, 8, 12]:
        for tol in range(4):
            arrange_tol_lg_og = [sp1_f1s_og[run_no][og_ind + tol] for run_no in range(len(result_folders_og))]
            arrange_tol_lg_sp1_wog.append(arrange_tol_lg_og)

    arrange_tol_lg_sp2_wog = []
    for og_ind in [0, 4, 8]:
        for tol in range(4):
            arrange_tol_lg_og = [sp2_f1s_og[run_no][og_ind + tol] for run_no in range(len(result_folders_og))]
            arrange_tol_lg_sp2_wog.append(arrange_tol_lg_og)


    arrange_tol_lg_tat_wog = []
    for og_ind in [0, 4, 8]:
        for tol in range(4):
            arrange_tol_lg_og = [tat_f1s_og[run_no][og_ind + tol] for run_no in range(len(result_folders_og))]
            arrange_tol_lg_tat_wog.append(arrange_tol_lg_og)

    arrange_sptype_tol_lg_wog = [np.array(arrange_tol_lg_sp1_wog), np.array(arrange_tol_lg_sp2_wog), np.array(arrange_tol_lg_tat_wog)]
    all_sptypes_all_mean_wog = [np.mean(arrange_sptype_tol_lg_wog[0], axis=1), np.mean(arrange_sptype_tol_lg_wog[1], axis=1),
                            np.mean(arrange_sptype_tol_lg_wog[2], axis=1)]
    all_sptypes_all_std_wog = [np.std(arrange_sptype_tol_lg_wog[0], axis=1), np.std(arrange_sptype_tol_lg_wog[1], axis=1),
                            np.std(arrange_sptype_tol_lg_wog[2], axis=1)]
    all_f1s_sp1 = [np.array(all_sptypes_all_mean[0]).reshape(-1),
                   np.array(all_sptypes_all_mean_wog[0]).reshape(-1)]
    all_f1s_sp2 = [np.array(all_sptypes_all_mean[1]).reshape(-1),
                   np.array(all_sptypes_all_mean_wog[1]).reshape(-1),]
    all_f1s_tat = [np.array(all_sptypes_all_mean[2]).reshape(-1),
                   np.array(all_sptypes_all_mean_wog[2]).reshape(-1)]
    all_sptypes_all_f1s = [all_f1s_sp1, all_f1s_sp2, all_f1s_tat]

    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 350
    mpl.rcParams['font.family'] = "Arial"
    fig, ax = plt.subplots(3, 1,figsize=(8, 6),dpi=350)
    line_w = 0.3
    offsets = [-line_w*0.5, line_w*0.5]
    sptypes=["Sec/SPase I", "Sec/SPase II", "Tat/SPase I"]
    names = ["TSignal", "TSignal_wog", "LipoP", "DeepSig", "Phobius"]
    colors = ["mediumblue", "green", "green", "black", "purple","red"]
    titles = ["", "", "", ""]


    for ind in range(3):
        upper_lim = 17 if ind == 0 else 13
        lower_lim = 0 if ind == 0 else 1
        lower_lim_plots = 1 if ind == 0 else 5
        all_f1s = all_sptypes_all_f1s[ind]
        ax[ind].plot([4.5,4.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        ax[ind].plot([8.5,8.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        ax[ind].plot([12.5,12.5], [0,1.5], linestyle='--',dashes=(1, 1), color='black')
        for j in range(2):
            ax[ind].bar([i + offsets[j] for i in range(lower_lim_plots, 17)], all_f1s[j],  label=names[j],
                        width=line_w,alpha=0.6, color=colors[j])
        # error plots (below, if std is zero - e.g. when having a single run - plot a "dot" from low-low+0.001)
        for i in range(lower_lim_plots, 17):

            low,high = all_sptypes_all_mean[ind][i - lower_lim_plots] - 2 * all_sptypes_all_std[ind][i - lower_lim_plots], \
                       all_sptypes_all_mean[ind][i - lower_lim_plots] + 2 *  all_sptypes_all_std[ind][i - lower_lim_plots]
            ax[ind].plot([i+offsets[0],i+offsets[0]],[low,max(high,low+0.001)], color='black')

            low, high = all_sptypes_all_mean_wog[ind][i - lower_lim_plots] - 2 * all_sptypes_all_std_wog[ind][i - lower_lim_plots], \
                        all_sptypes_all_mean_wog[ind][i - lower_lim_plots] + 2 * all_sptypes_all_std_wog[ind][i - lower_lim_plots]
            ax[ind].plot([i + offsets[1], i + offsets[1]], [low, max(high, low + 0.001)], color='black')

        box = ax[ind].get_position()
        ax[ind].set_xlim(0.5,16.5)
        ax[ind].set_position([box.x0, box.y0 + box.height * 0.35, box.width * 1, box.height * 0.95])
        ax[ind].set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax[ind].set_ylim([0, 1.1])
        ax[ind].grid(axis='y',color='black', linestyle='-', linewidth=0.5,alpha=0.4)
        ax[ind].set_xticks(list(range(lower_lim_plots, 17)))
        if ind == 2:
            handles, labels = ax[ind].get_legend_handles_labels()
        ax[ind].set_xticklabels(['{}{}'.format(titles[lower_lim + i//4], i%4) for i in range(upper_lim-1)], fontsize=12.5)
        ax[ind].set_ylabel("F1 score\n{}".format(sptypes[ind]), fontsize=12.5)
        ax[ind].yaxis.set_label_coords(-0.07, 0.42)
        ax[ind].set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=12.5)
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.0, 0.05), ncol=2, fontsize=12.5)
    fig.suppressComposite = False
    ax[2].set_xlabel("eukarya                 gn bacteria                  gp bacteria                 archaea\n\n "
                     "tolerance/life group",fontsize=12.5)

    plt.savefig("og_no_og_comparison.pdf")

    # repeat 4 times for each tolerance level
    # no_of_seqs_sp1 = np.array([146, 61, 15, 36]).repeat(4)
    # no_of_seqs_sp2 = np.array([257, 120, 9]).repeat(4)
    # no_of_seqs_tat = np.array([51, 18, 9]).repeat(4)
    # no_of_tested_sp_seqs = sum([146, 61, 15, 36]) + sum([257, 120, 9]) + sum([51, 18, 9])

    no_of_seqs_sp1 = np.array([2040, 44, 142, 356]).repeat(4)
    no_of_seqs_sp2 = np.array([1087, 516, 12]).repeat(4)
    no_of_seqs_tat = np.array([313, 39, 13]).repeat(4)
    no_of_tested_sp_seqs = sum([2040, 44, 142, 356]) + sum([1087, 516, 12]) + sum([313, 39, 13])

    print("Mean weighted F1 score across all SP types, organism groups and tolerance levels TSignal: ",
          (np.sum(all_sptypes_all_mean[0] * no_of_seqs_sp1) +
           np.sum(all_sptypes_all_mean[1] * no_of_seqs_sp2) +
           np.sum(all_sptypes_all_mean[2] * no_of_seqs_tat)) / np.sum(no_of_tested_sp_seqs * 4))
    print("Mean weighted F1 score across all SP types, organism groups and tolerance levels TSignal w_og: ",
          (np.sum(all_sptypes_all_mean_wog[0] * no_of_seqs_sp1) +
           np.sum(all_sptypes_all_mean_wog[1] * no_of_seqs_sp2) +
           np.sum(all_sptypes_all_mean_wog[2] * no_of_seqs_tat)) / np.sum(no_of_tested_sp_seqs * 4))



if __name__ == "__main__":


    # test F1 TSignal vs other models (Fig. 4 main text)
    plot_sp6_vs_tnmt(result_folders=("previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_1",
                                     "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_2",
                                     "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_3",
                                     "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_4",
                                     "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_5"))

    # test mcc TSignal vs other models (Fig. 3 main text)
    plot_sp6_vs_tnmt_mcc(result_folders=("previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_1",
                                     "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_2",
                                     "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_3",
                                     "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_4",
                                     "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_5"),benchmark=True)

    # plot F1-CS performance for TSignal vs all other models (Figure 5 main manuscript)
    plot_comparative_performance_sp1_mdls(result_folders=("previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_1",
                        "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_2",
                        "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_3",
                        "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_4",
                        "previous_results/only_decoder_tune_bert_extraOhOnOut_swa_run_5"))

    # input gradient visualization (Fig 6 main manuscript. Note: this uses sp6_partitioned_data_<fold1>_<fold2>_<train/test>.bin
    # files) - so first you need to follow README section 2.
    visualize_inp_gradients(files=["previous_results/added_oh_gen_fold0.bin", "added_oh_gen_fold1.bin", "added_oh_gen_fold2.bin"])

    # simple visualization for the sine-based dot products on various relative positions (Fig. 3 supplement)
    visualize_dot_products()


    # MCC performance on the whol data (since benchmark is set to False)
    plot_sp6_vs_tnmt_mcc(benchmark=False)


    # reproduce the ablation study between using and not using organism group information
    plot_og_vs_no_og(result_folders_no_og=("only_decoder_tune_bert_extraOhOnOut_swa_run_1",
                                           "only_decoder_tune_bert_extraOhOnOut_swa_run_2",
                                           "only_decoder_tune_bert_extraOhOnOut_swa_run_3",
                                           "only_decoder_tune_bert_extraOhOnOut_swa_run_4",
                                           "only_decoder_tune_bert_extraOhOnOut_swa_run_5"),
                     result_folders_og=("run_w_og", "run_w_og_run2", "run_w_og_run3", "run_w_og_run4", "run_w_og_run5"))
    # get whole data MCC scores for with-organism-group method
    plot_sp6_vs_tnmt_mcc(result_folders=("previous_results/run_w_og", "previous_results/run_w_og_run2", "previous_results/run_w_og_run3",
                        "previous_results/run_w_og_run4", "previous_results/run_w_og_run5"), benchmark=False)


    plot_perf_over_data_perc()


    # when training separately the SP identification task (this is not the final model variation presented in the manuscript)
    compute_mcc_sp_only_mdls()



    plot_compare_pos_nopos()
    lg_and_tol2_lg = extract_calibration_probs_for_mdl(model="repeat2_only_decoder_",
                                                       folder='tuning_bert_repeat2_only_decoder/',
                                                       plot=True)



    visualize_validation(run="repeat_only_decoder_", folds=[0, 2],
                         folder="tuning_bert_only_decoder_repeat/")

    plot_ece_over_tolerance(lg_and_tol2_lg)



    compute_diversity_within_partition()
